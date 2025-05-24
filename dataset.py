from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import os
import gc
#os.environ["OMP_NUM_THREADS"]="1" # set this env var before import pandas to prevent implicit parallelism in pandas(if there is no need to analyze data), refer to https://medium.com/affinityanswers-tech/why-parallelism-in-python-pandas-may-be-hurting-the-performance-of-your-programs-ea3e5485d0b7
import shutil
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import json
from tqdm import tqdm
from scipy import interpolate
from pandarallel import pandarallel
from py_vollib.black_scholes.implied_volatility import implied_volatility
import pdb
import config
from model import SingleModel, MultiModel
from benchmarks_fftoption import compute_call_price_by_parametric_model
from utils import filter, seed_initializer, seed_worker, release_memory, compute_func
from visulization import plot_relation_between_option_and_stirke
import torch
import random

# TODO: 20221029, residual model -> plot distribution
# TODO: check differences between different version of dataset

class DataProcessor:
    def __init__(self, root_path='./Dataset', model_type='multi', sample_start_date='19960101', sample_times=1, sample_interval=-1, on_gpu=False):
        self.root_path = root_path
        self.on_gpu = on_gpu
        self.model_type = model_type
        self.sample_start_date = sample_start_date[:4] + '-' + sample_start_date[4:6] + '-' + sample_start_date[6:8]
        self.sample_times = sample_times
        self.sample_interval = sample_interval
        self.partition_size_limit = "500MB"
        self.IAO_threshold = None
        self.residual_on = None
        self.seed_list = None
        self.window_size = None
        self.batch_size = None
        self.num_workers = None
        self.n_parallel_process = None
        self.learning_rate = None

    def __call__(self, window_size=5, batch_size=32, num_workers=0, n_parallel_process=1, residual_on=False, seed_list=[0, 1]):
        self.seed_list = seed_list
        self.residual_on = residual_on
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_parallel_process = n_parallel_process
        test_date_list, whole_date_list, syn_dataset_with_0strike_c6, syn_dataset_with_parametric_c6, prs_dataset = self._build_dataset()

        invm = prs_dataset['strike_price'] / prs_dataset['S']
        self.IAO_threshold = {
            # with supplement
            'range_ATM': [compute_func(invm.mean() - 2*invm.std()), compute_func(invm.mean() + 2*invm.std())],
            
            # without supplement
            # 'range_ATM': [1, 1],
            'deep_ITM': compute_func(invm.mean() - 2*invm.std()),
            'deep_OTM': compute_func(invm.mean() + 2*invm.std())
        }

        daily_gen = self._daily_data_generator(test_date_list, whole_date_list, syn_dataset_with_0strike_c6, syn_dataset_with_parametric_c6, prs_dataset)
        return test_date_list, daily_gen, self.IAO_threshold

    def check_data_about_option_price_not_monotonic_with_strike_price(self, dataset, type: str, is_put=False):
        # create blank folder
        result_folder = os.path.join(self.root_path, f"check_monotonic_between_call_and_strike_{type}")
        try:
            shutil.rmtree(result_folder)
        except FileNotFoundError:
            pass # fine
        os.makedirs(result_folder, exist_ok=True)

        # checking ...
        def func(specific_option):
            eps = 1
            if 'is_syn' in specific_option.columns:
                sorted_value = specific_option[['strike_price', 'option_price', 'volume', 'is_syn']].sort_values(by=['strike_price']).reset_index(drop=True)
                is_syn = sorted_value['is_syn'].values
            else:
                sorted_value = specific_option[['strike_price', 'option_price', 'volume']].sort_values(by=['strike_price']).reset_index(drop=True)
                is_syn = np.zeros(len(specific_option), dtype=bool)
            if is_put: # put option
                is_monotonic = (sorted_value['option_price'].shift(1) <= eps + sorted_value['option_price']).values[1:].all()
            else: # call option
                is_monotonic = (sorted_value['option_price'].shift(1) + eps >= sorted_value['option_price']).values[1:].all()

            specific_option['K_C_monotonic'] = is_monotonic
            if not is_monotonic:
                plot_relation_between_option_and_stirke(result_folder, str(specific_option['date'].iloc[0])[:10], str(specific_option['exdate'].iloc[0])[:10], sorted_value['option_price'].values, sorted_value['strike_price'].values, sorted_value['volume'].values, is_syn, specific_option.iloc[0]['S'])
            return specific_option
        checked_dataset = dataset.groupby(['date', 'exdate']).apply(func)
        checked_dataset[~checked_dataset['K_C_monotonic']].to_csv(f"{result_folder}/not_monotonic_{type}.csv", index=False)

    def _daily_data_generator(self, test_date_list, whole_date_list, syn_dataset_with_0strike_c6, syn_dataset_with_parametric_c6, prs_dataset):
        for test_date in test_date_list:
            # ----------Construct Data Loader---------- #
            # split ITM and OTM part
            train_dataloader_ITM_part = None
            val_dataloader_ITM_part = None
            test_dataloader_ITM_part = None
            train_dataloader_OTM_part = None
            val_dataloader_OTM_part = None
            test_dataloader_OTM_part = None

            if self.residual_on:
                # ITM part
                itm_part = filter(syn_dataset_with_parametric_c6['invm'], 'i', threshold=self.IAO_threshold['range_ATM'][1])
                train_dataset_ITM_part = OptionDataset(syn_dataset_with_parametric_c6[itm_part], 'train', test_date, whole_date_list, self.window_size)
                val_dataset_ITM_part = OptionDataset(syn_dataset_with_parametric_c6[itm_part], 'val', test_date, whole_date_list, self.window_size)
                test_dataset_ITM_part = OptionDataset(syn_dataset_with_parametric_c6[itm_part], 'test', test_date, whole_date_list, self.window_size)

                train_dataloader_ITM_part = DataLoader(train_dataset_ITM_part, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
                val_dataloader_ITM_part = DataLoader(val_dataset_ITM_part, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
                test_dataloader_ITM_part = DataLoader(test_dataset_ITM_part, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

                # OTM part
                otm_part = filter(syn_dataset_with_0strike_c6['invm'], 'o', threshold=self.IAO_threshold['range_ATM'][0])
                train_dataset_OTM_part = OptionDataset(syn_dataset_with_0strike_c6[otm_part], 'train', test_date, whole_date_list, self.window_size)
                val_dataset_OTM_part = OptionDataset(syn_dataset_with_0strike_c6[otm_part], 'val', test_date, whole_date_list, self.window_size)
                test_dataset_OTM_part = OptionDataset(syn_dataset_with_0strike_c6[otm_part], 'test', test_date, whole_date_list, self.window_size)

                train_dataloader_OTM_part = DataLoader(train_dataset_OTM_part, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
                val_dataloader_OTM_part = DataLoader(val_dataset_OTM_part, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
                test_dataloader_OTM_part = DataLoader(test_dataset_OTM_part, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

            # both ITM and OTM part
            train_dataset = OptionDataset(syn_dataset_with_0strike_c6, 'train', test_date, whole_date_list, self.window_size)
            val_dataset = OptionDataset(syn_dataset_with_0strike_c6, 'val', test_date, whole_date_list, self.window_size)
            test_dataset = OptionDataset(syn_dataset_with_0strike_c6, 'test', test_date, whole_date_list, self.window_size)
            pdf_dataset = OptionDataset(prs_dataset, 'pdf', test_date)

            # accord. to torch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html, https://towardsdatascience.com/random-seeds-and-reproducibility-933da79446e3
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, worker_init_fn=seed_worker)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, worker_init_fn=seed_worker)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, worker_init_fn=seed_worker)
            pdf_dataloader = DataLoader(pdf_dataset, batch_size=1, num_workers=self.num_workers)

            def device_setting_func(rank):
                modell = []
                for seed in self.seed_list:
                    # reset rand seed since rand seed will affect model initialization and data shuffle
                    # notice that shuffle occurs when dataloader is called. the data itself is not shuffled. refer to https://stackoverflow.com/questions/61115032/pytorch-dataloader-shuffle
                    # hence reset dataloader's rand seed at the beginning of training instead of here.
                    # that is, only reset model initialization's rand seed here
                    seed_initializer(seed)
                    # set device
                    if self.n_parallel_process == 1:  # single-process
                        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and self.on_gpu else "cpu")
                    else:  # multi-process
                        device = torch.device(rank if torch.cuda.is_available() and self.on_gpu else "cpu")
                    # set model
                    if self.model_type == "multi":  # multi-model
                        aaai17_model = MultiModel(J=5, I=9, K=5, device=device).to(device)
                        residual_itm_model = MultiModel(J=5, I=9, K=5, device=device, residual_model_call_itm_part=True).to(device)
                    elif self.model_type == "single":  # single-model
                        aaai17_model = SingleModel(J=5, device=device).to(device)
                        residual_itm_model = SingleModel(J=5, I=9, K=5, device=device, residual_model_call_itm_part=True).to(device)
                    else:
                        raise NotImplementedError
                    # if args.n_parallel_process != 1: # multi-process
                    #    nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
                    modell.append({'aaai17': aaai17_model, 'residual_itm': residual_itm_model, 'residual_otm': aaai17_model})
                return test_date, modell, train_dataloader_ITM_part, train_dataloader_OTM_part, train_dataloader, val_dataloader_ITM_part, val_dataloader_OTM_part,  val_dataloader, test_dataloader_ITM_part, test_dataloader_OTM_part, test_dataloader, pdf_dataloader

            yield device_setting_func

    def _build_dataset(self):
        prs_dataset = self._preprocess(drop0volume=True)
        prs_dataset_involve0volume = self._preprocess(drop0volume=False)
        syn_dataset_with_0strike_c6, syn_dataset_with_parametric_c6 = self._synthesize(prs_dataset, prs_dataset_involve0volume)
        # roll_dataset = rolling_ma(syn_dataset_with_0strike_c6)

        # to prevent RuntimeError: expected scalar type Double but found Float
        for col in ['S', 'strike_price']:
            prs_dataset[col] = prs_dataset[col].astype(np.float64)
            syn_dataset_with_0strike_c6[col] = syn_dataset_with_0strike_c6[col].astype(np.float64)
            syn_dataset_with_parametric_c6[col] = syn_dataset_with_parametric_c6[col].astype(np.float64)

        whole_date_list = np.sort(prs_dataset['date'].unique())  # from early date to later date to further pick the training trading day range
        start_sample_idx = max(0, np.where(whole_date_list >= np.datetime64(self.sample_start_date))[0][0] - self.window_size)
        sample_idx = np.zeros(len(whole_date_list), dtype=bool)
        sample_step = len(whole_date_list[start_sample_idx::self.sample_interval]) / self.sample_times
        for i in range(self.sample_times):
            start_pos = start_sample_idx + self.window_size + int(i*sample_step) * self.sample_interval
            end_pos = start_pos + self.sample_interval
            end_pos = None if end_pos == -1 else end_pos
            sample_idx[start_pos:end_pos] = True #start~end-1 = True
        test_date_list = whole_date_list[sample_idx]

        return test_date_list, whole_date_list, syn_dataset_with_0strike_c6, syn_dataset_with_parametric_c6, prs_dataset

    def _preprocess(self, drop0volume=True):
        
        # with syn. data
        result_path = f"{self.root_path}/prs_dataset_252tau.csv/part0.csv" if drop0volume else f"{self.root_path}/prs_dataset_involve0volume_252tau.csv"
        
        # without syn. data
        # result_path = f"{self.root_path}/prs_dataset_252tau.csv/2020.csv" if drop0volume else f"{self.root_path}/prs_dataset_involve0volume_252tau.csv/part.4.csv"
        
        # result_path = f"{self.root_path}/prs_dataset_252tau_20221202.csv" if drop0volume else f"{self.root_path}/prs_dataset_involve0volume_252tau_20221202.csv" # TODO: delete

        check_monotonic = False # TODO: recover
        
        if os.path.exists(result_path):
            check_monotonic = False
            if os.path.isdir(result_path):
                prs_dataset = dd.read_csv(result_path+"/*" if os.path.isdir(result_path) else result_path, parse_dates=['date', 'exdate'], blocksize=self.partition_size_limit)
            else:
                prs_dataset = pd.read_csv(result_path if os.path.isdir(result_path) else result_path, parse_dates=['date', 'exdate'])
            prs_dataset['is_syn'] = False
            return prs_dataset
        
        # read option contracts
        option_path = f"{self.root_path}/options_1996_2021.csv"
        raw_options = dd.read_csv(option_path, dtype=str, blocksize=self.partition_size_limit)
        prs_dataset = raw_options[['date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'impl_volatility', 'volume']]
        for date_col in ['date', 'exdate']:
            prs_dataset = prs_dataset.assign(**{date_col: dd.to_datetime(prs_dataset[date_col], format='%Y%m%d')})
        for num_col in ['strike_price', 'best_bid', 'best_offer', 'impl_volatility', 'volume']:
            prs_dataset = prs_dataset.assign(**{num_col: dd.to_numeric(prs_dataset[num_col], errors='coerce')})
        # strike price
        prs_dataset = prs_dataset.assign(strike_price=prs_dataset['strike_price']/1000) # because the strike price from Option Metric has been mutiplied by 1000, we return it to the original price

        # drop the option contract with 0 volumn
        if drop0volume:
            discard = prs_dataset['volume'] == 0
            prs_dataset = prs_dataset[~discard]

        # closing option price = the mid price of bid price and ask price
        value = ((prs_dataset['best_bid'] + prs_dataset['best_offer'])/2).values
        prs_dataset = prs_dataset.assign(option_price = value)

        # time to maturity (tau) = (expiration date - quote date), and then annually normalize it
        # value = ((prs_dataset['exdate'] - prs_dataset['date']).dt.days).values
        # prs_dataset = prs_dataset.assign(tau = value/365.0)
        value = da.map_blocks(np.busday_count, prs_dataset['date'].values.astype('datetime64[D]'), prs_dataset['exdate'].values.astype('datetime64[D]'))
        prs_dataset = prs_dataset.assign(tau = value/252.0)

        # inverse moneyness = strike price / underlying asset price
        raw_underlyings = pd.read_csv(f"{self.root_path}/underlyings.csv")
        prs_underlyings = raw_underlyings[['Date', 'Close']]
        prs_underlyings = prs_underlyings.rename(columns={'Date':'date', 'Close':'S'})
        prs_underlyings = prs_underlyings.assign(date = pd.to_datetime(prs_underlyings['date'], format='%Y-%m-%d'))
        prs_dataset = prs_dataset.merge(prs_underlyings, how='inner', on='date')
        value = (prs_dataset['strike_price'] / prs_dataset['S']).values
        prs_dataset = prs_dataset.assign(invm=value)

        if drop0volume and check_monotonic:
            self.check_data_about_option_price_not_monotonic_with_strike_price(prs_dataset[prs_dataset['cp_flag'] == 'C'], "market_call_option")
            self.check_data_about_option_price_not_monotonic_with_strike_price(prs_dataset[(prs_dataset['cp_flag'] == 'C') & (prs_dataset['S'] < prs_dataset['strike_price'])], "market_call_otm_option")
            self.check_data_about_option_price_not_monotonic_with_strike_price(prs_dataset[prs_dataset['cp_flag'] == 'P'], "market_put_option", is_put=True)
            self.check_data_about_option_price_not_monotonic_with_strike_price(prs_dataset[(prs_dataset['cp_flag'] == 'P') & (prs_dataset['S'] > prs_dataset['strike_price'])], "market_put_otm_option", is_put=True)

        # discard in-the-money option quotes i.e. call options' inverse (moneyness = K/S) < 1, put options' inverse (moneyness = K/S) > 1

        discard = ((prs_dataset['invm'] < 1) & (prs_dataset['cp_flag'] == 'C')) | ((prs_dataset['invm'] > 1) & (prs_dataset['cp_flag'] == 'P'))
        prs_dataset = prs_dataset[~discard]
        prs_dataset = prs_dataset.reset_index(drop=True)

        # omit contracts with maturity less than 2 days
        discard = (prs_dataset['tau'] < (2.0/252.0))
        prs_dataset = prs_dataset[~discard]
        prs_dataset = prs_dataset.reset_index(drop=True)

        # interpolate risk free rate with cubic spline to match the option maturity
        # process risk free rate from Federal Reserve Board
        # raw_risk_free_rate = pd.read_csv(f"{self.root_path}/risk_free_rate_constantMaturityTreasury.csv")
        # prs_risk_free_rate = raw_risk_free_rate.drop(index=[0, 1, 2, 3, 4]).reset_index(drop=True) # drop description
        # prs_risk_free_rate = prs_risk_free_rate.rename(columns={'Series Description':'date'})
        # prs_risk_free_rate.iloc[:, 1:] = prs_risk_free_rate.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0) # convert string to numeric except for the date column
        # prs_risk_free_rate = prs_risk_free_rate.assign(date=pd.to_datetime(prs_risk_free_rate['date'], format='%Y-%m-%d'))
        # bond_maturity = [30.0 / 365.0, 90.0 / 365.0, 180.0 / 365.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        # r_index = prs_risk_free_rate.columns.str.contains('Market yield')  # market yield index
        # prs_risk_free_rate = prs_risk_free_rate.T
        # prs_dataset['r'] = 0.0
        # def cubic_func(today_risk_free_rate_data):
        #     date = today_risk_free_rate_data['date']
        #     target_idx = (prs_dataset['date'] == date)
        #     if True in target_idx.values:
        #         target_x = prs_dataset[target_idx]['tau']
        #         target_y = interpolate.splev(target_x, interpolate.splrep(bond_maturity, today_risk_free_rate_data[r_index]))
        #         prs_dataset.loc[target_idx, 'r'] = target_y
        # prs_risk_free_rate.apply(cubic_func)
        # prs_dataset['r'] = prs_dataset['r'] * 0.01  # because of the percentage format

        # process risk free rate from Option Metric
        raw_risk_free_rate = pd.read_csv(f"{self.root_path}/risk_free_rate_from_option_metric.csv")
        prs_risk_free_rate = pd.DataFrame()
        raw_risk_free_rate.iloc[:, 1:] = raw_risk_free_rate.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
        # prs_risk_free_rate is for true_x and true_y. don't apply dask dataframe to it to make sure interpolate work appropriate
        prs_risk_free_rate['date'] = pd.to_datetime(raw_risk_free_rate['date'], format='%Y%m%d')
        prs_risk_free_rate['years'] = raw_risk_free_rate['days'] / 360.0
        prs_risk_free_rate['rate'] = raw_risk_free_rate['rate']
        prs_dataset['r'] = 0.0
        def cubic_func(date, prs_single_df, prs_risk_free_rate):
            true_idx = prs_risk_free_rate['date'] == date
            if sum(true_idx) > 0:
                true_x = prs_risk_free_rate.loc[true_idx, 'years']
                true_y = prs_risk_free_rate.loc[true_idx, 'rate']
                target_x = prs_single_df['tau']
                target_y = interpolate.splev(target_x, interpolate.splrep(true_x, true_y))
                prs_single_df['r'] = target_y * 0.01 # because of the percentage format
            return prs_single_df
        prs_dataset = prs_dataset.map_partitions(lambda part_df:\
                                   part_df.groupby(['date'], group_keys=False).apply(\
                                       lambda x: cubic_func(x.name, x, prs_risk_free_rate)), \
                                       meta=pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in prs_dataset.dtypes.to_dict().items()}))

        # deal with dividend rate according to the nearest "past" 1 year dividend rate
        # raw_dividend_rate = pd.read_csv(f"{self.root_path}/dividend_rate.csv")
        # prs_dividend_rate = raw_dividend_rate.rename(columns={'Date': 'date', 'Value':'d'})
        # prs_dividend_rate = prs_dividend_rate.assign(date=pd.to_datetime(prs_dividend_rate['date'], format='%Y-%m-%d'))
        # prs_dataset = pd.merge_asof(prs_dataset, prs_dividend_rate.sort_values('date'), on='date', direction='backward', tolerance=pd.Timedelta('1 y'))
        # prs_dataset['d'] = prs_dataset['d']*0.01 #because of the percentage format

        raw_dividend_rate = pd.read_csv(f"{self.root_path}/dividend_rate_from_option_metric.csv")
        prs_dividend_rate = raw_dividend_rate[['date', 'rate']].rename(columns={'rate': 'd'})
        prs_dividend_rate = prs_dividend_rate.assign(date=pd.to_datetime(prs_dividend_rate['date'], format='%Y%m%d'))
        prs_dataset = dd.merge(prs_dataset, prs_dividend_rate, on=['date'])
        prs_dataset = prs_dataset.assign(d=prs_dataset['d'] * 0.01) # because of the percentage format

        # TODO:infer forward price of underling asset by put-call parity; 隱憂: 0 volume contracts' price is not fresh <- hence usaully use at the money option data to infer (also because of (bid + ask)/2 may not have atbitrage-free property)
        # TODO: change put-call "future" parity from C = P+S_0-Ke^{-r\tau} to C = P+Fe^{-r\tau}-Ke^{-r\tau}; download underlying asset future price, 記得future到期日要對到option 到期日
        # transfer put option prices into call prices through put-call parity (C = P+S_0*e^{-\delta\tau}-Ke^{-r\tau}) rather than discarding all put prices
        # put option's corresponding call option will have the same underlying asset, quote date, expired date, strike price
        put_index = (prs_dataset['cp_flag'] == 'P')
        put_parity = prs_dataset['option_price'] + prs_dataset['S']*(np.exp(-prs_dataset['d']*prs_dataset['tau'])) - prs_dataset['strike_price']*np.exp(-prs_dataset['r']*prs_dataset['tau'])
        # prs_dataset.loc[put_index, 'option_price'] = prs_dataset[put_index]['option_price'] + prs_dataset[put_index]['S']*(np.exp(-prs_dataset[put_index]['d']*prs_dataset[put_index]['tau'])) - prs_dataset[put_index]['strike_price']*np.exp(-prs_dataset[put_index]['r']*prs_dataset[put_index]['tau'])
        prs_dataset['option_price'] = prs_dataset['option_price'].mask(put_index, put_parity)
        prs_dataset['cp_flag'] = prs_dataset['cp_flag'].mask(put_index, 'C_parity')

        prs_dataset = prs_dataset.assign(is_syn=False)
        prs_dataset = prs_dataset[['date', 'exdate', 'cp_flag', 'option_price', 'strike_price', 'invm', 'tau', 'S', 'r', 'd', 'impl_volatility', 'volume', 'is_syn']]
        prs_dataset = prs_dataset[prs_dataset.count(1) >= len(prs_dataset.columns)] # dropna
        prs_dataset = prs_dataset.reset_index(drop=True)

        if drop0volume and check_monotonic:
            self.check_data_about_option_price_not_monotonic_with_strike_price(prs_dataset, 'prs_dataset')

        name_function = lambda x: f"part.{x}.csv"
        for i, part_df in enumerate(prs_dataset.partitions):
            part_df.to_csv(f"{result_path}/{name_function(i)}", single_file=True, index=False)
            release_memory()
        return prs_dataset

    def _synthesize(self, prs_dataset, prs_dataset_involve0volume):
        # TODO: calc the implied volatility of synthetic data
        
        # with syn. data
        result_c5_path = f"{self.root_path}/syn_c5_dataset_252tau.csv"
        result_c6_by_0strike_path = f"{self.root_path}/syn_c6_dataset_by_0strike_252tau.csv"
        result_c6_by_parametric_path = f"{self.root_path}/syn_c6_dataset_by_parametric_252tau.csv"
        
        # without syn. data
        # result_c5_path = ""
        # result_c6_by_0strike_path = ""
        # result_c6_by_parametric_path = ""
        
        check_monotonic = False
        syn_min_K = 10
        # C5: for every unique S_t, fix tau = 0, uniformly sample K in [0,S_t], and the option price should be exactly S_t-K
        if result_c5_path == "":
            C5_dataset = None
        elif os.path.exists(result_c5_path):
            C5_dataset = pd.read_csv(result_c5_path, parse_dates=['date', 'exdate'])
        else:
            check_monotonic = True
            eps = 1e-31
            interval = 10

            unique_S_dataset = compute_func(prs_dataset.drop_duplicates(subset = ['S'], ignore_index=True))
            unique_S_dataset = unique_S_dataset.assign(tau=0.0)

            # 2 minutes, K in [0, S_t]
            C5_dataset = pd.DataFrame(columns=unique_S_dataset.columns).astype(unique_S_dataset.dtypes)
            for i in range(len(unique_S_dataset)):
                row = unique_S_dataset.iloc[i]
                syn_k = np.arange(syn_min_K, row['S'], interval)
                df = pd.DataFrame({k:[row.to_dict()[k]]*syn_k.shape[0] for k in row.to_dict()}) # eps to prevent it is divided with no remainder
                df['exdate'] = df['date']
                df['strike_price'] = syn_k
                df['invm'] = df['strike_price']/row['S']
                df['option_price'] = df['S'] - df['strike_price']  # call option price = S-K; due to zero time to maturity, there is no discount factor
                df['impl_volatility'] = np.nan
                C5_dataset = pd.concat([C5_dataset, df])

            # 6 hours, K in existing market strike price range
            # C5_dataset = pd.DataFrame(columns=prs_dataset.columns).astype(prs_dataset.dtypes)
            # for S in tqdm(unique_S_dataset['S']):
            #     df = compute_func(prs_dataset[prs_dataset['S'] == S])
            #     df = df.drop_duplicates(subset = ['strike_price'], ignore_index=True) # existing strike price
            #     df = df[df['strike_price'] <= S] # synthesize ITM options
            #     df['tau'] = 0.0
            #     df['exdate'] = df['date']
            #     df['invm'] = df['strike_price']/S
            #     df['option_price'] = df['S'] - df['strike_price']  # call option price = S-K; due to zero time to maturity, there is no discount factor
            #     C5_dataset = pd.concat([C5_dataset, df])

            # 1 minutes, K in existing market strike price range
            # def c5_func(df):
            #     S = df['S'].iloc[0]
            #     df = df.drop_duplicates(subset = ['strike_price'], ignore_index=True) # existing strike price
            #     df = df[df['strike_price'] <= S] # synthesize ITM options
            #     df['tau'] = 0.0
            #     df['exdate'] = df['date']
            #     df['invm'] = df['strike_price']/S
            #     df['option_price'] = df['S'] - df['strike_price']  # call option price = S-K; due to zero time to maturity, there is no discount factor
            #     return df
            # C5_dataset = prs_dataset.groupby(['S']).apply(c5_func, meta=pd.DataFrame(columns=prs_dataset.columns))

            C5_dataset = compute_func(C5_dataset.assign(is_syn=True))
            C5_dataset.to_csv(result_c5_path, index=False)

        # C6: for every unique tau, create an option with K=0 corresponding to the most expensive option
        # notice that date1, call=P1, K=0, tau=T, S=P1 vs. date2, call=P2, K=0, tau=T, S=P2 are both should be include since you can’t surely (leave P1, drop P2) or (drop P1, leave P2). P1 and P2 are both possible values. hence unique subset not only involve tau, but also the date.
        # this is for aaai17 multi (call itm + call otm) and residual call otm part
        if result_c6_by_0strike_path == "":
            C6_dataset_by_0strike = None
        elif os.path.exists(result_c6_by_0strike_path):
            C6_dataset_by_0strike = pd.read_csv(result_c6_by_0strike_path, parse_dates=['date', 'exdate'])
        else:
            check_monotonic = True
            C6_dataset_by_0strike = compute_func(prs_dataset.drop_duplicates(subset=['date', 'exdate'], ignore_index=True))
            C6_dataset_by_0strike = C6_dataset_by_0strike.assign(strike_price=syn_min_K)
            C6_dataset_by_0strike = C6_dataset_by_0strike.assign(invm=C6_dataset_by_0strike['strike_price'] / C6_dataset_by_0strike['S'])
            C6_dataset_by_0strike = C6_dataset_by_0strike.assign(option_price = C6_dataset_by_0strike['S'] - C6_dataset_by_0strike['strike_price']*np.exp(-C6_dataset_by_0strike['r']*C6_dataset_by_0strike['tau'])) # expensive call option: call option's upper bound is its underlying asset's price
            # put = C6_dataset_by_0strike['option_price'] - C6_dataset_by_0strike['S']*np.exp(-C6_dataset_by_0strike['d']*C6_dataset_by_0strike['tau']) + C6_dataset_by_0strike['strike_price']*np.exp(-C6_dataset_by_0strike['r']*C6_dataset_by_0strike['tau'])
            C6_dataset_by_0strike = C6_dataset_by_0strike.assign(is_syn=True)
            C6_dataset_by_0strike = C6_dataset_by_0strike.assign(impl_volatility=np.nan)
            C6_dataset_by_0strike.to_csv(result_c6_by_0strike_path, index=False)

        # alternative C6: retain 0 volume option contracts, for every quote date and expired date, extract the one with max moneyness, replace the option market price with 0 volume by calibrated price from parametric model
        # this is for residual call itm part
        if result_c6_by_parametric_path == "":
            C6_dataset_by_parametric = None
        elif os.path.exists(result_c6_by_parametric_path):
            C6_dataset_by_parametric = pd.read_csv(result_c6_by_parametric_path, parse_dates=['date', 'exdate'])
        else:
            # for every (qdate, exdate), find min market strike price, then apply parametric model (since 0 volume market data is not fresh)
            check_monotonic = True
            itm_part = filter(prs_dataset_involve0volume['invm'], 'i')

            # market_strke_price_lower_bound_idx = prs_dataset_involve0volume[itm_part].groupby(['date', 'exdate'])['strike_price'].idxmin()
            # C6_dataset_by_parametric = prs_dataset_involve0volume.loc[market_strke_price_lower_bound_idx.values].reset_index(drop=True)
            def get_row_with_min_strike_price_for_part_df(date_exdate, data):
                return data.loc[data['strike_price'].idxmin()] # is loc, not iloc, refer to https://stackoverflow.com/questions/67002499/how-can-i-get-the-index-of-smallest-value-in-series
            C6_dataset_by_parametric = prs_dataset_involve0volume[itm_part].map_partitions(lambda part_df:\
                                   part_df.groupby(['date', 'exdate'], group_keys=False).apply(\
                                       lambda x: get_row_with_min_strike_price_for_part_df(x.name, x)), \
                                       meta=pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in prs_dataset_involve0volume.dtypes.to_dict().items()}))
            C6_dataset_by_parametric = compute_func(C6_dataset_by_parametric)
            # (date1, exdate1) may both exist in part_df1 and part_df2. we need to pick the one with the lowest strike price
            duplicated_candidates_idx = C6_dataset_by_parametric.index.isin(C6_dataset_by_parametric.index[C6_dataset_by_parametric.index.duplicated()])
            duplicated_candidates = C6_dataset_by_parametric.loc[duplicated_candidates_idx].reset_index(drop=True)
            picked_candidates = duplicated_candidates.loc[duplicated_candidates.groupby(['date', 'exdate'])['strike_price'].idxmin()]
            C6_dataset_by_parametric = pd.concat([C6_dataset_by_parametric[~duplicated_candidates_idx].reset_index(drop=True), picked_candidates]).reset_index(drop=True) # drop duplicated ones and concat picked ones

            parametric_model_calibrated_dataset_path = f"{self.root_path}/kj_calibrated_dataset.csv"
            if not os.path.exists(parametric_model_calibrated_dataset_path):
                print(f"\"{parametric_model_calibrated_dataset_path}\" doesn't exist, please run run_parametric_method.py first, then move the calibrated dataframe to \"{self.root_path}\"")
                print("furthermore, it is recommend to run the calibration process parallelly, or running with single processor may consume you about a-half month to finish calibration")
                raise NotImplementedError
            parametric_model_calibrated_dataset = pd.read_csv(parametric_model_calibrated_dataset_path, parse_dates=['date'])
            C6_dataset_by_parametric = C6_dataset_by_parametric.merge(parametric_model_calibrated_dataset[['date', 'calib_params']], on='date')
            # about 10 minutes
            def func(specific_option):
                # replace un-fresh max moneyness option contract's market price by the calibrated price from parametric model
                calib_params = json.loads(specific_option['calib_params'])
                calib_option_price = compute_call_price_by_parametric_model(2 ** 15, 0.01, 1, specific_option['S'], specific_option['r'], specific_option['d'], specific_option['tau'], specific_option['strike_price'], 'kj', *calib_params).item()
                specific_option['option_price'] = calib_option_price
                return specific_option

            volume0part = (C6_dataset_by_parametric['volume'] == 0)
            pandarallel.initialize()
            C6_dataset_by_parametric[volume0part] = C6_dataset_by_parametric[volume0part].T.parallel_apply(func).T
            C6_dataset_by_parametric = C6_dataset_by_parametric.assign(impl_volatiltiy=np.nan)
            C6_dataset_by_parametric.drop(columns=['calib_params'], inplace=True)
            C6_dataset_by_parametric = C6_dataset_by_parametric.assign(is_syn=True)
            C6_dataset_by_parametric.to_csv(result_c6_by_parametric_path, index=False)

        concat_func = dd.concat if 'dask' in str(type(prs_dataset)) else pd.concat
        syn_dataset_with_0strike_c6 = concat_func([prs_dataset, C5_dataset, C6_dataset_by_0strike], axis=0, ignore_index=True)
        syn_dataset_with_parametric_c6 = concat_func([prs_dataset, C5_dataset, C6_dataset_by_parametric], axis=0, ignore_index=True)

        check_monotonic = False # TODO: delete

        if check_monotonic:
            self.check_data_about_option_price_not_monotonic_with_strike_price(syn_dataset_with_0strike_c6, 'syn_dataset_with_0strike_c6')
            self.check_data_about_option_price_not_monotonic_with_strike_price(syn_dataset_with_parametric_c6, 'syn_dataset_with_parametric_c6')
        return syn_dataset_with_0strike_c6, syn_dataset_with_parametric_c6

    def _rolling_ma(self, prs_dataset, rolling_size=1):
        # construct rolling window dataset by moving average
        # actually we use date 4/2 underlying asset close price to price date 4/2 call option close price instead of using date 4/1 underlying asset close price
        # hence in this project, we don't use roll_dataset

        # rolling window, rolling_size means the shift time interval
        # row: [day t] option_price, tau, date; [day t-5, t-4, t-3, t-2, t-1] moving average of  S, invm, r, and d with same strike price, expiration date (not tau, tau will change when one day pass)
        roll_dataset = prs_dataset.sort_values(by=['exdate', 'strike_price', 'date']).reset_index(drop=True)

        #unique_exdate_strike_combination_set = roll_dataset.groupby(['exdate', 'strike_price'])
        #for contract, specific_option in tqdm(unique_exdate_strike_combination_set):
        #    specific_option_rolled = specific_option.rolling(window=rolling_size).mean()[['invm', 'S', 'r', 'd']].shift(1).replace(np.nan,-1)
        #    roll_dataset.update(specific_option_rolled)
        # ... 6 hours ... QQ

        def func(specific_option):
            return specific_option.rolling(window=rolling_size).mean()[['invm', 'S', 'r', 'd']].shift(1).replace(np.nan,-1) # shfit 1 day to store (day t-5 ~ day t-1)'s feature (invm, S, r, d) to as (day t)'s feature, while (day t)'s other features (date(on day t), exdate(same), option_price(on day t), strike_price(same), tau(on day t, not t-1)) remain unchanged
        rolled = roll_dataset.groupby(['exdate', 'strike_price']).apply(func)
        roll_dataset.update(rolled)
        # ... 6 minutes ...

        roll_dataset.replace(-1, np.nan, inplace=True)
        roll_dataset.dropna(inplace=True)
        roll_dataset.reset_index(drop=True, inplace=True)
        roll_dataset.to_csv(f"{self.root_path}/roll_dataset.csv", index=False)  # save
        return roll_dataset

class OptionDataset(data.Dataset):
    def __init__(self, dataset, mode='train', rolling_test_date=np.datetime64('1996-04-09'), date_list=None, window_size=5):
        date_list = np.sort(date_list) if date_list is not None else date_list
        if mode == 'train':
            rolling_train_day_start = date_list[np.where(date_list == rolling_test_date)[0].item() - (window_size)]
            rolling_train_day_end = date_list[np.where(date_list == rolling_train_day_start)[0].item() + window_size - 1]
            index = (dataset['date'] >= rolling_train_day_start) & (dataset['date'] < rolling_train_day_end)
            tar_df = (dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd', 'is_syn']] if 'is_syn' in dataset.columns else dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd']])
            # tar_df = (dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd', 'impl_volatility', 'is_syn']] if 'is_syn' in dataset.columns else dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd']])

        elif mode == 'val':
            unique_date = compute_func(dataset['date'].unique())
            index = (dataset['date'] == (unique_date[np.where((unique_date == np.datetime64(rolling_test_date)))[0].item() - 1]))
            tar_df = (dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd', 'is_syn']] if 'is_syn' in dataset.columns else dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd']])
            # tar_df = (dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd', 'impl_volatility', 'is_syn']] if 'is_syn' in dataset.columns else dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd']])

        elif mode == 'test':
            index = (dataset['date'] == rolling_test_date)
            tar_df = (dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd', 'is_syn']] if 'is_syn' in dataset.columns else dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd']])
            # tar_df = (dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd', 'impl_volatility', 'is_syn']] if 'is_syn' in dataset.columns else dataset[index][['option_price', 'strike_price', 'tau', 'S', 'r', 'd']])

        elif mode == 'pdf':
            # same date (corresponding to same underlying asset price, dividend rate)
            # same time to maturity not yet (corresponding to same risk free rate)
            rolling_test_2ndexdate = compute_func((dataset[dataset['date'] == rolling_test_date])['exdate'].unique()[1])
            
            # test
            if not isinstance(rolling_test_2ndexdate, pd._libs.tslibs.timestamps.Timestamp):
               rolling_test_2ndexdate = rolling_test_2ndexdate.item()
            
            index = (dataset['date'] == rolling_test_date) & (dataset['exdate'] == rolling_test_2ndexdate)
            index = index & (dataset['is_syn'] == False)
            tar_df = (dataset[index][['strike_price', 'tau', 'S', 'r', 'd', 'impl_volatility', 'volume']].sort_values(by='strike_price'))
        elif mode == 'CDI':
            # same date (corresponding to same underlying asset price, dividend rate)
            # same time to maturity not yet (corresponding to same risk free rate)
            rolling_test_exdate_list = compute_func((dataset[dataset['date'] == rolling_test_date])['exdate'].unique())

            # test
            tar_df = []
            for rolling_test_exdate in rolling_test_exdate_list:
                if not isinstance(rolling_test_exdate, pd._libs.tslibs.timestamps.Timestamp):
                   rolling_test_exdate = rolling_test_exdate.item()
                
                index = (dataset['date'] == rolling_test_date) & (dataset['exdate'] == rolling_test_exdate)
                #取同日同到期日的option(同tau)
                index = index & (dataset['is_syn'] == False)
                #tar_df = (dataset[index][['strike_price', 'tau', 'S', 'r', 'd', 'impl_volatility', 'volume']].sort_values(by='strike_price'))
                df = (dataset[index][['date','exdate','strike_price', 'tau', 'S', 'r', 'd', 'impl_volatility', 'volume']].sort_values(by='strike_price'))
                df['tau'] = (df['exdate'] - df['date']).dt.days/252
                df = compute_func(df)
                df.reset_index(drop=True, inplace=True)
                tar_df.append(df)
        else:
            raise NotImplementedError
        
        if not mode == 'CDI':
            self.dataset = compute_func(tar_df)
            self.dataset.reset_index(drop=True, inplace=True)
        else:
            self.dataset = tar_df

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return list(self.dataset.iloc[index])

