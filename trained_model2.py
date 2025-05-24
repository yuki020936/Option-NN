import pandas as pd
import numpy as np
import os
import sys
from model import MultiModel
from dataset import DataProcessor, OptionDataset
from visulization import plot_3d_relation, plot_risk_neutral_density,plot_syn_risk_neutral_density
from train import TrainEngine
import config
from utils import filter
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from functools import reduce
from utils import merge_itm_otm
from tqdm import tqdm
import dask.dataframe as dd
import re
from collections import namedtuple
import pdb
from argparse import ArgumentParser
from scipy import interpolate
from scipy.stats import norm ,ks_2samp
from scipy.optimize import brentq
import time

Config = namedtuple('Config', ['test_date','result_dir_path', 'model_path', 'dataset_path', 'prs_dataset_path', 'c5_dataset_path', 'c6_0K_dataset_path', 'c6_paramK_dataset_path',
                               'prs_dataset', 'syn_dataset_with_0strike_c6', 'syn_dataset_with_parametric_c6', 'df_result',
                               'scheduler_setting', 'supd_std','syn_pred','fake_tau','CDI','standard_df'])
trained_config = Config("","", "", "", "", "", "", "", None, None, None, None, None, None, None, None,None,None)

def init_config(syn_pred = False,test_date = '2017-01-10',fake_tau = None,CDI = False):
    global trained_config
    result_dir_path = "../2017_0101_to_0731_residualmodel_WithSynData_WithSupplement"
    dataset_path = "../Dataset"
    trained_config = Config(
        test_date = test_date,
        result_dir_path = result_dir_path,
        model_path = os.path.join(result_dir_path, 'model/itm_part/model_weights_'+test_date+'.pt'), #這個不能操控.pt開始的日期
        dataset_path = dataset_path,
        prs_dataset_path = os.path.join(dataset_path, "prs_dataset_252tau.csv/2017.csv"),
        c5_dataset_path = os.path.join(dataset_path, "syn_c5_dataset_252tau.csv"), #os.path.join(dataset_path, "syn_c5_dataset_taiwan_dirty_0S_10step_20230321.csv"),
        c6_0K_dataset_path = os.path.join(dataset_path, "syn_c6_dataset_by_0strike_252tau.csv"), #os.path.join(dataset_path, "syn_c6_dataset_by_0strike_taiwan_dirty_20230321.csv"),
        c6_paramK_dataset_path = os.path.join(dataset_path, "syn_c6_dataset_by_parametric_252tau.csv"), #os.path.join(dataset_path, "syn_c6_dataset_by_parametric_taiwan_dirty_20230321.csv"),

        scheduler_setting = config.cyclic_shceduler_setting,
        supd_std = 2,

        prs_dataset = None,
        syn_dataset_with_0strike_c6 = None,
        syn_dataset_with_parametric_c6 = None,
        df_result = None,
        syn_pred = syn_pred,
        fake_tau = fake_tau,
        CDI = CDI,
        standard_df = pd.DataFrame()
    )
    
def load_dataset_and_metric():
    global trained_config
    
    prs_dataset = pd.read_csv(trained_config.prs_dataset_path, parse_dates=['date', 'exdate'])
    prs_dataset['is_syn'] = False
    c5_dataset = pd.read_csv(trained_config.c5_dataset_path, parse_dates=['date', 'exdate']) if trained_config.c5_dataset_path != "" else None
    c6_0K_dataset = pd.read_csv(trained_config.c6_0K_dataset_path, parse_dates=['date', 'exdate']) if trained_config.c6_0K_dataset_path != "" else None
    c6_paramK_dataset = pd.read_csv(trained_config.c6_paramK_dataset_path, parse_dates=['date', 'exdate']) if trained_config.c6_paramK_dataset_path != "" else None
    syn_dataset_with_0strike_c6 = pd.concat([prs_dataset, c5_dataset, c6_0K_dataset], axis=0, ignore_index=True)
    syn_dataset_with_parametric_c6 = pd.concat([prs_dataset, c5_dataset, c6_paramK_dataset], axis=0, ignore_index=True)
    df_result = pd.read_csv(os.path.join(trained_config.result_dir_path, "avg_loss_of_model_multi_among_different_dates.csv"))

    trained_config = trained_config._replace(prs_dataset=prs_dataset)
    trained_config = trained_config._replace(syn_dataset_with_0strike_c6=syn_dataset_with_0strike_c6)
    trained_config = trained_config._replace(syn_dataset_with_parametric_c6=syn_dataset_with_parametric_c6)
    trained_config = trained_config._replace(df_result=df_result)

def generate_samples_from_pdf(x, n_samples, pdf):
    x = np.asarray(x)
    pdf = np.asarray(pdf)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    pdf_sorted = pdf[sort_idx]

    # 使用梯形法估計每段機率（面積）
    dx = np.diff(x_sorted)
    mid_height = (pdf_sorted[:-1] + pdf_sorted[1:]) / 2
    probabilities = dx * mid_height
    probabilities = probabilities / np.sum(probabilities) 

    lower_bounds = x_sorted[:-1]
    upper_bounds = x_sorted[1:]

    interval_indices = np.random.choice(len(probabilities), size=n_samples, p=probabilities)

    u = np.random.uniform(0, 1, n_samples)
    samples = lower_bounds[interval_indices] + u * (upper_bounds[interval_indices] - lower_bounds[interval_indices])
    
    return samples

def KS_2sample_test(std,pdf):
    save_path = os.path.join(trained_config.result_dir_path,f"syn_fitting_result/{args.test_date}_tau{trained_config.fake_tau}/KS_result")
    os.makedirs(save_path, exist_ok=True)
        
    std['prob_model'] = std['prob_model'].apply(lambda x: max(x, 1e-10))  
    pdf['prob_model'] = pdf['prob_model'].apply(lambda x: max(x, 1e-10))
    #eps = max(0.5, np.median(np.diff(np.sort(std['KorS'].values))))
    P_value = {}
    for x,y in zip(['KorS','log_rt'],['prob_model','prob_log_rt']):
        columns = ['Standard Date','Standard_data_size','TEST DATE','TEST_data_size','sample_size','p value']
        KS_result = pd.DataFrame(columns=columns)
        for sample_size in range(round(((len(std) + len(pdf)) // 2)*0.1),round(((len(std) + len(pdf)) // 2)*2.5),100):
            np.random.seed(0)
            std_data = generate_samples_from_pdf(std[x], sample_size, std[y])
            pdf_data = generate_samples_from_pdf(pdf[x], sample_size, pdf[y])
            ks_statistic, p_value = ks_2samp(std_data, pdf_data)
            KS_result.loc[len(KS_result)] = [args.test_date,len(std),trained_config.test_date, len(pdf), sample_size, p_value]
        plt.plot(KS_result['sample_size'],KS_result['p value'])
        plt.axhline(y=0.1, color='g', linestyle='--', label='p=0.1')
        plt.axhline(y=0.05, color='#9370DB', linestyle='--', label='p=0.05')
        plt.axhline(y=0.01, color='r', linestyle='--', label='p=0.01')
        plt.axvline(x=(len(std) + len(pdf)) // 2, linestyle='--', color='blue', label='avg_data_size')
        plt.title(f"{trained_config.test_date} KS test p-value for different sample size\navg data size {(len(std) + len(pdf)) // 2}")
        plt.legend()
        plt.savefig(os.path.join(save_path,f"png_{x}_KS_result_{trained_config.test_date}_{x}.png"))
        plt.show()
        plt.close()
        KS_result.to_csv(os.path.join(save_path,f"KS_result_{trained_config.test_date}_{x}.csv"),index = False)
        P_value[x] = KS_result['p value']
    return P_value

def iterate_loading_residual_model_metrics():
    global trained_config
    for itm_otm in os.scandir(os.path.join(trained_config.result_dir_path, "model")):
        #os.scandir 來列舉指定目錄中的所有項目（檔案和子目錄），返回一個迭代器
        # whether itm_otm is a directory
        if not os.path.isdir(itm_otm): #不是資料夾就跳過
            continue
        for model_name in tqdm(os.listdir(itm_otm)):
            if model_name.endswith('.pt'): #這裡會影響.pt開始日期，就算前面改過了，這裡也會從頭開始
                model_path = os.path.join(itm_otm, model_name)
                lrmm = LoadModelMetrics(model_path, trained_config.supd_std, trained_config.scheduler_setting)
                #lrmm = LoadModelMetrics(model_path, trained_config.supd_std, trained_config.scheduler_setting , True) #這是為了CDI寫的
                lrmm.calc_option_price(save=True)

                    
def iterate_loading_syn_residual_model_metrics():
    global trained_config
    itm_path = os.path.join(trained_config.result_dir_path, "model\itm_part")
    itm_files = os.listdir(itm_path)
    itm_dates = np.array([re.search(r'\d{4}-\d{2}-\d{2}', file).group(0) for file in itm_files if re.search(r'\d{4}-\d{2}-\d{2}', file)])
    
    otm_path = os.path.join(trained_config.result_dir_path, "model\otm_part")
    otm_files = os.listdir(otm_path)
    otm_dates = np.array([re.search(r'\d{4}-\d{2}-\d{2}', file).group(0) for file in otm_files if re.search(r'\d{4}-\d{2}-\d{2}', file)])
    
    KS = []
    for i in range(0,180//trained_config.fake_tau):
        if not trained_config.standard_df.empty :
            trained_config = trained_config._replace(
            test_date= '' if (np.where(itm_dates == trained_config.test_date)[0].item() - trained_config.fake_tau < 0)
            else str(itm_dates[np.where(itm_dates == trained_config.test_date)[0].item() - trained_config.fake_tau]))
        if trained_config.test_date == '':break
        model_name = 'model_weights_'+ trained_config.test_date + '.pt'
        itm_model_path = os.path.join(itm_path, model_name)
        otm_model_path = os.path.join(otm_path, model_name)
        if os.path.exists(itm_model_path) and os.path.exists(otm_model_path):
            lrmm_itm = LoadModelMetrics(itm_model_path, trained_config.supd_std, trained_config.scheduler_setting)
            pdf = lrmm_itm.calc_syn_option_price(save=True)
            lrmm_otm = LoadModelMetrics(otm_model_path, trained_config.supd_std, trained_config.scheduler_setting)
            pdf = lrmm_otm.calc_syn_option_price(save=True)
            
            if not trained_config.standard_df.empty :
                KS.append(KS_2sample_test(trained_config.standard_df,pdf))
            else:
                if trained_config.test_date == args.test_date:
                    trained_config = trained_config._replace(standard_df = pdf)
            
def iterate_loading_multi_model_metrics():
    for model_name in tqdm(os.listdir(os.path.join(trained_config.result_dir_path, "model"))):
        if model_name.endswith('.pt'):
            model_path = os.path.join(trained_config.result_dir_path, "model", model_name)
            lmm = LoadModelMetrics(model_path, trained_config.supd_std, trained_config.scheduler_setting)
            lmm.calc_option_price(save=True)

class LoadModelMetrics:
    def __init__(self, model_path, supd_std, scheduler_setting):
        self.model_path = model_path
        self.supd_std = supd_std
        self.scheduler_setting = scheduler_setting
        self.prs_dataset = trained_config.prs_dataset
        self.syn_dataset_with_0strike_c6 = trained_config.syn_dataset_with_0strike_c6
        self.syn_dataset_with_parametric_c6 = trained_config.syn_dataset_with_parametric_c6
        self.df_result = trained_config.df_result
        self.residual_on = True if re.search('residual', self.model_path) != None else False
        try:
            self.result_dir_path = os.path.join(*model_path.split('/')[:-3]) if self.residual_on else os.path.join(*model_path.split('/')[:-2])
        except:
            self.result_dir_path = os.path.join(*model_path.split('\\')[:-3]) if self.residual_on else os.path.join(*model_path.split('\\')[:-2])
        self.itm_part = True if re.search('itm', self.model_path) != None else False
        self.test_date =  re.search(r'\d{4}-\d{2}-\d{2}', self.model_path).group(0)
        self.CDI = trained_config.CDI
        self.syn_pred  = trained_config.syn_pred
        self.fake_tau = trained_config.fake_tau

    def _set_model(self, train_df):
        model = MultiModel(residual_model_call_itm_part=self.itm_part)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        train_engine = TrainEngine(True, self.residual_on, False, self.scheduler_setting, converge_delta=1e-5,
                                   converge_patience=300, seed_list=[0, 1], result_root_path="/dev/null",
                                   IAO_threshold=None)
        train_engine.input_fscale_dict = {
            'invm_mean': (train_df['strike_price'] /
                          train_df['S']).mean(),
            'invm_std': (train_df['strike_price'] /
                         train_df['S']).std(),
            'invm_min':(train_df['strike_price'] /
                         train_df['S']).min(),
            'invm_max':(train_df['strike_price'] /
                         train_df['S']).max(),
            'tau_mean': train_df['tau'].mean(),
            'tau_std': train_df['tau'].std(),
            'tau_min':train_df['tau'].min(),
            'tau_max':train_df['tau'].max()
        }
        train_engine.output_fscale_dict = {'mean': 0, 'std': 1, 'shift': 0}
        train_engine.model = model
        train_engine.residual_model_call_itm_part = self.itm_part
        # train_engine.test_date = test_date
        return train_engine

    def _set_dataset(self):
        # parse date from self.model_path with format "YYYY-MM-DD" by regular expression
        # prs_dataset 是2017 syn_dataset是1996~2021，先用2017的invm的mean,std切itm otm再抓日期
        window_size = 5
        whole_date_list = np.sort(self.prs_dataset['date'].unique())
        invm = self.prs_dataset['strike_price'] / self.prs_dataset['S']
        if not self.residual_on:
            syn_dataset = self.syn_dataset_with_0strike_c6
        elif self.itm_part:
            syn_dataset = self.syn_dataset_with_parametric_c6[filter(self.syn_dataset_with_parametric_c6['invm'], "i", threshold=invm.mean() + self.supd_std*invm.std() if self.supd_std != 0 else 1)]
        else:
            syn_dataset = self.syn_dataset_with_0strike_c6[filter(self.syn_dataset_with_0strike_c6['invm'], "o", threshold=invm.mean() - self.supd_std*invm.std() if self.supd_std != 0 else 1)]

        train_dataset = OptionDataset(syn_dataset, 'train', np.datetime64(self.test_date), whole_date_list, window_size).dataset
        val_dataset = OptionDataset(syn_dataset, 'val', np.datetime64(self.test_date), whole_date_list, window_size).dataset
        test_dataset = OptionDataset(syn_dataset, 'test', np.datetime64(self.test_date), whole_date_list, window_size).dataset
        pdf_dataset = OptionDataset(syn_dataset, 'pdf', np.datetime64(self.test_date), whole_date_list, window_size).dataset if not self.CDI \
            else OptionDataset(syn_dataset, 'CDI', np.datetime64(self.test_date), whole_date_list, window_size).dataset

        return train_dataset, val_dataset, test_dataset, pdf_dataset
    
    def _set_syn_pred_dataset(self):
        
        window_size = 5
        whole_date_list = np.sort(self.prs_dataset['date'].unique())
        invm = self.prs_dataset['strike_price'] / self.prs_dataset['S']
        if not self.residual_on:
            syn_dataset = self.syn_dataset_with_0strike_c6
        elif self.itm_part:
            syn_dataset = self.syn_dataset_with_parametric_c6
        else:
            syn_dataset = self.syn_dataset_with_0strike_c6

        train_dataset = OptionDataset(syn_dataset, 'train', np.datetime64(self.test_date), whole_date_list, window_size).dataset
        val_dataset = OptionDataset(syn_dataset, 'val', np.datetime64(self.test_date), whole_date_list, window_size).dataset
        test_dataset = OptionDataset(syn_dataset, 'test', np.datetime64(self.test_date), whole_date_list, window_size).dataset
        pdf_dataset = OptionDataset(syn_dataset, 'pdf', np.datetime64(self.test_date), whole_date_list, window_size).dataset if not self.CDI \
            else OptionDataset(syn_dataset, 'CDI', np.datetime64(self.test_date), whole_date_list, window_size).dataset

        return train_dataset, val_dataset, test_dataset, pdf_dataset

    def calc_option_price(self,save=False):
        train_df, val_df, test_df, pdf_df = self._set_dataset()
        #pdf_df的date&tau都是相同的
        #print(f'date = {pdf_df["date"].unique()}, tau = {pdf_df["tau"].unique()}')
        train_engine = self._set_model(train_df)
        print(self.df_result[self.df_result['date'] == self.test_date].T)
        for loss_type, df_lt in zip(['train', 'test'], [train_df, test_df]): # zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
            df_lt = df_lt[filter(df_lt['strike_price']/df_lt['S'], 'i' if self.itm_part else 'o', 1)] if self.residual_on else df_lt
            df_lt = self._calc_option_price(train_engine, loss_type, df_lt)
            print(df_lt[[col for col in df_lt.columns if 'mse' in col or 'mape' in col]].mean())
            if save:
                fitting_path = os.path.join(self.result_dir_path, "fitting_result")
                prefix = "" if not self.residual_on else "itm_" if self.itm_part else "otm_"
                os.makedirs(fitting_path, exist_ok=True)
                print(f'self.test_date:{self.test_date}')
                if not self.CDI:
                    pdf = plot_risk_neutral_density(pdf_df, train_engine.model, fitting_path, self.test_date, train_engine.input_fscale_dict, train_engine.output_fscale_dict, daily_plot_off = False, is_residual=self.residual_on, is_residual_itm=self.itm_part)
                else:
                    for pdf_df_df in pdf_df:
                        pdf = plot_risk_neutral_density(pdf_df_df, train_engine.model, fitting_path, self.test_date, train_engine.input_fscale_dict, train_engine.output_fscale_dict, daily_plot_off = False, is_residual=self.residual_on, is_residual_itm=self.itm_part)
                df_lt.to_csv(os.path.join(fitting_path, f"{self.test_date}_{loss_type}_{prefix}fit_result.csv"), index=False)
                self._plot_option_price(df_lt, loss_type, fitting_path, prefix)

    def _calc_option_price(self, train_engine, loss_type, df_data):
        invm = (df_data['strike_price'] / df_data['S']).values

        if self.syn_pred and loss_type == 'test':
            mask = filter(invm, "io") if not self.residual_on else filter(invm, 'i') if self.itm_part else filter(invm, 'o')
            df_data = df_data[(mask&~df_data['is_syn'].values)] # metric on market data, not synthetic data
            _ , call_pred = train_engine._compute_call_price(df_data,True)
            call_pred = call_pred.detach().numpy()
            df_result = pd.DataFrame()
            for col in ['strike_price', 'tau', 'S', 'r', 'd']:
                df_result = pd.concat([df_result, df_data[col]], axis=1)
            df_result['call_pred'] = call_pred
            df_result['iv'] = df_result.apply(lambda r: self.implied_volatility(r['S'],r['strike_price'], r['tau'], r['d'], r['r'], r['call_pred']),axis = 1)
        else:   
            mask = filter(invm, "io") if not self.residual_on else filter(invm, 'i') if self.itm_part else filter(invm, 'o')
            df_data = df_data[(mask&~df_data['is_syn'].values)] # metric on market data, not synthetic data
            call_true, call_pred = train_engine._compute_call_price(df_data,False)
            call_pred = call_pred.detach().numpy()
            se = train_engine._mean_square_error(call_true, call_pred, 'none')
            ape = train_engine._mean_absolute_percentage_error(call_true, call_pred, 'none')
            df_result = pd.DataFrame()
            # for col in ['strike_price', 'tau', 'S', 'r', 'd', 'impl_volatility']:
            for col in ['strike_price', 'tau', 'S', 'r', 'd']:
                df_result = pd.concat([df_result, df_data[col]], axis=1)
            df_result['call_true'] = call_true
            df_result['call_pred'] = call_pred
            pre_fix = loss_type + "_" if not self.residual_on else loss_type + "_ITM" if self.itm_part else loss_type + "_OTM"
            df_result[pre_fix + 'se'] = se
            df_result[pre_fix + 'ape'] = ape
            df_result[pre_fix + 'mse'] = se.mean().item()
            df_result[pre_fix + 'mape'] = ape.mean().item()
        return df_result
    
    def calc_syn_option_price(self,save=False):
        train_df, val_df, test_true_df, pdf_df = self._set_dataset()

        eps = 0.5
        #eps = 2.055
        train_engine = self._set_model(train_df)

        raw_dividend_rate = pd.read_csv(f"{trained_config.dataset_path}/dividend_rate_from_option_metric.csv")
        prs_dividend_rate = raw_dividend_rate[['date', 'rate']].rename(columns={'rate': 'd'})
        prs_dividend_rate = prs_dividend_rate.assign(date=pd.to_datetime(prs_dividend_rate['date'], format='%Y%m%d'))
        prs_dividend_rate = prs_dividend_rate[prs_dividend_rate['date' ]== trained_config.test_date]
        prs_dividend_rate.reset_index(drop=True)
        S =  test_true_df['S'].unique()[0]
        test_df ={'strike_price':np.arange(train_engine.input_fscale_dict['invm_min']*S,train_engine.input_fscale_dict['invm_max']*S,eps)}
        test_df = pd.DataFrame(test_df)
        test_df = test_df.assign(tau = trained_config.fake_tau/252, S = S, r = 0.0,d = prs_dividend_rate['d'].unique()[0]*0.01,is_syn = False)
        # process risk free rate from Option Metric
        raw_risk_free_rate = pd.read_csv(f"{trained_config.dataset_path}/risk_free_rate_from_option_metric.csv")
        prs_risk_free_rate = pd.DataFrame()
        raw_risk_free_rate.iloc[:, 1:] = raw_risk_free_rate.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
        # prs_risk_free_rate is for true_x and true_y. don't apply dask dataframe to it to make sure interpolate work appropriate
        prs_risk_free_rate['date'] = pd.to_datetime(raw_risk_free_rate['date'], format='%Y%m%d')
        prs_risk_free_rate['years'] = raw_risk_free_rate['days'] / 360.0
        prs_risk_free_rate['rate'] = raw_risk_free_rate['rate']
        def cubic_func(date, prs_single_df, prs_risk_free_rate):
            true_idx = prs_risk_free_rate['date'] == date
            if sum(true_idx) > 0:
                true_x = prs_risk_free_rate.loc[true_idx, 'years']
                true_y = prs_risk_free_rate.loc[true_idx, 'rate']
                target_x = prs_single_df['tau']
                target_y = interpolate.splev(target_x, interpolate.splrep(true_x, true_y))
                prs_single_df['r'] = target_y * 0.01 # because of the percentage format
            return prs_single_df
        test_df = cubic_func(self.test_date,test_df,prs_risk_free_rate)
        #print(self.df_result[self.df_result['date'] == self.test_date].T)
        print(f'Try to synthesis {"itm" if self.itm_part else "otm"} option value for test_date:{self.test_date},S:{S}, tau:{self.fake_tau}')
        for loss_type, df_lt in zip(['train', 'test'], [train_df, test_df]): # zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
            if loss_type == 'train':
                continue
            df_lt = df_lt[filter(df_lt['strike_price']/df_lt['S'], 'i' if self.itm_part else 'o',  1)] if self.residual_on else df_lt
            df_lt = df_lt.reset_index(drop = True)
            df_lt = self._calc_option_price(train_engine, loss_type, df_lt)
            #print(df_lt[[col for col in df_lt.columns if 'mse' in col or 'mape' in col]].mean())
            if save:
                fitting_path = os.path.join(self.result_dir_path, f"syn_fitting_result/{args.test_date}_tau{trained_config.fake_tau}")
                prefix = "" if not self.residual_on else "itm_" if self.itm_part else "otm_"
                os.makedirs(fitting_path, exist_ok=True)
                print(f'self.test_date:{self.test_date}')
                df_lt.to_csv(os.path.join(fitting_path, f"{self.test_date}_{loss_type}_{prefix}_synfit_result.csv"), index=False)
                if loss_type == 'train':
                    pdf = plot_risk_neutral_density(pdf_df, train_engine.model, fitting_path, self.test_date, train_engine.input_fscale_dict, train_engine.output_fscale_dict, daily_plot_off = False, is_residual=self.residual_on, is_residual_itm=self.itm_part)
                else:
                    pdf = plot_syn_risk_neutral_density(df_lt, train_engine.model, fitting_path, self.test_date, train_engine.input_fscale_dict, train_engine.output_fscale_dict, daily_plot_off = False, is_residual=self.residual_on, is_residual_itm=self.itm_part)
                    return pdf
                

    def _plot_option_price(self, df_lt, loss_type, fitting_path, prefix):
        try:
            intrinsic_value = df_lt['S'] - df_lt['strike_price']
            intrinsic_value = intrinsic_value.apply(lambda x: x if x > 0 else 0)
            plot_3d_relation(df_lt['tau'], df_lt['strike_price'] / df_lt['S'],
                             {'true': df_lt['call_true'],
                              'pred': df_lt['call_pred']}, 'tau', 'invm',
                             f'{loss_type}_option_value', self.test_date, angle=(30, 0),
                             path=os.path.join(fitting_path, f"{self.test_date}_{loss_type}_ov_{prefix}fitting.png"))
            plot_3d_relation(df_lt['tau'], df_lt['strike_price'] / df_lt['S'],
                             {'true': df_lt['call_true'] - intrinsic_value,
                              'pred': df_lt['call_pred'] - intrinsic_value}, 'tau', 'invm',
                             f'{loss_type}_time_value', self.test_date, angle=(30, 0),
                             path=os.path.join(fitting_path, f"{self.test_date}_{loss_type}_tv_{prefix}fitting.png"))
        except Exception as e:
            print(self.test_date + ": " + str(e))
            
    def implied_volatility(self,S, K, tau, d, r, option_price):

        if option_price <= 0 :
            return np.nan
        def objective(sigma):
            BS = lambda S, K, tau, d, r, sigma: S * np.exp(-d * tau) * norm.cdf(
                ((np.log(S / K) + ((r - d) + 0.5 * (sigma ** 2)) * tau) / (sigma * np.sqrt(tau))).item()) - K * np.exp(
                -r * tau) * norm.cdf(
                ((np.log(S / K) + ((r - d) - 0.5 * (sigma ** 2)) * tau) / (sigma * np.sqrt(tau))).item())
            
            return BS(S, K, tau, d, r, sigma) - option_price

        try:
            sigma = brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=10000)
            return sigma
        except (ValueError, RuntimeError):
            return np.nan
         

if __name__ == '__main__':
    start_time = time.time()
    sys.argv = ['train_model2.py','--syn_pred','--test_date','2017-08-14','--tau','30']
    parser = ArgumentParser()
    parser.add_argument('--syn_pred', action='store_true')
    parser.add_argument('--test_date', type=str, default='2017-01-10')
    parser.add_argument('--tau', type=int, default=30)
    args = parser.parse_args()
    print("start ... this program will create fitting_result folder")
    if args.syn_pred:
        print(f"This program will creat nonexistent option price data : date = {args.test_date}, tau = {args.tau} ")
        init_config(args.syn_pred,args.test_date,args.tau)
    else:
        init_config()
    load_dataset_and_metric()
    
    
    #lmm = LoadModelMetrics(trained_config.model_path, trained_config.supd_std, trained_config.scheduler_setting)
    #lmm.calc_option_price(save=True)

    if 'residual' in trained_config.result_dir_path:
        if args.syn_pred:
            iterate_loading_syn_residual_model_metrics()
        else:
            iterate_loading_residual_model_metrics()
    else:
        iterate_loading_multi_model_metrics()
    end_time = time.time()
    print(f"執行時間：{end_time - start_time:.3f} 秒")