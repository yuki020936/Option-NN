import torch
import torch.optim as optim
import torch.nn as nn
#import pytorch_forecasting
import numpy as np
import pandas as pd
import config
import random
import os
from visulization import plot_learning_rate, plot_lr_range_test, plot_3d_relation
from sklearn.neighbors import KernelDensity
import copy
import shutil
### from torch_lr_finder import LRFinder # lr range test
from utils import feature_scale, inverse_feature_scale, filter, merge_itm_otm, seed_initializer
from visulization import plot_risk_neutral_density, plot_date_loss_curve
import time

### TODO: OTM residual train 不好可能是因爲 no C5
### TODO: moneyness 裡面的 S 是不是也應該要考慮去除掉 dividend 的影響啊？
### TODO: check OTM residual on 2010-01-28 after 250 epoch
### TODO: 試試看讓 OTM 也用 square plus activation function
### TODO: time value _/\_, x-axis is S_T, how about change x-axis to K so that time value shape is \/ which is smoother

class TrainEngine:
    def __init__(self, daily_plot_off, residual_on, use_multistep_learning_rate, scheduler_setting, converge_delta, converge_patience, seed_list, IAO_threshold, result_root_path,start_time):
        # hyper-parameters
        self.daily_plot_off = daily_plot_off
        self.residual_on = residual_on
        # use_step_lr : 判斷是否使用分段的學習率調整策略，True意即在不同階段使用不同的學習率，幫助模型更快收斂至min或避免陷入local min
        self.use_step_lr = use_multistep_learning_rate
        # scdl_setting : 用於控制schduler中的每一項參數，詳細記錄在"config.py"當中
        self.scdl_setting = scheduler_setting
        self.seed_list = seed_list
        self.converge_delta = converge_delta
        self.converge_patience = converge_patience
        self.IAO_threshold = IAO_threshold
        self.result_root_path = result_root_path
        self.start_time = start_time

        # daily required training materials
        self.residual_model_call_itm_part = None
        self.loss_weight_table_for_residual_model = None
        self.validate_folder = None
        self.test_folder = None
        self.model_folder = None
        self.test_date = None
        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.pdf_dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.wu_scheduler = None
        self.ms_scheduler = None
        self.input_fscale_dict = None
        self.output_fscale_dict = None
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.args.on_gpu else "cpu")

    def __call__(self, test_date, modell, train_dataloader_ITM_part, train_dataloader_OTM_part, train_dataloader, val_dataloader_ITM_part, val_dataloader_OTM_part,  val_dataloader, test_dataloader_ITM_part, test_dataloader_OTM_part, test_dataloader, pdf_dataloader):
        ### TODO: 可是五天的資料一起丟下去還能學到 monotonic K, NN output (call) 對 K 微分<0 嗎？因為這個這個特性要 fix date exdate呀，還是因為有 tau? 如果 K 上升同時 tau 也上升, NN output call option price 會?
        # 將test_date的日期轉為字串，其中'D'表示轉換格是為 年-月-日 ex. '2023-01-01'
        self.test_date = np.datetime_as_string(test_date, 'D')
        self.pdf_dataloader = pdf_dataloader
        ### train 4 days, validate 1 days, test 1 days
        ### for each hyper-parameters setting, pick the model which stop training at specific epoch that have the lowest validate loss
        ### after training with all possible combination of hyper-parameters, there will have # (aka. amount of hyper-parameters combination) models. again, pick the one with the lowest validate loss
        ### we will use the model to do the test
        if self.residual_on:
            # 創建folder的路徑
            test_merge_folder = f"{self.result_root_path}/test/merge_itm_otm"
            # os.makedirs : 在路徑中創建一個folder
            # exist_ok = true : 若不存在則創建，若存在則不會有錯誤
            os.makedirs(test_merge_folder, exist_ok=True)
            # 整理出一個可以用invm和tau進行索引，包含了invm及loss weight的表，其中不存在索引值相同的數據(invm和tau完全相同之值)
            self._calc_loss_weight_for_residual_model(train_dataloader_ITM_part.dataset.dataset, train_dataloader_OTM_part.dataset.dataset)
            itm_loss_dict = self._grid_search(modell, train_dataloader_ITM_part, val_dataloader_ITM_part, test_dataloader_ITM_part, residual_model_call_itm_part=True)
            otm_loss_dict = self._grid_search(modell, train_dataloader_OTM_part, val_dataloader_OTM_part, test_dataloader_OTM_part, residual_model_call_itm_part=False)
            itm_loss_curve = pd.read_csv(f"{self.result_root_path}/test/itm_part/loss_message_{self.test_date}.csv").to_dict('list')
            otm_loss_curve = pd.read_csv(f"{self.result_root_path}/test/otm_part/loss_message_{self.test_date}.csv").to_dict('list')
            loss_dict, loss_curve = dict(), dict()
            loss_curve['epoch'] = itm_loss_curve['epoch'] if len(itm_loss_curve['epoch']) > len(otm_loss_curve['epoch']) else otm_loss_curve['epoch']
            for phase in ['train', 'val', 'test']:
                itm_count, otm_count = eval(f'{phase}_dataloader_ITM_part.dataset.dataset'), \
                                       eval(f'{phase}_dataloader_OTM_part.dataset.dataset')
                itm_count = len(itm_count[filter(itm_count['strike_price']/itm_count['S'], 'i') & (~itm_count['is_syn'].values)])
                otm_count = len(otm_count[filter(otm_count['strike_price']/otm_count['S'], 'o') & (~otm_count['is_syn'].values)])
                loss_dict.update(merge_itm_otm(phase, itm_count, otm_count, itm_loss_dict, otm_loss_dict, split_itm_otm_loss=False))
                loss_curve.update(merge_itm_otm(phase, itm_count, otm_count, itm_loss_curve, otm_loss_curve, split_itm_otm_loss=True))

            pd.DataFrame(loss_curve).to_csv(f"{test_merge_folder}/loss_message_{self.test_date}.csv", index=False)
            plot_date_loss_curve(loss_curve, test_merge_folder, self.test_date, is_merge=True)
        else:
            loss_dict = self._grid_search(modell, train_dataloader, val_dataloader, test_dataloader)

        return (self.test_date, *[v for v in loss_dict.values()])

    def _transfer_to_index_of_loss_weight_table(self, invm, tau):
        return ('invm' + pd.Series(invm.detach().cpu().numpy() if torch.is_tensor(invm) else invm).map('{:.5f}'.format) + '_tau' + pd.Series(tau.detach().cpu().numpy() if torch.is_tensor(tau) else tau).map('{:.5f}'.format)).values

    def _calc_loss_weight_for_residual_model(self, trainITM_dataset, trainOTM_dataset):
        # pd.concat : 將兩個series或dataframe進行合併
        #             axis = 0 : 垂直，axis = 1 : 水平
        #             reset_index(drop=True): 將index從0開始排，並捨棄舊的index
        train_dataset = pd.concat([trainITM_dataset, trainOTM_dataset], axis=0).reset_index(drop=True)
        # 按照公式計算invm
        train_dataset['invm'] = train_dataset['strike_price'] / train_dataset['S']
        x = train_dataset[['invm']]
        # 用Kernal Density Estimation(KDE)的方式來找出pdf
        # kernal = 'gaussian' : 選定高斯核函數作為估計的核函數(在每個數據點周圍建一個高斯分布)
        # bandwidth = 控制核函數的寬度，用於影響KDE的平滑度，寬度越大越平滑
        kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(x)
        # kde.score_sample : 計算給定數據點的對數pdf
        log_pdf = kde.score_samples(x)
        # 加入對數pdf的最小值，使所有數據點的pdf皆為正數
        pos_log_pdf = (log_pdf + np.abs(log_pdf.min()))
        # 將資料合併成每個invm配上一個loss weight
        self.loss_weight_table_for_residual_model = pd.concat([x, pd.Series(pos_log_pdf).rename('loss_weight')], axis=1)
        # 將資料的index改成可以用invm和tau進行索引
        self.loss_weight_table_for_residual_model.index = self._transfer_to_index_of_loss_weight_table(train_dataset['invm'], train_dataset['tau'])
        # 刪除所有重複的值，僅留下第一次出現的index的結果
        # duplicate(keep='first'): 將傳回一個bool值的series，第一次出現的為False，重複出現的則改為True
        # 此處使用~duplicate(keep='first')，意即將重複出現的改為False，所以會刪除
        self.loss_weight_table_for_residual_model = self.loss_weight_table_for_residual_model[~self.loss_weight_table_for_residual_model.index.duplicated(keep='first')] # drop identical indices

    def _grid_search(self, modell, train_dataloader, val_dataloader, test_dataloader, residual_model_call_itm_part=False):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.residual_model_call_itm_part = residual_model_call_itm_part
        # 用於儲存路徑的字串
        self.test_folder = f"{self.result_root_path}/test"
        # 用於儲存路徑的字串
        self.model_folder = f"{self.result_root_path}/model"
        ### Z-score feature scaling
        ### to shift the ground truth output from negative region and [0,1]. we hope the target_true_all_fscaling > 1 to ensure MAPE workable
        ### because Multi model can't output negative value. if we don't shift it from negative region, the output of Multi model will converge to 0 ...
        ### also, if the output is locate on [0,1], it will converge to 0, too ...
        ### hence we don't use feature scaling for output
        # 建構一個dictionary，裡面包含'mean', 'std', 'shift'，且都先給定一固定值
        self.output_fscale_dict = {'mean': 0, 'std': 1, 'shift': 0}
        # 建構參數並先賦值None，以便後續修改
        folder_suffix = None
        model_selection = None
        # 判斷 1.是否為學姊論文的版本，2.是否為ITM的資料
        # 若皆為True則進入if，若非則進入else
        if self.residual_on and self.residual_model_call_itm_part:
            folder_suffix = '/itm_part'
            model_selection = 'residual_itm'
            ### for residual model's call itm part
            # 建構一個dictionary並計算包含invm及tau的mean和std
            self.input_fscale_dict = {
                'invm_mean': (self.train_dataloader.dataset.dataset['strike_price'] /
                           self.train_dataloader.dataset.dataset['S']).mean(),
                'invm_std': (self.train_dataloader.dataset.dataset['strike_price'] /
                          self.train_dataloader.dataset.dataset['S']).std(),
                'tau_mean': self.train_dataloader.dataset.dataset['tau'].mean(),
                'tau_std': self.train_dataloader.dataset.dataset['tau'].std(),
            }
        else:
            # 判斷是否為學姊論文的版本，若為True則進入if，若非則進入else
            if self.residual_on:
                folder_suffix = '/otm_part'
                model_selection = 'residual_otm'
            else:
                folder_suffix = ''
                model_selection = 'aaai17'
            ### for aaai17 multi model's call itm part and call otm part
            ### also, for residual model's call itm part
            # 建構一個dictionary並計算包含invm及tau的mean和std
            self.input_fscale_dict = {
                'invm_mean': (self.train_dataloader.dataset.dataset['strike_price'] /
                              self.train_dataloader.dataset.dataset['S']).mean(),
                'invm_std': (self.train_dataloader.dataset.dataset['strike_price'] /
                             self.train_dataloader.dataset.dataset['S']).std(),
                'tau_mean': self.train_dataloader.dataset.dataset['tau'].mean(),
                'tau_std': self.train_dataloader.dataset.dataset['tau'].std(),
            }

        # 將資料屬性(itm或otm)的字串加入原先路徑的字串
        self.test_folder += folder_suffix
        self.model_folder += folder_suffix
        # 在路徑中創建並判斷folder是否存在
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        # 用一個dictionary儲存與損失相關的資料，通常用於訓練時追蹤最佳模型的性能，訓練過程中若發現更好的性能則更新dictionary中的值
        # 其中包含一個key : 'min_val_metric'和一個值inf(正無窮)
        # inf用於初始化，以確保開始訓練時的初始值是一個極大的數
        best_loss_dict = {'min_val_metric': float("inf")}

        ### set scheduler & optimizer, and then update the corresponding performance of best parameters
        # 以下是在做grid search找出best loss
        # 若use_step_lr為True則進入if
        if self.use_step_lr:
            # 使用兩個變數進行model1的迭代
            # seed_idx : 紀錄當前迭代中的索引值
            # untrained_model : 紀錄當前迭代中model1的值
            for seed_idx, untrained_model in enumerate(modell):
                # 對schduler中的學習率進行迭代，此處共兩個值分別為0.05和0.1
                for init_lr in self.scdl_setting['init_lr_list']:
                    # 對schduler中的warmup進行迭代，此處共兩個值分別為0和100
                    for warmup in self.scdl_setting['epoch_warmup_list']:
                        # 用於儲存路徑的字串
                        self.validate_folder = f"{self.result_root_path}/validation_{self.seed_list[seed_idx]}seed_{init_lr}lr_{warmup}warmup"
                        # 對不同的warmup重新計算epoch的數量並記錄當前epoch的大小，通常用於控制學習率的調整或其他訓練中的操作
                        self.scdl_setting['cur_epoch_size'] = warmup + self.scdl_setting['epoch_decay'] * (self.scdl_setting['decay_times'] + 1) + self.scdl_setting['epoch_last'] # re-calculate epoch_size based on diff warmup
                        # 紀錄當前warmup的epoch的大小
                        self.scdl_setting['cur_epoch_warmup'] = warmup
                        self.model = copy.deepcopy(untrained_model[model_selection]).to(self.device)
                        if next(self.model.parameters()).is_cuda:
                          print(f"模型 {model_selection} 已成功移至 GPU 訓練")
                        else:
                          print(f"模型 {model_selection} 仍在 CPU，請確認有正確使用 .to(device)")
                        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
                        self.ms_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.scdl_setting['epoch_decay']*i+1 for i in range(1, self.scdl_setting['decay_times']+1) if self.scdl_setting['epoch_decay']>0], gamma=self.scdl_setting['gamma'], last_epoch=-1)
                        warm_up_func = lambda epoch: epoch / warmup if epoch <= warmup and warmup != 0 else 1
                        self.wu_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=warm_up_func)
                        # https://github.com/pytorch/pytorch/issues/67586
                        # self.scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=self.optimizer, schedulers=[self.wu_scheduler, self.ms_scheduler], milestones=[self.scdl_setting['cur_epoch_warmup']])
                        self._update_the_best_param_and_corres_performance(folder_suffix, seed_idx, best_loss_dict)
        else:
            for seed_idx, untrained_model in enumerate(modell):
                self.validate_folder = f"{self.result_root_path}/validation_{self.seed_list[seed_idx]}seed"
                cycle_len = 2*len(self.train_dataloader)
                self.model = copy.deepcopy(untrained_model[model_selection]).to(self.device)
                if next(self.model.parameters()).is_cuda:
                  print(f"模型 {model_selection} 已成功移至 GPU 訓練")
                else:
                  print(f"模型 {model_selection} 仍在 CPU，請確認有正確使用 .to(device)")
                base_lr, max_lr = self._lr_range_test(min_lr=1e-10, max_lr=1, num_iter=cycle_len, smooth_f=0.5, diverge_th=5)
                self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=cycle_len, mode='exp_range', gamma=self.scdl_setting['gamma'], cycle_momentum=False)
                # self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=cycle_len, mode='triangular2', cycle_momentum=False)
                self._update_the_best_param_and_corres_performance(folder_suffix, seed_idx, best_loss_dict)

        if next(self.model.parameters()).is_cuda:
          print(f"模型 {model_selection} 已成功移至 GPU 訓練")
        else:
          print(f"模型 {model_selection} 仍在 CPU，請確認有正確使用 .to(device)")
        best_loss_dict.pop('min_val_metric') # the key set and order in dict must be: test_date, train_loss, train_mse, train_mape, train_ITMmse, train_ITMmape, train_OTMmse, train_OTMmape, test_loss, test_mse, test_mape, test_ITMmse, test_ITMmape, test_OTMmse, test_OTMmape
        return best_loss_dict

    def _update_the_best_param_and_corres_performance(self, folder_suffix, seed_idx, best_loss_dict):
        # reset dataloader's rand seed
        seed_initializer(self.seed_list[seed_idx])

        # modify folder name
        self.validate_folder += folder_suffix
        os.makedirs(self.validate_folder, exist_ok=True)

        loss_dict, model_weights = self._train()
        if loss_dict['val_mse'] + loss_dict['val_mape'] < best_loss_dict['min_val_metric']:
            best_loss_dict['min_val_metric'] = loss_dict['val_mse'] + loss_dict['val_mape']
            torch.save(model_weights, f"{self.model_folder}/model_weights_{self.test_date}.pt")
            for source_file in [f"loss_{self.test_date}.png", f"loss_message_{self.test_date}.csv",
                                f"implied_distribution_{self.test_date}_current.png",
                                f"implied_distribution_{self.test_date}_current.json"]:
                source_path = f"{self.validate_folder}/{source_file}"
                target_path = f"{self.test_folder}/{source_file}"
                if os.path.exists(source_path):
                    shutil.copyfile(source_path, target_path)

            for k in loss_dict.keys():
                if 'train' in k or 'test' in k:
                    best_loss_dict[k] = loss_dict[k]

    def _lr_range_test(self, min_lr, max_lr, num_iter, skip_start=10, skip_end=5, smooth_f=0.05, diverge_th=5):
        # refer to https://github.com/davidtvs/pytorch-lr-finder
        # retain original model
        retain_model = copy.deepcopy(self.model)

        # reset start lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=min_lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=num_iter, mode='triangular', cycle_momentum=False)
        train_data_iterator = iter(self.train_dataloader)
        running_lr_list, loss_list = [], []
        for iteration in range(num_iter):
            try:
                one_batch_input = next(train_data_iterator)
            except StopIteration:
                print(f'except StopIteration')
                train_data_iterator = iter(self.train_dataloader)
                one_batch_input = next(train_data_iterator)
            self.epoch = 0
            train_loss = self._train_one_iteration(*one_batch_input)
            train_loss = smooth_f * train_loss + (1 - smooth_f) * loss_list[-1] if len(loss_list)>0 else train_loss # smooth
            running_lr_list.append(self.optimizer.param_groups[0]['lr'])
            loss_list.append(train_loss)
            # if train_loss > diverge_th * np.min(loss_list):
            #     print("Stopping early, the loss has diverged")
            #     break
            self.scheduler.step()

        # find the best learning rate
        running_lr_list = running_lr_list[skip_start:-skip_end]
        loss_list = loss_list[skip_start:-skip_end]
        max_grad_idx = (np.abs(np.gradient(loss_list))).argmax()
        # stable_grad_idx = np.argmax(np.abs(np.gradient(loss_list)) < 0.1)
        plot_lr_range_test(running_lr_list, loss_list, max_grad_idx)

        # reset model
        self.model = retain_model
        return running_lr_list[max_grad_idx]*(self.scdl_setting['lr_max_to_base_ratio']) , running_lr_list[max_grad_idx]

    def _train(self):
        # TODO: why 2012-03-27 test_mse explode?!
        loss_dict = {'train_loss_list': [], 'train_mse_list': [], 'train_mape_list': [],
                     'train_ITMmse_list': [], 'train_ITMmape_list': [], 'train_OTMmse_list': [], 'train_OTMmape_list': [],
                     'val_loss_list':[], 'val_mse_list':[], 'val_mape_list':[],
                     'val_ITMmse_list':[], 'val_ITMmape_list':[], 'val_OTMmse_list':[], 'val_OTMmape_list':[],
                     'test_loss_list': [], 'test_mse_list': [], 'test_mape_list': [],
                     'test_ITMmse_list': [], 'test_ITMmape_list': [], 'test_OTMmse_list': [], 'test_OTMmape_list': []
                     }
        converge_trigger_times = 0
        min_val_metric, epoch_with_min_val_metric, best_model_weights_on_specific_epoch = float("inf"), -1, None
        running_lr_list= []
        for self.epoch in range(self.scdl_setting['cur_epoch_size']):
            # train
            # if you observe that train loss becomes large abruptly, it may because of mape's denominator's 1e-8; btw, without 1e-8, denominator will be 0 and cause mape -> oo
            train_loss, *train_metrics = self._train_one_epoch(running_lr_list)
            val_loss, *val_metrics = self._evaluate(self.val_dataloader)
            test_loss, *test_metrics = self._evaluate(self.test_dataloader)

            if self.use_step_lr:
                if self.epoch < self.scdl_setting['cur_epoch_warmup']:
                    self.wu_scheduler.step()
                elif self.scdl_setting['decay_times'] != 0:
                    self.ms_scheduler.step()

            # update best results
            if val_metrics[0] + val_metrics[1] < min_val_metric:
                min_val_metric = val_metrics[0] + val_metrics[1] # val mse + val mape
                epoch_with_min_val_metric = self.epoch
                best_model_weights_on_specific_epoch = copy.deepcopy(self.model.state_dict()) # deepcopy is necessary

            # early stop for validation phase; stop when converge for test phase
            if len(loss_dict['val_loss_list']) != 0 and val_loss > np.mean(loss_dict['val_loss_list'][-config.moving_average_epoch:]) - self.converge_delta and val_metrics[0]<100: # MSE < 100
                converge_trigger_times += 1
                if converge_trigger_times >= self.converge_patience:
                    break
            else:
                converge_trigger_times = 0

            # record
            for loss, k in zip([train_loss, *train_metrics, val_loss, *val_metrics, test_loss, *test_metrics], loss_dict.keys()):
                loss_dict[k].append(loss)

            if ((self.epoch % 100 == 0) & (self.epoch != 0)) or self.epoch == (self.scdl_setting['cur_epoch_size'] - 1):
                print(f"epoch {self.epoch}, train_loss: {loss_dict['train_loss_list'][-1]:.3f}, train_mse: {loss_dict['train_mse_list'][-1]:.3f}, train_mape: {loss_dict['train_mape_list'][-1]:.3f}, test_loss:{loss_dict['test_loss_list'][-1]:.3f}, test_mse: {loss_dict['test_mse_list'][-1]:.3f}, test_mape: {loss_dict['test_mape_list'][-1]:.3f}")
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = int(elapsed_time % 60)
                print(f"執行時間：{hours:02d}小時 {minutes:02d}分 {seconds:02d}秒")
                # self.show_detailed_perfromance(self.train_dataloader.dataset.dataset, phase='train', savefig=True)
                # self.show_detailed_perfromance(self.val_dataloader.dataset.dataset, phase='val', savefig=True)
                # self.show_detailed_perfromance(self.test_dataloader.dataset.dataset, phase='test', savefig=True)
                plot_learning_rate(self.validate_folder, running_lr_list, self.daily_plot_off)
                plot_date_loss_curve(loss_dict, self.validate_folder, self.test_date, self.daily_plot_off)

                # plot_risk_neutral_density(self.pdf_dataloader.dataset.dataset, self.model, self.validate_folder, self.test_date, self.input_fscale_dict, self.output_fscale_dict, self.daily_plot_off, is_residual=self.residual_on, is_residual_itm=self.residual_model_call_itm_part)

        return {k[:-5]:loss_dict[k][epoch_with_min_val_metric] for k in loss_dict.keys()}, best_model_weights_on_specific_epoch

    def show_detailed_perfromance(self, dataset, phase, savefig=False):

        # TODO: debug, K=1008, [big gap, why?], K=1009
        t1 = pd.DataFrame(
            {'r':0.01437, 'd': 0.01790, 'tau': 2.92857, 'S': 1307.4, 'strike_price': 1050, 'call_true': 332.51294, 'call_pred': 12727.90738,
             'call_intrinsic': 190.62961,
             'call_time_true': 141.88332, 'call_time_pred': 12737.27777, 'target_true': 0.07866,
             'target_pred': 9.96724}, index=[0])
        t1['option_price'] = t1['call_true']
        t1['invm'] = t1['strike_price'] / t1['S']
        call_true, call_pred = self._compute_call_price(t1)
        call_pred

        tau = dataset['tau']
        invm = dataset['strike_price']/dataset['S']
        call_true, call_pred = self._compute_call_price(dataset)
        target_true, target_pred = self._compute_target(call_true, torch.tensor(dataset['strike_price'].to_numpy()).to(self.model.device), torch.tensor(dataset['tau'].to_numpy()).to(self.model.device), torch.tensor(dataset['S'].to_numpy()).to(self.model.device), torch.tensor(dataset['r'].to_numpy()).to(self.model.device), torch.tensor(dataset['d'].to_numpy()).to(self.model.device))
        call_intrinsic = dataset['S']*np.exp(-dataset['d']*dataset['tau']) - dataset['strike_price']
        call_intrinsic[call_intrinsic < 0] = 0
        call_time_true = call_true.detach().cpu().numpy() - call_intrinsic
        call_time_pred = call_pred.detach().cpu().numpy() - call_intrinsic




        t = pd.DataFrame()
        t['tau'] = tau
        t['invm'] = invm
        t['S'] = dataset['S']
        t['r'] = dataset['r']
        t['d'] = dataset['d']
        t['strike_price'] = dataset['strike_price']
        t['call_true'] = call_true
        t['call_pred'] = call_pred.detach().cpu().numpy()
        t['call_intrinsic'] = call_intrinsic
        t['call_time_true'] = call_time_true
        t['call_time_pred'] = call_time_pred
        t['target_true'] = target_true.detach().cpu().numpy()
        t['target_pred'] = target_pred.detach().cpu().numpy()
        t['mse'] = (t['call_true'] - t['call_pred'])**2
        t['mape'] = np.abs(t['call_true'] - t['call_pred'])/t['call_true']
        t.sort_values(by=['tau', 'invm'], inplace=True)

        # plot_3d_relation(tau, invm, {'ape': self._mean_absolute_percentage_error(*self._compute_call_price(dataset), 'none').detach().numpy(), 'se': self._mean_square_error(*self._compute_call_price(dataset), 'none').detach().numpy()}, 'tau', 'invm', f'{phase}_loss', self.test_date, angle=(30, 0), path=f"{self.validate_folder}/{self.test_date}_{phase}_loss_{self.epoch}epoch.png" if savefig else None)
        # plot_3d_relation(tau, invm, {'true': call_time_true, 'pred': call_time_pred}, 'tau', 'invm', f'{phase}_time_value', self.test_date, angle=(30, 0), path=f"{self.validate_folder}/{self.test_date}_{phase}_timevalue_{self.epoch}epoch.png" if savefig else None)
        # plot_3d_relation(tau, invm, {'true': call_true.detach().numpy(), 'pred': call_pred.detach().numpy()}, 'tau', 'invm', f'{phase}_option_value', self.test_date, angle=(30, 0), path=f"{self.validate_folder}/{self.test_date}_{phase}_optionvalue_{self.epoch}epoch.png" if savefig else None)
        plot_3d_relation(tau, invm, {'true': target_true.detach().cpu().numpy(), 'pred': target_pred.detach().cpu().numpy()}, 'tau', 'invm', f'{phase}_target_value', self.test_date, angle=(30, 0), path=f"{self.validate_folder}/{self.test_date}_{phase}_targetvalue_{self.epoch}epoch.png" if savefig else None)
        print("debug")

    def _train_one_iteration(self, call_true, K, tau, S, r, d, is_syn):
        call_true, K, tau, S, r, d = call_true.to(self.model.device), K.to(self.model.device), tau.to(self.model.device), S.to(self.model.device), r.to(self.model.device), d.to(self.model.device)
        target_true_fs, target_pred_fs = self._compute_target(call_true, K, tau, S, r, d)
        loss = self._compute_loss(target_true_fs, target_pred_fs, K, S, tau, is_syn, for_train=True)
        current_loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

    def _train_one_epoch(self, running_lr_list):
        self.model.train()
        for iteration, one_batch_input in enumerate(self.train_dataloader):
            current_loss = self._train_one_iteration(*one_batch_input) # because dataloader will shuffle, current_loss's corresponding data is different for each time we reach the last times of the loop
            if not self.use_step_lr:  # cyclical scheduler
                self.scheduler.step()
            running_lr_list += [self.optimizer.param_groups[0]['lr']]
        return (current_loss, *self._compute_metric(self.train_dataloader.dataset.dataset))

    def _evaluate(self, eval_dataloader):
        # train loss measure small batch target_xxxx_fs' performance; val loss & test loss measure whole data's target_xxxx_fs' performance
        # train/val/test mse & mape measure whole call option price's performance
        self.model.eval()
        df = eval_dataloader.dataset.dataset
        call_true = torch.from_numpy(df['option_price'].values).to(self.model.device)
        K = torch.from_numpy(df['strike_price'].values).to(self.model.device)
        tau = torch.from_numpy(df['tau'].values).to(self.model.device)
        S = torch.from_numpy(df['S'].values).to(self.model.device)
        r = torch.from_numpy(df['r'].values).to(self.model.device)
        d = torch.from_numpy(df['d'].values).to(self.model.device)
        is_syn = torch.from_numpy(df['is_syn'].values)
        target_true_fs, target_pred_fs = self._compute_target(call_true, K, tau, S, r, d)
        loss = self._compute_loss(target_true_fs, target_pred_fs, K, S, tau, is_syn, for_train=False)
        return (loss.item(), *self._compute_metric(eval_dataloader.dataset.dataset))

    def _compute_target(self, call_true, K, tau, S, r, d, for_metric=False,syn_pred = False):
        # Z-score feature scaling accord. train dataset before feeding into model
        invm = K/S
        invm_fs = feature_scale(invm, mean=self.input_fscale_dict['invm_mean'], std=self.input_fscale_dict['invm_std'])
        if not syn_pred:
            tau_fs = feature_scale(tau, mean=self.input_fscale_dict['tau_mean'], std=self.input_fscale_dict['tau_std'])
        else:
            tau_fs = tau
        if self.residual_on:
            if self.residual_model_call_itm_part:
                # i.e. predict put out the money side
                put_true = call_true - S*torch.exp(-d*tau) + K*torch.exp(-r*tau)
                put_intrinsic_value = K - S * torch.exp(-d * tau)
                put_intrinsic_value[put_intrinsic_value < 0] = 0
                put_intrinsic_value = 0 # for OTM put with ATM put, force to not exercise the ATM part
                put_time_value_true = put_true - put_intrinsic_value
                target_true = put_time_value_true * torch.exp(r * tau) / S
                target_pred_fs = self.model(invm_fs, tau_fs)
                if for_metric:
                    # convert otm put target to itm call target
                    target_true = call_true * torch.exp(r * tau) / S
                    put_time_value_pred = target_pred_fs * S / torch.exp(r * tau)
                    put_pred = put_time_value_pred + put_intrinsic_value
                    call_pred = put_pred + S*torch.exp(-d * tau) - K*torch.exp(-r*tau)
                    target_pred_fs = call_pred * torch.exp(r*tau) / S
            else:
                call_intrinsic_value = S * torch.exp(-d * tau) - K
                call_intrinsic_value[call_intrinsic_value < 0] = 0
                call_intrinsic_value = 0 # for OTM call with ATM call, force to not exercise the ATM part
                call_time_value_true = call_true - call_intrinsic_value
                target_true = call_time_value_true * torch.exp(r * tau) / S
                target_pred_fs = self.model(invm_fs, tau_fs)  # accord. to the paper: "though for numerical stability we train the model to (equivalently) minimise the difference on (exp(r*tau)*c/S_t , y)"
                if for_metric:
                    target_true = call_true * torch.exp(r * tau) / S
                    call_time_value_pred = target_pred_fs * S / torch.exp(r * tau)
                    call_pred = call_time_value_pred + call_intrinsic_value
                    target_pred_fs = call_pred * torch.exp(r * tau) / S
        else:
            target_true = call_true * torch.exp(r * tau) / S
            target_pred_fs = self.model(invm_fs, tau_fs) # accord. to the paper: "though for numerical stability we train the model to (equivalently) minimise the difference on (exp(r*tau)*c/S_t , y)"

        target_true_fs = feature_scale(x=target_true, shift=self.output_fscale_dict['shift'], mean=self.output_fscale_dict['mean'], std=self.output_fscale_dict['std'])
        return target_true_fs, target_pred_fs

    def _compute_call_price(self, dataset,syn_pred = False):
        if syn_pred:
            call_true = torch.zeros((dataset.shape[0],1), dtype=torch.float64, device=self.model.device)
        else:
            call_true = torch.tensor(dataset['option_price'].values, device=self.model.device)
        K = torch.tensor(dataset['strike_price'].values, device=self.model.device)
        tau = torch.tensor(dataset['tau'].values, device=self.model.device)
        S = torch.tensor(dataset['S'].values, device=self.model.device)
        r = torch.tensor(dataset['r'].values, device=self.model.device)
        d = torch.tensor(dataset['d'].values, device=self.model.device)
        _, target_pred_fs = self._compute_target(call_true, K, tau, S, r, d, for_metric=True,syn_pred = syn_pred)
        # return scaled output to unscaled output & return to call price by multiplying S*exp(-r*tau)
        # assert(torch.tensor(df['option_price'].values, device = self.model.device) == inverse_feature_scale(y=target_true_fs, shift=self.output_fscale_dict['shift'], mean=self.output_fscale_dict['mean'], std=self.output_fscale_dict['std'])*S*torch.exp(-r*tau))
        call_pred = inverse_feature_scale(y=target_pred_fs, shift=self.output_fscale_dict['shift'], mean=self.output_fscale_dict['mean'], std=self.output_fscale_dict['std']) * S * torch.exp(-r * tau)

        #if self.epoch % 20 == 0:
        #    print(f"pred_tar={target_pred_fs[:5].detach().cpu().numpy()}")
        #    print(f"pred_call={call_pred[:5].detach().cpu().numpy()}")
        return call_true, call_pred

    def _mean_square_error(self, true, pred, reduce):
        if reduce == 'none':
            res = (true - pred) ** 2
        else:
            res = nn.MSELoss(reduction = reduce)(pred, true) #torch.mean((true - pred) ** 2)
        return res

    def _mean_absolute_percentage_error(self, true, pred, reduce):
        # 1e-8 refer to pytorch_forecasting.metrics.MAPE(reduction='mean')(pred, true)
        ape = (true - pred).abs() / (true.abs() + 1e-8)
        if reduce == 'none':
            res = ape
        elif reduce == 'mean':
            res = torch.mean(ape)
        elif reduce == 'sum':
            res = torch.sum(ape)
        return res

    def _compute_metric(self, dataset):
        # in evaluation, we should only calculate market option's performance, not the synthesized one
        # the synthesized option contracts are just for training, not evaluating like validating or testing
        invm = (dataset['strike_price'] / dataset['S']).values
        mask = filter(invm, 'i') if self.residual_model_call_itm_part else filter(invm, 'o') if self.residual_on else True
        call_true, call_pred = self._compute_call_price(dataset[mask & (~dataset['is_syn'].values)])
        mse = self._mean_square_error(call_true, call_pred, 'mean')
        mape = self._mean_absolute_percentage_error(call_true, call_pred, 'mean')

        mask = filter(invm, 'i')
        call_true, call_pred = self._compute_call_price(dataset[mask & (~dataset['is_syn'].values)])
        itm_mse = self._mean_square_error(call_true, call_pred, 'mean')
        itm_mape = self._mean_absolute_percentage_error(call_true, call_pred, 'mean')

        mask = filter(invm, 'o')
        call_true, call_pred = self._compute_call_price(dataset[mask & (~dataset['is_syn'].values)])
        otm_mse = self._mean_square_error(call_true, call_pred, 'mean')
        otm_mape = self._mean_absolute_percentage_error(call_true, call_pred, 'mean')

        return mse.item(), mape.item(), itm_mse.item(), itm_mape.item(), otm_mse.item(), otm_mape.item()

    def _compute_loss(self, target_true_fs, target_pred_fs, K, S, tau, is_syn, for_train):
        if self.residual_on and for_train:
            key = self._transfer_to_index_of_loss_weight_table(invm=K/S, tau=tau)
            loss_weight = torch.from_numpy(self.loss_weight_table_for_residual_model.loc[key]['loss_weight'].values).to(self.model.device)
            loss_weight /= torch.sum(loss_weight)
            ape = self._mean_absolute_percentage_error(target_true_fs, target_pred_fs, 'none')
            se = self._mean_square_error(target_true_fs, target_pred_fs, 'none')
            mse = torch.sum(loss_weight * se)
            mape = torch.sum(loss_weight * ape)
        else:
            mape = self._mean_absolute_percentage_error(target_true_fs, target_pred_fs, 'mean')
            mse = self._mean_square_error(target_true_fs, target_pred_fs, 'mean')
        c2_loss = torch.tensor(0, device=self.model.device, dtype=torch.float64)  # arbitrage condition 2 penalty
        eps = torch.tensor(1e-3).to(self.model.device)
        invm = K/S
        if True in is_syn:
            g1 = self.model.first_diff_wrt_invm(invm[is_syn], tau[is_syn])
            g2 = self.model.first_diff_wrt_invm(invm[is_syn] + eps, tau[is_syn])
            if True in ((g1 - g2) > 0):
                c2_loss += torch.sum((g1 - g2)[(g1 - g2) > 0])  # C2 loss' value range is about in [1e-8, 1e-4]; accord. to paper
        return mse + mape + c2_loss # +1000*c2_loss

