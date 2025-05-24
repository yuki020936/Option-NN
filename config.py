result_folder = None
progress_bar = None
daily_gen = None
train_engine = None
args = None
moving_average_epoch = 50

ms_step_scheduler_setting = {
    'epoch_warmup_list': [0, 100],
    'epoch_decay': 300,
    'decay_times': 2,
    'epoch_last': 500,
    'init_lr_list': [5e-2, 1e-1], # initial learning rate # 1024batchsize_0.1LR_0_100warmup_1000decaylen_2decaytimes_500epochlast_0.5gamma_0_1seed # 32batchsize_0.05LR_300cyclen_0.25gamma_0_1seed_linear # 32batchsize_0.05LR_0_100warmup_300decaylen_2decaytimes_500epochlast_0.25gamma_0_1seed_test
    'gamma': 0.25,
    'cur_validate_folder': None,
    'cur_epoch_warmup': None,
    'cur_epoch_size': None
}

cyclic_shceduler_setting = {
    # cyclical learning rate should stay at the lowest point after lots of epoch of training learning rate to prevent test loss increase abruptly
    # hence we set cyclical scheduler's model to exp_range
    'cur_epoch_size': 1500,
    'lr_max_to_base_ratio': 0.0625, # 1/16
    'gamma': 0.99994,
    'cur_validate_folder': None
}

# fin_count is to print the bar progress
# loss queues is to avoid blocking as the event of KeyboardInterrupt occurs
# process_fd_table is to wait child processes
# process_list is to record child process that should be terminated as KeyboardInterrupt occurs
# avaliable_device make sure new process start after clear up exited process
procs_info = {'fin_count': 0, 'loss_queues': [], 'process_fd_table': [], 'process_list': [], 'avaliable_device': [], 'EOF': False}

model_loss_list_dict = {'date_list':[], 'train_loss':[], 'train_mse':[], 'train_mape':[], 'train_ITMmse':[],
                       'train_ITMmape':[], 'train_OTMmse':[], 'train_OTMmape':[],
                       'test_loss':[], 'test_mse':[], 'test_mape':[], 'test_ITMmse':[],
                       'test_ITMmape':[], 'test_OTMmse':[], 'test_OTMmape':[]}
