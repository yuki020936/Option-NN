import pdb
import numpy as np
import os
from time import sleep
import signal
import sys
import queue
import shutil
import psutil
from multiprocessing.connection import wait
import torch.multiprocessing as mp
import torch
from argparse import ArgumentParser
from dataset import DataProcessor
from utils import clear_files
from visulization import plot_model_avg_loss
import config
from train import TrainEngine
from tqdm import tqdm
import copy

#torch.multiprocessing.set_sharing_strategy('file_system')
def single_running_worker():
    try:
        while True:
            # run model
            test_date, train_loss, train_mse, train_mape, train_ITMmse, train_ITMmape, train_OTMmse, train_OTMmape, test_loss, test_mse, test_mape, test_ITMmse, test_ITMmape, test_OTMmse, test_OTMmape = config.train_engine(*next(config.daily_gen)(0))
            print(f"date: {test_date}, train_mse: {train_mse:.3f}, train_mape: {train_mape:.3f}, test_mse: {test_mse:.3f}, test_mape: {test_mape:.3f}",flush=True)
            # record and update
            for v, k in zip([test_date, train_loss, train_mse, train_mape, train_ITMmse, train_ITMmape, train_OTMmse, train_OTMmape, test_loss, test_mse, test_mape, test_ITMmse, test_ITMmape, test_OTMmse, test_OTMmape], config.model_loss_list_dict):
                config.model_loss_list_dict[k] += [v]
            plot_model_avg_loss(config.result_folder, config.args.model_type, config.model_loss_list_dict)
    except StopIteration:
        pass  # EOF of generator, fine

def multi_running_check_in(rank):
    # rank is to allocate particular device
    # torch.distributed.init_process_group("nccl", rank=rank, world_size=torch.cuda.device_count()))
    try:
        q = mp.Queue()
        p = mp.Process(target=multi_running_job, args=(next(config.daily_gen)(rank), q, config.train_engine))
    except StopIteration:
        config.procs_info['EOF'] = True
        return # do nothing. that is, no new process will be allocated afterwards.

    # if we don't update these process info before create new child process, parent process may receive SIGCHLD signal while config.procs_info['process_fd_table'] does not include this new process. hence parent process fail to wait this child process. however, it (child process exits immediately) rarely happens.
    p.start()
    config.procs_info['loss_queues'].append(q)
    config.procs_info['process_list'].append(p)
    config.procs_info['process_fd_table'].append(p.sentinel)

def multi_running_job(daily_materials, loss_queue, train_engine):
    # register signal as we hope SIGINT signal can only be processed by parent process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # run model
    test_date, train_loss, train_mse, train_mape, train_ITMmse, train_ITMmape, train_OTMmse, train_OTMmape, test_loss, test_mse, test_mape, test_ITMmse, test_ITMmape, test_OTMmse, test_OTMmape = train_engine(*daily_materials)
    print(f"[process {os.getpid()}] date: {test_date}, train_mse: {train_mse:.3f}, train_mape: {train_mape:.3f}, test_mse: {test_mse:.3f}, test_mape: {test_mape:.3f}",flush=True)
    # return results
    loss_queue.put([train_engine.model.device, test_date, train_loss, train_mse, train_mape, train_ITMmse, train_ITMmape, train_OTMmse, train_OTMmape, test_loss, test_mse, test_mape, test_ITMmse, test_ITMmape, test_OTMmse, test_OTMmape])

def sigchld_handler(signum, frame):
    # adopt the exited child processes, collect results, identify avaliable devices
    # since signal handler allows signal reentrancy, it may require atomic operation. but it's not safe to use lock in signal handler, refer to https://stackoverflow.com/questions/12445618/accessing-shared-data-from-a-signal-handler
    # hence here ignore SIGCHLD signal before finishing cleaning up the process. note that new process should not be created before leaving sigchld_handler
    # wait() will be always valid after child process status becomes exited unless pop out the process in config.procs_info['process_fd_table']
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    while True:
        p_sentns = wait(config.procs_info['process_fd_table'], 0)
        if len(p_sentns) == 0:
            break
        # for each exited child process
        for p_sentn in p_sentns:
            # collect results
            idx = config.procs_info['process_fd_table'].index(p_sentn)
            # to prevent if one of the child process unexpectedly is killed and hence no results can be push into the queue, the parent process will be blocked in this line
            try:
                rank, test_date, *losses = config.procs_info['loss_queues'][idx].get(block=False)  # "hang" & Remove and return an item from the queue. If optional args block is True (the default) and timeout is None (the default), "block" if necessary until an item is available.
            except queue.Empty:
                print("at least one of the child process crashed, exit!")
                os.kill(os.getpid(), signal.SIGINT)
            for v, k in zip([test_date]+losses, config.model_loss_list_dict):
                config.model_loss_list_dict[k] += [v]

            # update process info
            config.procs_info['process_list'].pop(idx)
            config.procs_info['loss_queues'].pop(idx)
            config.procs_info['process_fd_table'].pop(idx)

            # update progress bar and print results
            plot_model_avg_loss(config.result_folder, config.args.model_type, {
                k: [l for _, l in sorted(zip(config.model_loss_list_dict['date_list'], config.model_loss_list_dict[k]))]
                for k in config.model_loss_list_dict.keys()})

            if config.procs_info['fin_count'] % config.args.n_parallel_process == (config.args.n_parallel_process - 1) and config.procs_info['fin_count'] != 0:
                config.progress_bar.update()
            config.procs_info['fin_count'] = config.procs_info['fin_count'] + 1

            # identify avaliable deiveces to allocate new process onto the rank which the exited process has stayed
            config.procs_info['avaliable_device'].append(rank)
    signal.signal(signal.SIGCHLD, sigchld_handler) # remember to reset the handler again, since we don't want to actually ignore SIGCHLD signal afterwards

def sigint_handler(signum, frame):
    # ctrl+c by user
    # clear results in queues and terminate all the child process immediately
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    print("catch KeyboardInterrupt, terminate all the child processes ...")
    for p, q in zip(config.procs_info['process_list'], config.procs_info['loss_queues']):
        try:
            _ = q.get(
                block=False)  # Warning If a process is killed using Process.terminate() or os.kill() while it is trying to use a Queue, then the data in the queue is likely to become corrupted. This may cause any other process to get an exception when it tries to use the queue later on.
        except queue.Empty:
            pass  # fine
        finally:
            p.terminate()  # send SIGTERM signal to the child process
            p.join()  # "hang" and adopt the child process
            print(f"child process {p.pid} terminated.")
    sys.exit(0)
    # original_sigint_handler = signal.getsignal(signal.SIGINT)
    # original_sigint_handler()

def multi_running_worker(test_date_list):
    # TODO: run on multiple GPUs: check the bugs related to torch.distributed.init_process_group & torch.nn.parallel.DistributedDataParallel
    config.progress_bar = tqdm(test_date_list[::config.args.n_parallel_process])
    config.progress_bar.reset()
    signal.signal(signal.SIGCHLD, sigchld_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    # initialize
    for i in range(config.args.n_parallel_process):
        multi_running_check_in(rank=i % torch.cuda.device_count() if torch.cuda.is_available() and config.args.on_gpu else -1)

    while True:
        if config.procs_info['EOF'] and len(config.procs_info['process_list']) == 0: # both conditions should be met
            config.progress_bar.update()
            break
        elif len(config.procs_info['process_list']) < config.args.n_parallel_process and len(config.procs_info['avaliable_device']) > 0:
            multi_running_check_in(rank=config.procs_info['avaliable_device'].pop(0))
        else:
            # suspend until receiving a signal when there still exist at least one process
            signal.pause()
    config.progress_bar.close()

def main():
    # ----------Hyper Parameters----------#
    parser = ArgumentParser()
    parser.add_argument('--result_folder_tag', type=str, default='test')
    parser.add_argument('--on_gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_step_lr', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--raw_dataset_path', type=str, default='../Dataset')
    parser.add_argument('--seed_list', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--sample_start_date', type=str, default='20160601')
    parser.add_argument('--sample_times', type=int, default=36)
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--n_parallel_process', type=int, default=1)
    parser.add_argument('--model_type', type=str, default="multi", choices=['multi', 'single'])
    parser.add_argument('--daily_plot_off', action='store_true')
    parser.add_argument('--residual_on', action='store_true')
    parser.add_argument('--converge_delta', type=float, default=1e-5)
    parser.add_argument('--converge_patience', type=int, default=300)
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='16235')
    # formula of rank
    config.args = parser.parse_args()

    config.result_folder = f"../{config.args.batch_size}batchsize"
    if config.args.use_step_lr:
        scheduler_setting = config.ms_step_scheduler_setting
        config.result_folder += f"_{'_'.join(str(lr) for lr in scheduler_setting['init_lr_list'])}initLR_{'_'.join(str(w) for w in scheduler_setting['epoch_warmup_list'])}warmup_{scheduler_setting['epoch_decay']}decaylen_{scheduler_setting['decay_times']}decaytimes_{scheduler_setting['epoch_last']}epochlast"
    else: # cyclical lr
        scheduler_setting = config.cyclic_shceduler_setting
        config.result_folder += f"_autoLR_autoCyclen_{scheduler_setting['cur_epoch_size']}epochsize_{scheduler_setting['lr_max_to_base_ratio']}MtoBratio"
    config.result_folder += f"_{scheduler_setting['gamma']}gamma_{'_'.join(str(s) for s in config.args.seed_list)}seed"
    config.result_folder += '_residual' if config.args.residual_on else '_multi'
    config.result_folder += f'_{config.args.result_folder_tag}'

    clear_files(config.args.resume, config.args.sample_start_date, config.args.model_type, config.result_folder)

    print("To reproduce AAAI Gated-neural-networks-for-option-pricing paper results")
    print(config.result_folder)

    # ----------Initialization---------- #

    mp.set_start_method('spawn')
    os.environ['MASTER_ADDR'] = config.args.master_addr
    os.environ['MASTER_PORT'] = config.args.master_port
    data_processor = DataProcessor(root_path=config.args.raw_dataset_path, model_type='multi', sample_start_date=config.args.sample_start_date, sample_times=config.args.sample_times, sample_interval=config.args.sample_interval, on_gpu=config.args.on_gpu)
    test_date_list, config.daily_gen, IAO_threshold = data_processor(config.args.window_size, config.args.batch_size, config.args.num_workers, config.args.n_parallel_process, config.args.residual_on, config.args.seed_list)
    config.train_engine = TrainEngine(config.args.daily_plot_off, config.args.residual_on, config.args.use_step_lr, scheduler_setting, config.args.converge_delta, config.args.converge_patience, config.args.seed_list, IAO_threshold, config.result_folder)

    # ----------For each testing date---------- #
    if config.args.n_parallel_process == 1:
        single_running_worker()

    else:
        multi_running_worker(test_date_list)

if __name__ == '__main__':
    main()
