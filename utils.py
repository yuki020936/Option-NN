import numpy as np
import shutil
import os
import io
import gc
import ctypes
import json
import config
import dateutil
import itertools
import dateutil.parser as dparser
import pandas as pd
import random
import torch
import dask.array as da
import dask.dataframe as dd

def feature_scale(x, shift=0, mean=0, std=1):
    return shift + (x-mean)/std

def inverse_feature_scale(y, shift=0, mean=0, std=1):
    return mean + std*(y-shift)

def filter(invm, spec, threshold=1):
    # parse in the money, out the money, at the money
    if 'dask' in str(type(invm)):
        mask = invm < -999999
    else:
        mask = np.zeros_like(invm, dtype=bool)
    if 'i' in spec:
        mask = mask | (invm < threshold)
    if 'o' in spec:
        mask = mask | (invm > threshold)
    return mask

def seed_initializer(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    seed_initializer(worker_seed)

def release_memory() -> int:
    gc.collect()
    # libc = ctypes.CDLL("libc.so.6")
    # return libc.malloc_trim(0)
    ctypes.windll.kernel32.SetProcessWorkingSetSize(ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    return 0

def clear_files(resume: bool, sample_start_date: str, model_name: str, result_folder: str):
    if resume:
        result_file_name = f"avg_loss_of_model_{model_name}_among_different_dates.csv"
        sample_start_date = f'{sample_start_date[:4]}-{sample_start_date[4:6]}-{sample_start_date[6:]}'
        try:
            # retrive model_loss_list_dict and delete the contents after sample_start_date
            result = pd.read_csv(f"{result_folder}/{result_file_name}")
            drop_date_idx = np.where(np.array(result['date'], dtype=np.datetime64) >= np.datetime64(sample_start_date))[0] #找出sample_start_date之後的日期
            if len(drop_date_idx) > 1:
                retain_result = {k: list(v.values()) for k, v in result.rename(columns={'date': 'date_list'}).iloc[:drop_date_idx[0]].to_dict().items()}
                delete_date = result['date'][drop_date_idx[0]]
            else:
                retain_result = {k: list(v.values()) for k, v in result.rename(columns={'date': 'date_list'}).to_dict().items()}
                delete_date = str(np.datetime64(result['date'].iloc[-1]) + np.timedelta64(1, 'D'))
            config.model_loss_list_dict = retain_result
            # if len(drop_date_idx) == 0:
            #     return # delete nothing <--- wrong, ongoing training won't be recorded in avg_loss_of_model_{model_name}_among_different_dates.csv but has loss_*.png, loss_message_*.csv, etc.
            # delete the data after sample_start_date
            def delete_files_by_date_list(file_name):
                try:
                    if dparser.parse(file_name.name, fuzzy=True) >= dparser.parse(delete_date):
                       os.remove(file_name)
                except (dateutil.parser._parser.ParserError if hasattr(dateutil.parser._parser, 'ParserError') else ValueError) as e:
                    pass
            for dir_path1 in os.scandir(result_folder):
                if not os.path.isdir(dir_path1):
                    continue
                for dir_path2 in os.scandir(dir_path1):
                    if os.path.isdir(dir_path2):
                        for file_name in os.scandir(dir_path2):
                            delete_files_by_date_list(file_name=file_name)
                    else:
                        delete_files_by_date_list(file_name=dir_path2)

        except (FileNotFoundError, io.UnsupportedOperation) as e:
            print(f"Oops, some errors occur on {result_file_name}, {e}")
            exit()
    else:
        try:
            shutil.rmtree(result_folder)
        except FileNotFoundError:
            pass # fine
        os.makedirs(result_folder, exist_ok=True)

def merge_itm_otm(phase: str, itm_count: int, otm_count: int, itm_loss_dict: dict, otm_loss_dict: dict, split_itm_otm_loss: bool):
    # itm_loss_curve.to_dict('list')
    if itm_loss_dict.keys() != otm_loss_dict.keys():
        raise ValueError
    merge_loss_dict = dict()
    padding_length = 0
    longer_length = 0
    for k in itm_loss_dict.keys():
        if phase not in k or 'epoch' in k:
            continue
        if split_itm_otm_loss and '_loss' in k:
            merge_loss_dict.update({k[:k.index('_loss')] + "_ITMloss" + k[k.index('_loss') + len("_loss"):]: itm_loss_dict[k]})
            merge_loss_dict.update({k[:k.index('_loss')] + "_OTMloss" + k[k.index('_loss') + len("_loss"):]: otm_loss_dict[k]})
        if '_loss' in k or '_mse' in k or '_mape' in k:
            # itm_loss_dict's xxx_mse_list & xxx_mape_list is actually xxx_ITMmse_list & xxx_ITMmape_list
            # otm_loss_dict's xxx_mse_list & xxx_mape_list is actually xxx_OTMmse_list & xxx_OTMmape_list
            if isinstance(itm_loss_dict[k], list) and isinstance(otm_loss_dict[k], list) and (len(itm_loss_dict[k])!=len(otm_loss_dict[k])):
                # handle the case that the length of itm_loss_dict[k] list and otm_loss_dict[k] list are mismatched to avoid ValueError: operands could not be broadcast together with shapes (1400,) (1500,)
                if len(itm_loss_dict[k]) < len(otm_loss_dict[k]):
                    shorter_loss_list = itm_loss_dict[k]
                    shorter_loss_count = itm_count
                    longer_loss_list = otm_loss_dict[k]
                    longer_loss_count = otm_count
                else:
                    shorter_loss_list = otm_loss_dict[k]
                    shorter_loss_count = otm_count
                    longer_loss_list = itm_loss_dict[k]
                    longer_loss_count = itm_count
                padding_length = len(longer_loss_list) - len(shorter_loss_list) # padding_length should not equal to zero, or longer_loss[-padding_length:] will be not what I want
                longer_length = len(longer_loss_list)
                v = ((np.array(shorter_loss_list + longer_loss_list[-padding_length:]) * shorter_loss_count + np.array(longer_loss_list) * longer_loss_count) / (shorter_loss_count + longer_loss_count))
            else:
                v = ((np.array(itm_loss_dict[k]) * itm_count + np.array(otm_loss_dict[k]) * otm_count) / (itm_count + otm_count))
            v = list(v) if isinstance(v, np.ndarray) and len(v.shape) > 0 else float(v)
            merge_loss_dict.update({k: v})
        if '_ITMmse' in k or '_ITMmape' in k:
            merge_loss_dict.update({k: itm_loss_dict[k]})
        if '_OTMmse' in k or '_OTMmape' in k:
            merge_loss_dict.update({k: otm_loss_dict[k]})

    # padding
    if longer_length != 0:
        for k in merge_loss_dict:
            if len(merge_loss_dict[k]) < longer_length:
                merge_loss_dict[k] += [np.nan] * padding_length
    return merge_loss_dict

def compute_func(x):
    
    return x.compute() if 'dask' in str(type(x)) else x

