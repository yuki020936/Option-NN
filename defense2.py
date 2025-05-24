import numpy as np
import pandas as pd

data_path_MM = r'D:\課程\Option_NN\defense\fitting_result_multi_model_20160608\2016-06-08_test_fit_result.csv'
df_MM = pd.read_csv(data_path_MM, index_col=False)

df_MM['moneyness'] = df_MM['strike_price']/df_MM['S']
df_MM = df_MM[(df_MM['moneyness'] > 0.8) & (df_MM['moneyness'] < 1.2)]

up_MM = df_MM['test_se'].sum()
down_MM = (df_MM['call_true']**2).sum()

MSE_MM = up_MM/df_MM.shape[0]
NRMSE_MM = np.sqrt(up_MM/down_MM)

data_path_R_itm = r'D:\課程\Option_NN\defense\fitting_result_itm\2016-06-08_test_itm_fit_result.csv'
df_itm = pd.read_csv(data_path_R_itm, index_col=False)
data_path_R_otm = r'D:\課程\Option_NN\defense\fitting_result_otm\2016-06-08_test_otm_fit_result.csv'
df_otm = pd.read_csv(data_path_R_otm, index_col=False)

itm_drop = ['tau', 'r', 'd', 'test_ITMse', 'test_ITMape', 'test_ITMmse', 'test_ITMmape']
df_itm.drop(columns=itm_drop, inplace=True)
otm_drop = ['tau', 'r', 'd', 'test_OTMse', 'test_OTMape', 'test_OTMmse', 'test_OTMmape']
df_otm.drop(columns=otm_drop, inplace=True)

df_R = pd.concat([df_itm, df_otm], axis=0, ignore_index=True)

df_R['moneyness'] = df_R['strike_price']/df_R['S']
df_R = df_R[(df_R['moneyness'] > 0.8) & (df_R['moneyness'] < 1.2)]

df_R['test_se'] = (df_R['call_pred']-df_R['call_true'])**2
up_R = df_R['test_se'].sum()
down_R = (df_R['call_true']**2).sum()

MSE_R = up_R/df_R.shape[0]
NRMSE_R = np.sqrt(up_R/down_R)
