import numpy as np
import pandas as pd
import os

data_path = r'D:\課程\Option_NN\code\Dataset\prs_dataset_252tau.csv'
raw_data_1 = pd.read_csv(os.path.join(data_path,'part.5.csv'),index_col=False) 
raw_data_2 = pd.read_csv(os.path.join(data_path,'part.4.csv'),index_col=False) 

test_data_1 = raw_data_1[0:967199]

test_data_2 = raw_data_2[922097:1012240]

new_data = pd.concat((test_data_2, test_data_1), axis = 0)
new_data = new_data.reset_index(inplace=False)
new_data = new_data.drop(['index'],axis=1)

save_path = r'D:\課程\Option_NN\code\Dataset\prs_dataset_252tau.csv'
new_data.to_csv( ''.join([ save_path ,"/2021.csv"]), index=False)
