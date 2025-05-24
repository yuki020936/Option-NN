import pandas as pd
import os

data_path = r'D:\課程\Option_NN\code\Dataset\options_1996_2021.csv'
df = pd.read_csv(data_path, index_col=False)
ul_path = r'D:\課程\Option_NN\code\Dataset\underlyings.csv'
ul = pd.read_csv(ul_path, index_col=False)

df_2010 = df[(df['date'] == 20100607) | (df['date'] == 20100608) | (df['date'] == 20100609) | (df['date'] == 20100610) | (df['date'] == 20100611)]
df_2012 = df[(df['date'] == 20120604) | (df['date'] == 20120605) | (df['date'] == 20120606) | (df['date'] == 20120607) | (df['date'] == 20120608)]
df_2014 = df[(df['date'] == 20140602) | (df['date'] == 20140603) | (df['date'] == 20140604) | (df['date'] == 20140605) | (df['date'] == 20140606)]
df_2016 = df[(df['date'] == 20160606) | (df['date'] == 20160607) | (df['date'] == 20160608) | (df['date'] == 20160609) | (df['date'] == 20160610)]

df_2010 = df_2010[df_2010['volume'] != 0]
df_2012 = df_2012[df_2012['volume'] != 0]
df_2014 = df_2014[df_2014['volume'] != 0]
df_2016 = df_2016[df_2016['volume'] != 0]

ul_2010 = ul[(ul['Date'] == '2010-06-07') | (ul['Date'] == '2010-06-08') | (ul['Date'] == '2010-06-09') | (ul['Date'] == '2010-06-10') | (ul['Date'] == '2010-06-11')]
ul_2012 = ul[(ul['Date'] == '2012-06-04') | (ul['Date'] == '2012-06-05') | (ul['Date'] == '2012-06-06') | (ul['Date'] == '2012-06-07') | (ul['Date'] == '2012-06-08')]
ul_2014 = ul[(ul['Date'] == '2014-06-02') | (ul['Date'] == '2014-06-03') | (ul['Date'] == '2014-06-04') | (ul['Date'] == '2014-06-05') | (ul['Date'] == '2014-06-06')]
ul_2016 = ul[(ul['Date'] == '2016-06-06') | (ul['Date'] == '2016-06-07') | (ul['Date'] == '2016-06-08') | (ul['Date'] == '2016-06-09') | (ul['Date'] == '2016-06-10')]

df_2010['date'] = pd.to_datetime(df_2010['date'], format='%Y%m%d')
df_2010['exdate'] = pd.to_datetime(df_2010['exdate'], format='%Y%m%d')
df_2010['tau'] = (df_2010['exdate'] - df_2010['date']).dt.days.astype(int)/252
df_2010['moneyness'] = ""
df_2010.loc[df_2010['date'] == '2010-06-07', 'moneyness'] = (df_2010['strike_price']/1000)/ul_2010['Close'].iloc[0]
df_2010.loc[df_2010['date'] == '2010-06-08', 'moneyness'] = (df_2010['strike_price']/1000)/ul_2010['Close'].iloc[1]
df_2010.loc[df_2010['date'] == '2010-06-09', 'moneyness'] = (df_2010['strike_price']/1000)/ul_2010['Close'].iloc[2]
df_2010.loc[df_2010['date'] == '2010-06-10', 'moneyness'] = (df_2010['strike_price']/1000)/ul_2010['Close'].iloc[3]
df_2010.loc[df_2010['date'] == '2010-06-11', 'moneyness'] = (df_2010['strike_price']/1000)/ul_2010['Close'].iloc[4]

df_2012['date'] = pd.to_datetime(df_2012['date'], format='%Y%m%d')
df_2012['exdate'] = pd.to_datetime(df_2012['exdate'], format='%Y%m%d')
df_2012['tau'] = (df_2012['exdate'] - df_2012['date']).dt.days.astype(int)/252
df_2012['moneyness'] = ""
df_2012.loc[df_2012['date'] == '2012-06-04', 'moneyness'] = (df_2012['strike_price']/1000)/ul_2012['Close'].iloc[0]
df_2012.loc[df_2012['date'] == '2012-06-05', 'moneyness'] = (df_2012['strike_price']/1000)/ul_2012['Close'].iloc[1]
df_2012.loc[df_2012['date'] == '2012-06-06', 'moneyness'] = (df_2012['strike_price']/1000)/ul_2012['Close'].iloc[2]
df_2012.loc[df_2012['date'] == '2012-06-07', 'moneyness'] = (df_2012['strike_price']/1000)/ul_2012['Close'].iloc[3]
df_2012.loc[df_2012['date'] == '2012-06-08', 'moneyness'] = (df_2012['strike_price']/1000)/ul_2012['Close'].iloc[4]

df_2014['date'] = pd.to_datetime(df_2014['date'], format='%Y%m%d')
df_2014['exdate'] = pd.to_datetime(df_2014['exdate'], format='%Y%m%d')
df_2014['tau'] = (df_2014['exdate'] - df_2014['date']).dt.days.astype(int)/252
df_2014['moneyness'] = ""
df_2014.loc[df_2014['date'] == '2014-06-02', 'moneyness'] = (df_2014['strike_price']/1000)/ul_2014['Close'].iloc[0]
df_2014.loc[df_2014['date'] == '2014-06-03', 'moneyness'] = (df_2014['strike_price']/1000)/ul_2014['Close'].iloc[1]
df_2014.loc[df_2014['date'] == '2014-06-04', 'moneyness'] = (df_2014['strike_price']/1000)/ul_2014['Close'].iloc[2]
df_2014.loc[df_2014['date'] == '2014-06-05', 'moneyness'] = (df_2014['strike_price']/1000)/ul_2014['Close'].iloc[3]
df_2014.loc[df_2014['date'] == '2014-06-06', 'moneyness'] = (df_2014['strike_price']/1000)/ul_2014['Close'].iloc[4]

df_2016['date'] = pd.to_datetime(df_2016['date'], format='%Y%m%d')
df_2016['exdate'] = pd.to_datetime(df_2016['exdate'], format='%Y%m%d')
df_2016['tau'] = (df_2016['exdate'] - df_2016['date']).dt.days.astype(int)/252
df_2016['moneyness'] = ""
df_2016.loc[df_2016['date'] == '2016-06-06', 'moneyness'] = (df_2016['strike_price']/1000)/ul_2016['Close'].iloc[0]
df_2016.loc[df_2016['date'] == '2016-06-07', 'moneyness'] = (df_2016['strike_price']/1000)/ul_2016['Close'].iloc[1]
df_2016.loc[df_2016['date'] == '2016-06-08', 'moneyness'] = (df_2016['strike_price']/1000)/ul_2016['Close'].iloc[2]
df_2016.loc[df_2016['date'] == '2016-06-09', 'moneyness'] = (df_2016['strike_price']/1000)/ul_2016['Close'].iloc[3]
df_2016.loc[df_2016['date'] == '2016-06-10', 'moneyness'] = (df_2016['strike_price']/1000)/ul_2016['Close'].iloc[4]

columns_to_drop = ['secid', 'date', 'symbol', 'exdate', 'last_date', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'optionid', 'index_flag', 'issuer', 'exercise_style']
df_2010.drop(columns=columns_to_drop, inplace=True)
df_2012.drop(columns=columns_to_drop, inplace=True)
df_2014.drop(columns=columns_to_drop, inplace=True)
df_2016.drop(columns=columns_to_drop, inplace=True)

df_2010 = df_2010[df_2010['tau'] >= 2/252]
df_2012 = df_2012[df_2012['tau'] >= 2/252]
df_2014 = df_2014[df_2014['tau'] >= 2/252]
df_2016 = df_2016[df_2016['tau'] >= 2/252]

df_2010 = df_2010[((df_2010['cp_flag'] == 'C') & (df_2010['moneyness'] > 1)) | ((df_2010['cp_flag'] == 'P') & (df_2010['moneyness'] < 1))]
df_2012 = df_2012[((df_2012['cp_flag'] == 'C') & (df_2012['moneyness'] > 1)) | ((df_2012['cp_flag'] == 'P') & (df_2012['moneyness'] < 1))]
df_2014 = df_2014[((df_2014['cp_flag'] == 'C') & (df_2014['moneyness'] > 1)) | ((df_2014['cp_flag'] == 'P') & (df_2014['moneyness'] < 1))]
df_2016 = df_2016[((df_2016['cp_flag'] == 'C') & (df_2016['moneyness'] > 1)) | ((df_2016['cp_flag'] == 'P') & (df_2016['moneyness'] < 1))]

df_2010.reset_index(drop=True, inplace=True)
df_2012.reset_index(drop=True, inplace=True)
df_2014.reset_index(drop=True, inplace=True)
df_2016.reset_index(drop=True, inplace=True)

save_path = r'D:\課程\Option_NN\defense'
df_2010.to_csv( ''.join([ save_path ,"/", "2010.csv"]))
df_2012.to_csv( ''.join([ save_path ,"/", "2012.csv"]))
df_2014.to_csv( ''.join([ save_path ,"/", "2014.csv"]))
df_2016.to_csv( ''.join([ save_path ,"/", "2016.csv"]))

df_20100614 = df[df['date'] == 20100614]
df_20100614 = df_20100614[df_20100614['volume'] != 0]
ul_20100614 = ul[ul['Date'] == '2010-06-14']
df_20100614['date'] = pd.to_datetime(df_20100614['date'], format='%Y%m%d')
df_20100614['exdate'] = pd.to_datetime(df_20100614['exdate'], format='%Y%m%d')
df_20100614['tau'] = (df_20100614['exdate'] - df_20100614['date']).dt.days.astype(int)/252
df_20100614['moneyness'] = ""
df_20100614.loc[df_20100614['date'] == '2010-06-14', 'moneyness'] = (df_20100614['strike_price']/1000)/ul_20100614['Close'].iloc[0]
columns_to_drop = ['secid', 'date', 'symbol', 'exdate', 'last_date', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'optionid', 'index_flag', 'issuer', 'exercise_style']
df_20100614.drop(columns=columns_to_drop, inplace=True)
df_20100614 = df_20100614[df_20100614['tau'] >= 2/252]
df_20100614 = df_20100614[((df_20100614['cp_flag'] == 'C') & (df_20100614['moneyness'] > 1)) | ((df_20100614['cp_flag'] == 'P') & (df_20100614['moneyness'] < 1))]
df_20100614.reset_index(drop=True, inplace=True)
df_20100614.to_csv( ''.join([ save_path ,"/", "20100614.csv"]))

df_20120611 = df[df['date'] == 20120611]
df_20120611 = df_20120611[df_20120611['volume'] != 0]
ul_20120611 = ul[ul['Date'] == '2012-06-11']
df_20120611['date'] = pd.to_datetime(df_20120611['date'], format='%Y%m%d')
df_20120611['exdate'] = pd.to_datetime(df_20120611['exdate'], format='%Y%m%d')
df_20120611['tau'] = (df_20120611['exdate'] - df_20120611['date']).dt.days.astype(int)/252
df_20120611['moneyness'] = ""
df_20120611.loc[df_20120611['date'] == '2012-06-11', 'moneyness'] = (df_20120611['strike_price']/1000)/ul_20120611['Close'].iloc[0]
columns_to_drop = ['secid', 'date', 'symbol', 'exdate', 'last_date', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'optionid', 'index_flag', 'issuer', 'exercise_style']
df_20120611.drop(columns=columns_to_drop, inplace=True)
df_20120611 = df_20120611[df_20120611['tau'] >= 2/252]
df_20120611 = df_20120611[((df_20120611['cp_flag'] == 'C') & (df_20120611['moneyness'] > 1)) | ((df_20120611['cp_flag'] == 'P') & (df_20120611['moneyness'] < 1))]
df_20120611.reset_index(drop=True, inplace=True)
df_20120611.to_csv( ''.join([ save_path ,"/", "20120611.csv"]))

df_20140609 = df[df['date'] == 20140609]
df_20140609 = df_20140609[df_20140609['volume'] != 0]
ul_20140609 = ul[ul['Date'] == '2014-06-09']
df_20140609['date'] = pd.to_datetime(df_20140609['date'], format='%Y%m%d')
df_20140609['exdate'] = pd.to_datetime(df_20140609['exdate'], format='%Y%m%d')
df_20140609['tau'] = (df_20140609['exdate'] - df_20140609['date']).dt.days.astype(int)/252
df_20140609['moneyness'] = ""
df_20140609.loc[df_20140609['date'] == '2014-06-09', 'moneyness'] = (df_20140609['strike_price']/1000)/ul_20140609['Close'].iloc[0]
columns_to_drop = ['secid', 'date', 'symbol', 'exdate', 'last_date', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'optionid', 'index_flag', 'issuer', 'exercise_style']
df_20140609.drop(columns=columns_to_drop, inplace=True)
df_20140609 = df_20140609[df_20140609['tau'] >= 2/252]
df_20140609 = df_20140609[((df_20140609['cp_flag'] == 'C') & (df_20140609['moneyness'] > 1)) | ((df_20140609['cp_flag'] == 'P') & (df_20140609['moneyness'] < 1))]
df_20140609.reset_index(drop=True, inplace=True)
df_20140609.to_csv( ''.join([ save_path ,"/", "20140609.csv"]))

df_20160613 = df[df['date'] == 20160613]
df_20160613 = df_20160613[df_20160613['volume'] != 0]
ul_20160613 = ul[ul['Date'] == '2016-06-13']
df_20160613['date'] = pd.to_datetime(df_20160613['date'], format='%Y%m%d')
df_20160613['exdate'] = pd.to_datetime(df_20160613['exdate'], format='%Y%m%d')
df_20160613['tau'] = (df_20160613['exdate'] - df_20160613['date']).dt.days.astype(int)/252
df_20160613['moneyness'] = ""
df_20160613.loc[df_20160613['date'] == '2016-06-13', 'moneyness'] = (df_20160613['strike_price']/1000)/ul_20160613['Close'].iloc[0]
columns_to_drop = ['secid', 'date', 'symbol', 'exdate', 'last_date', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'optionid', 'index_flag', 'issuer', 'exercise_style']
df_20160613.drop(columns=columns_to_drop, inplace=True)
df_20160613 = df_20160613[df_20160613['tau'] >= 2/252]
df_20160613 = df_20160613[((df_20160613['cp_flag'] == 'C') & (df_20160613['moneyness'] > 1)) | ((df_20160613['cp_flag'] == 'P') & (df_20160613['moneyness'] < 1))]
df_20160613.reset_index(drop=True, inplace=True)
df_20160613.to_csv( ''.join([ save_path ,"/", "20160613.csv"]))