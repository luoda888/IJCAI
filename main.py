#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:27:33 2018

@author: XFBY
"""
import pandas as pd
import numpy as np
import time
import datetime
import gc
import copy
from math import radians, cos, sin, asin, sqrt
from collections import Counter
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import preprocessing


all_col = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION',
       'HEIGHT', 'SPEED', 'CALLSTATE', 'Y','week','trip_idx','call_statex']


use_cols = ['TERMINALNO-用户id', 'TIME-时间戳', 'TRIP_ID-行程id', 'LONGITUDE-经度', 
            'LATITUDE-纬度', 'DIRECTION-方向角度','HEIGHT-海拔', 'SPEED-速度', 
            'CALLSTATE-电话状态', 'Y-客户赔付率']

f_col = ['maxspeed', 'minspeed', 'speed_4', 'speed_1', 'speed_std',
       'speed_kurt', 'speed_skew', 'star_end_num', 's_e_time_mean',
       's_e_time_max', 's_e_time_min', 's_e_time_std', 'direction_std',
       's_e_direction_std_mean', 's_e_direction_std_max',
       's_e_direction_std_min', 's_e_direction_std_4', 's_e_speed_mean',
       's_e_speed_max', 's_e_speed_min', 's_e_speed_4', 's_e_speed_1',
       's_e_speed_std', 's_e_speed_std_max', 's_e_speed_std_min',
       's_e_speed_std_4', 's_e_speed_std_1', 's_e_speed_kurt',
       's_e_speed_skew', 's_e_trip_num', 's_e_tripid_mean', 's_e_tripid_std',
       's_e_miles_max', 's_e_miles_min', 's_e_miles_mean',
       's_e_hight_std_mean', 'call_out_num', 'call_out_times_4',
       'call_times_all', 'call_drive', 'weeken_time']

f_col_test = ['maxspeed', 'minspeed', 'speed_4', 'speed_1', 'speed_std',
       'speed_kurt', 'speed_skew', 'star_end_num', 's_e_time_mean',
       's_e_time_max', 's_e_time_min', 's_e_time_std', 'direction_std',
       's_e_direction_std_mean', 's_e_direction_std_max',
       's_e_direction_std_min', 's_e_direction_std_4', 's_e_speed_mean',
       's_e_speed_max', 's_e_speed_min', 's_e_speed_4', 's_e_speed_1',
       's_e_speed_std', 's_e_speed_std_max', 's_e_speed_std_min',
       's_e_speed_std_4', 's_e_speed_std_1', 's_e_speed_kurt',
       's_e_speed_skew', 's_e_trip_num', 's_e_tripid_mean', 's_e_tripid_std',
       's_e_miles_max', 's_e_miles_min', 's_e_miles_mean',
       's_e_hight_std_mean', 'call_out_num', 'call_out_times_4',
       'call_times_all', 'call_drive', 'weeken_time','Id']


    
def st_end(g_df,gap=5):
    ttx = np.array(g_df.index)
    c = list(np.diff(ttx)/np.timedelta64(1, 'm'))
    c.insert(0,1);g_df.loc[:,'temp'] = c
    start_t1 = g_df[g_df.temp >= gap].index.tolist()
    start_t2 = copy.deepcopy(start_t1)
    start_t2.insert(0,pd.to_datetime('1988-4-24 12:51:00'))
    start_t1.append(pd.to_datetime('2025-4-24 12:51:00'))
    result = [g_df[i:j][:-1] for (i,j) in zip(start_t2,start_t1)]
    return result
    
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """  
    # 将十进制度数转化为弧度  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
  
    # haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    return c * r
    
def test(g_df):
    l = []
    l.append(g_df.shape[0])
    re = Counter(l)
    print(re[0])
    print(re)
        
    
def maxspeed(g_df):
    return g_df.SPEED.max()
    
def minspeed(g_df):
    return g_df.SPEED.min()
    
def meanspeed(g_df):
    return g_df.SPEED.mean()
    
def speeds_4(g_df):
    return g_df.SPEED.quantile(0.4) 
    
def speeds_1(g_df):
    return g_df.SPEED.quantile(0.1)
    
def speeds_std(g_df):
    return g_df.SPEED.std()
    
def speeds_kurt(g_df):
    return g_df.SPEED.kurt()
    
def speeds_skew(g_df):
    return g_df.SPEED.skew()
    
def star_end_nums(g_df,gap=5):
    return len(st_end(g_df))
    
def s_e_time_means(g_df):
    result = st_end(g_df)
    t = [temp.shape[0] for temp in result]#minite
    try:
        return sum(t)/len(t)
    except:
        return 0
    
def s_e_time_maxs(g_df):
    result = st_end(g_df)
    t = [temp.shape[0] for temp in result]#minite
    return max(t)
    
def s_e_time_mins(g_df):
    result = st_end(g_df)
    t = [temp.shape[0] for temp in result]#minite
    return min(t)
    
def s_e_time_stds(g_df):
    result = st_end(g_df)
    t = [temp.shape[0] for temp in result]#minite
    data=pd.DataFrame({'temp':t})
    return data.temp.std()
    
def direction_stds(g_df):
    return g_df.DIRECTION.std()
    
def s_e_direction_std_means(g_df):
    result = st_end(g_df)
    t = [temp.DIRECTION.std() for temp in result]
    try:
        return sum(t)/len(t)
    except:
        return 0
    
def s_e_direction_std_maxs(g_df):
    result = st_end(g_df)
    t = [temp.DIRECTION.std() for temp in result]
    return max(t)
    
def s_e_direction_std_mins(g_df):
    result = st_end(g_df)
    t = [temp.DIRECTION.std() for temp in result]
    return min(t)
    
def s_e_direction_std_4s(g_df):
    result = st_end(g_df)
    t = [temp.DIRECTION.std() for temp in result]
    data=pd.DataFrame({'temp':t})
    return data.temp.quantile(0.4) 
    
def s_e_speed_means(g_df):
    return g_df.SPEED.mean()
    
def s_e_speed_maxs(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.mean() for temp in result]
    return max(t)
    
def s_e_speed_mins(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.mean() for temp in result]
    return min(t)
    
def s_e_speed_4s(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.mean() for temp in result]
    data=pd.DataFrame({'temp':t})
    return data.temp.quantile(0.4)
    
def s_e_speed_1s(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.mean() for temp in result]
    data=pd.DataFrame({'temp':t})
    return data.temp.quantile(0.1)
    
def s_e_speed_stds(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.std() for temp in result]
    try:
        return sum(t)/len(t)
    except:
        return 0
    
def s_e_speed_std_maxs(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.std() for temp in result]
    return max(t)
    
def s_e_speed_std_mins(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.std() for temp in result]
    return min(t)
    
def s_e_speed_std_4s(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.std() for temp in result]
    data=pd.DataFrame({'temp':t})
    return data.temp.quantile(0.4)
    
def s_e_speed_std_1s(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.std() for temp in result]
    data=pd.DataFrame({'temp':t})
    return data.temp.quantile(0.1)
    
def s_e_speed_kurts(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.kurt() for temp in result]
    try:
        return sum(t)/len(t)
    except:
        return 0
    
def s_e_speed_skews(g_df):
    result = st_end(g_df)
    t = [temp.SPEED.skew() for temp in result]
    try:
        return sum(t)/len(t)
    except:
        return 0
    
def s_e_trip_nums(g_df):
    result = st_end(g_df)
    t = [len(set(temp.TRIP_ID)) for temp in result]
    return sum(t)
    
def s_e_tripid_means(g_df):
    result = st_end(g_df)
    t = [len(set(temp.TRIP_ID)) for temp in result]
    try:
        return sum(t)/len(t)
    except:
        return 0
    
def s_e_tripid_stds(g_df):
    return g_df.TRIP_ID.std()
    
def s_e_miles_maxs(g_df):
    result = st_end(g_df)
    t = [haversine(temp.LONGITUDE[0],temp.LATITUDE[0],temp.LONGITUDE[-1],
                            temp.LATITUDE[-1]) for temp in result if temp.shape[0]>0]
    return max(t)
    
def s_e_miles_mins(g_df):
    result = st_end(g_df)
    t = [haversine(temp.LONGITUDE[0],temp.LATITUDE[0],temp.LONGITUDE[-1],
                        temp.LATITUDE[-1]) for temp in result if temp.shape[0]>0]
    return min(t)
    
def s_e_miles_means(g_df):
    result = st_end(g_df)
    t = [haversine(temp.LONGITUDE[0],temp.LATITUDE[0],temp.LONGITUDE[-1],
                        temp.LATITUDE[-1]) for temp in result if temp.shape[0]>0]
    try:
        return sum(t)/len(t)
    except:
        return 0
    
def s_e_hight_std_means(g_df):
    result = st_end(g_df)
    t = [temp.HEIGHT.std() for temp in result]
    return sum(t)/len(t)
    
def call_out_nums(g_df):
    c_list = list(g_df.CALLSTATE)
    result = Counter(c_list)
    return result[1]/len(c_list)
    
def call_out_times_4s(g_df):
    c_list = list(g_df.CALLSTATE)
    result = Counter(c_list)
    return result[4]/len(c_list)
    
def call_times_alls(g_df):
    c_list = list(g_df.CALLSTATE)
    result = Counter(c_list)
    return (result[1]+result[2]+result[3])/len(c_list)
    
def call_drives(g_df):
    c_list = list(g_df[g_df.SPEED > 0].loc[:,'CALLSTATE'])
    result = Counter(c_list)
    return (result[1]+result[2]+result[3])/len(c_list)
    
def weeken_times(g_df):
    c_list = g_df[g_df.week >=4]
    return c_list.shape[0]/g_df.shape[0]
    
def Ys(g_df):
    if len(set(g_df.Y))==1:
        return g_df.Y[0]
    else:
        print('error data !')
        return g_df.Y.mean()
    
def IDS(g_df):
    return g_df.TERMINALNO[0]
        
        
    
    
    
def concat_data(ID_group):
#    tests = self.ID_group.apply(self.test)
    print(len(ID_group))
    try:
        maxspeeds = ID_group.apply(maxspeed).to_frame(name='speed_max')
        minspeeds = ID_group.apply(minspeed).to_frame(name='speed_min')
        meanspeeds = ID_group.apply(meanspeed).to_frame(name='speed_mean')
        speeds_4s = ID_group.apply(speeds_4).to_frame(name='speed_4')
        speeds_1s = ID_group.apply(speeds_1).to_frame(name='speed_1')
        speeds_stds = ID_group.apply(speeds_std).to_frame(name='speed_std')
        speeds_kurts = ID_group.apply(speeds_kurt).to_frame(name='speed_kurt')
        speeds_skews = ID_group.apply(speeds_skew).to_frame(name='speed_skew')
        star_end_numss = ID_group.apply(star_end_nums).to_frame(name='star_end_num')
        s_e_time_meanss = ID_group.apply(s_e_time_means).to_frame(name='s_e_time_mean')
        s_e_time_maxss = ID_group.apply(s_e_time_maxs).to_frame(name='s_e_time_max')
        s_e_time_minss = ID_group.apply(s_e_time_mins).to_frame(name='s_e_time_min')
        s_e_time_stdss = ID_group.apply(s_e_time_stds).to_frame(name='s_e_time_std')
        direction_stdss = ID_group.apply(direction_stds).to_frame(name='direction_std')
        s_e_direction_std_meanss = ID_group.apply(s_e_direction_std_means).to_frame(name='s_e_direction_std_mean')
        s_e_direction_std_maxss = ID_group.apply(s_e_direction_std_maxs).to_frame(name='s_e_direction_std_max')
        s_e_direction_std_minss = ID_group.apply(s_e_direction_std_mins).to_frame(name='s_e_direction_std_min')
        s_e_direction_std_4ss = ID_group.apply(s_e_direction_std_4s).to_frame(name='s_e_direction_std_4')
        s_e_speed_meanss = ID_group.apply(s_e_speed_means).to_frame(name='s_e_speed_mean')
        s_e_speed_maxss = ID_group.apply(s_e_speed_maxs).to_frame(name='s_e_speed_max')
        s_e_speed_minss = ID_group.apply(s_e_speed_mins).to_frame(name='s_e_speed_min')
        s_e_speed_4ss = ID_group.apply(s_e_speed_4s).to_frame(name='s_e_speed_4')
        s_e_speed_1ss = ID_group.apply(s_e_speed_1s).to_frame(name='s_e_speed_1')
        s_e_speed_stdss = ID_group.apply(s_e_speed_stds).to_frame(name='s_e_speed_std')
        s_e_speed_std_maxss = ID_group.apply(s_e_speed_std_maxs).to_frame(name='s_e_speed_std_max')
        s_e_speed_std_minss = ID_group.apply(s_e_speed_std_mins).to_frame(name='s_e_speed_std_min')
        s_e_speed_std_4ss = ID_group.apply(s_e_speed_std_4s).to_frame(name='s_e_speed_std_4')
        s_e_speed_std_1ss = ID_group.apply(s_e_speed_std_1s).to_frame(name='s_e_speed_std_1')
        s_e_speed_kurtss = ID_group.apply(s_e_speed_kurts).to_frame(name='s_e_speed_kurt')
        s_e_speed_skewss = ID_group.apply(s_e_speed_skews).to_frame(name='s_e_speed_skew')
        s_e_trip_numss = ID_group.apply(s_e_trip_nums).to_frame(name='s_e_trip_num')
        s_e_tripid_meanss = ID_group.apply(s_e_tripid_means).to_frame(name='s_e_tripid_mean')
        s_e_tripid_stdss = ID_group.apply(s_e_tripid_stds).to_frame(name='s_e_tripid_std')
        s_e_miles_maxss = ID_group.apply(s_e_miles_maxs).to_frame(name='s_e_miles_max')
        s_e_miles_minss = ID_group.apply(s_e_miles_mins).to_frame(name='s_e_miles_min')
        s_e_miles_meanss = ID_group.apply(s_e_miles_means).to_frame(name='s_e_miles_mean')
        s_e_hight_std_meanss = ID_group.apply(s_e_hight_std_means).to_frame(name='s_e_hight_std_mean')
        call_out_numss = ID_group.apply(call_out_nums).to_frame(name='call_out_num')
        call_out_times_4ss = ID_group.apply(call_out_times_4s).to_frame(name='call_out_times_4')
        call_times_allss = ID_group.apply(call_times_alls).to_frame(name='call_times_all')
        call_drivess = ID_group.apply(call_drives).to_frame(name='call_drive')
        weeken_timess = ID_group.apply(weeken_times).to_frame(name='weeken_time')
        Yss = ID_group.apply(Ys).to_frame(name='Y')
        IDSS = ID_group.apply(IDS).to_frame(name='Id')
    
    except Exception as err:
        print(err)
        
    vin_data = pd.concat([maxspeeds,minspeeds,meanspeeds,speeds_4s,speeds_1s,speeds_stds,
                          speeds_kurts,speeds_skews,star_end_numss,s_e_time_meanss,
                          s_e_time_maxss,s_e_time_minss,s_e_time_stdss,direction_stdss,
                          s_e_direction_std_meanss,s_e_direction_std_maxss,s_e_direction_std_minss,
                          s_e_direction_std_4ss,s_e_speed_meanss,s_e_speed_maxss,
                          s_e_speed_minss,s_e_speed_4ss,s_e_speed_1ss,s_e_speed_stdss,
                          s_e_speed_std_maxss,s_e_speed_std_minss,s_e_speed_std_4ss,
                          s_e_speed_std_1ss,s_e_speed_kurtss,s_e_speed_skewss,
                          s_e_trip_numss,s_e_tripid_meanss,s_e_tripid_stdss,s_e_miles_maxss,
                          s_e_miles_minss,s_e_miles_meanss,s_e_hight_std_meanss,
                          call_out_numss,call_out_times_4ss,call_times_allss,call_drivess,weeken_timess,Yss,IDSS],axis=1)
    vin_data = vin_data.fillna(method='pad');vin_data = vin_data.fillna(method='bfill')
    vin_data = vin_data.sort_index()
    return vin_data

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


start_all = datetime.datetime.now()
train_path = '/data/dm/train.csv'
#train_path = '/Users/XFBY/Desktop/fight/drive/PINGAN-2018-train_demo.csv'
test_path = '/data/dm/test.csv'
path_result_out =  "model/pro_result.csv"
df_train = pd.read_csv(train_path)
print('data read complite!')
   
df_train['time'] = df_train.TIME.apply(timestamp_datetime)
df_train.index = pd.to_datetime(df_train.time)
df_train = df_train.sort_index()
df_train['week'] = [int(x) for x in df_train.index.weekday]
col = ['TERMINALNO','LONGITUDE','TRIP_ID', 'LATITUDE', 'DIRECTION',
                      'HEIGHT', 'SPEED','CALLSTATE', 'Y','week']
select_df = df_train[col]
ID_group_train = select_df.groupby('TERMINALNO')
print('train data pre dell complete !')


########################
try:  
    data_train = concat_data(ID_group_train)
    print('train data compte over')
except Exception as err:
    print(err)
########################

df_test = pd.read_csv(test_path)
df_test['Y'] = 0

df_test['time'] = df_test.TIME.apply(timestamp_datetime)
df_test.index = pd.to_datetime(df_test.time)
df_test = df_test.sort_index()
df_test['week'] = [int(x) for x in df_test.index.weekday]
select_df_test = df_test[col]
ID_group_test = select_df_test.groupby('TERMINALNO')
try:  
    data_test = concat_data(ID_group_test)
    print('test data compte over')
except Exception as err:
    print(err)

print('feature eng construction complite!')

xr = data_train[f_col];yr = data_train.Y;test_datax = data_test[f_col_test]

train_data, test_data, train_label, test_label = train_test_split(xr,yr,train_size=0.8,test_size=0.1)

clf = xgb.XGBRegressor(max_depth=35, learning_rate=0.05, n_estimators=500, silent=0, objective='reg:gamma')

clf.fit(train_data,train_label)

see = clf.predict(test_data)


print('\n=============>\n'+"RMSE:",np.sqrt(metrics.mean_squared_error(test_label,see)))
print('\n=============>\n'+'the mean sqare error:%.2f' %np.mean(abs(see-test_label)))
Id = list(test_datax['Id']);test_all = test_datax[f_col];Pred = list(clf.predict(test_all))
result_dict = {"Id" : Id,"Pred" : Pred}
result = pd.DataFrame(result_dict)
result.to_csv(path_result_out,header=True,index=False)
print('\n=============>\n'+'give result complite!')
print("Time used:",(datetime.datetime.now()-start_all).seconds)

    