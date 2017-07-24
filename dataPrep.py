# -*- coding: utf-8 -*-
"""
Created on Wed May 15017

@author: Jun Wang
"""
import numpy as np
import pandas as pd
from datetime import datetime, date
from operator import le, eq
import gc
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s", datefmt='%m/%d/%y %H:%M:%S')

path = 'H:/K/Sberbank'

logging.info("loading data...")
train = pd.read_csv(path + '/data/train.csv').set_index('id')
test = pd.read_csv(path + '/data/test.csv').set_index('id')
test['split'] = 0
train['split'] = 1
train_test = pd.concat([train,test])
macro = pd.read_csv(path + '/data/macro.csv')

train_test['timestamp'] = pd.to_datetime(train_test['timestamp'])
train_test['apartment_name'] = train_test.sub_area + train_test['metro_km_avto'].astype(str)
eco_map = {'excellent':4, 'good':3, 'satisfactory':2, 'poor':1, 'no data':0}
train_test['ecology'] = train_test['ecology'].map(eco_map)

cats = train_test.dtypes[train_test.dtypes == 'object'].index
lbl = preprocessing.LabelEncoder()
for var in cats:
    train_test[var] = lbl.fit_transform(train_test[var].values)


logging.info("Dealing with Outlier...")

train_test.loc[train_test.full_sq>2000,'full_sq'] = np.nan
train_test.loc[train_test.full_sq<3,'full_sq'] = np.nan
train_test.loc[train_test.life_sq>500,'life_sq'] = np.nan
train_test.loc[train_test.life_sq<3,'life_sq'] = np.nan
train_test.loc[train_test.life_sq>0.8*train_test.full_sq,'life_sq'] = np.nan
train_test.loc[train_test.kitch_sq>=train_test.life_sq,'kitch_sq'] = np.nan
train_test.loc[train_test.kitch_sq>500,'kitch_sq'] = np.nan
train_test.loc[train_test.kitch_sq<2,'kitch_sq'] = np.nan
train_test.loc[train_test.state>30,'state'] = np.nan
train_test.loc[train_test.build_year<1800,'build_year'] = np.nan
train_test.loc[train_test.build_year==20052009,'build_year'] = 2005
train_test.loc[train_test.build_year==4965,'build_year'] = np.nan
train_test.loc[train_test.build_year>2021,'build_year'] = np.nan
train_test.loc[train_test.num_room>15,'num_room'] = np.nan
train_test.loc[train_test.num_room==0,'num_room'] = np.nan
train_test.loc[train_test.floor==0,'floor'] = np.nan
train_test.loc[train_test.max_floor==0,'max_floor'] = np.nan
train_test.loc[train_test.floor>train_test.max_floor,'max_floor'] = np.nan

#---------------------------------------------------------------
# brings error down a lot by removing extreme price per sqm
#---------------------------------------------------------------

bad_index = train_test[train_test.price_doc/train_test.full_sq > 600000].index
bad_index = bad_index.append(train_test[train_test.price_doc/train_test.full_sq < 10000].index)
len(bad_index)
train_test.drop(bad_index,0,inplace=True)


logging.info('Feature Engineering...')
gc.collect()

train_test['year'] = train_test.timestamp.dt.year  
train_test['weekday'] = train_test.timestamp.dt.weekday

#----------------------------
#    Assign weight
#---------------------------
train_test['w'] = 1
train_test.loc[train_test.price_doc==1000000,'w'] *= 0.5
train_test.loc[train_test.year==2015,'w'] *= 1.5
train_test.w.value_counts()


#Floor
train_test['floor_by_max_floor'] = train_test.floor / train_test.max_floor

#Room
train_test['avg_room_size'] = (train_test.life_sq - train_test.kitch_sq) / train_test.num_room
train_test['life_sq_prop'] = train_test.life_sq / train_test.full_sq
train_test['kitch_sq_prop'] = train_test.kitch_sq / train_test.full_sq

#Calculate age of building
train_test['build_age'] = train_test.year - train_test.build_year
train_test = train_test.drop('build_year', 1)

#Population
train_test['popu_den'] = train_test.raion_popul / train_test.area_m
train_test['gender_rate'] = train_test.male_f / train_test.female_f
train_test['working_rate'] = train_test.work_all / train_test.full_all

#Education
train_test.loc[train_test.preschool_quota==0,'preschool_quota'] = np.nan
train_test['preschool_ratio'] =  train_test.children_preschool / train_test.preschool_quota
train_test['school_ratio'] = train_test.children_school / train_test.school_quota

train_test['square_full_sq'] = (train_test.full_sq - train_test.full_sq.mean()) ** 2
train_test['square_build_age'] = (train_test.build_age - train_test.build_age.mean()) ** 2
train_test['nan_count'] = train_test[['full_sq','build_age','life_sq','floor','max_floor','num_room']].isnull().sum(axis=1)
train_test['full*maxfloor'] = train_test.max_floor * train_test.full_sq
train_test['full*floor'] = train_test.floor * train_test.full_sq

train_test['full/age'] = train_test.full_sq / (train_test.build_age + 0.5)
train_test['age*state'] = train_test.build_age * train_test.state

# new trial
train_test['main_road_diff'] = train_test['big_road2_km'] - train_test['big_road1_km']
train_test['rate_metro_km'] = train_test['metro_km_walk'] / train_test['ID_metro'].map(train_test.metro_km_walk.groupby(train_test.ID_metro).mean().to_dict())
train_test['rate_road1_km'] = train_test['big_road1_km'] / train_test['ID_big_road1'].map(train_test.big_road1_km.groupby(train_test.ID_big_road1).mean().to_dict())
# best on LB with weekday

train_test['rate_road2_km'] = train_test['big_road2_km'] / train_test['ID_big_road2'].map(train_test.big_road2_km.groupby(train_test.ID_big_road2).mean().to_dict())
train_test['rate_railroad_km'] = train_test['railroad_station_walk_km'] / train_test['ID_railroad_station_walk'].map(train_test.railroad_station_walk_km.groupby(train_test.ID_railroad_station_walk).mean().to_dict())
train_test.drop(['year','timestamp'], 1, inplace = True)

#Separate train and test again
train = train_test[train_test.split==1].drop(['split'],1)
test = train_test[train_test.split==0].drop(['split','price_doc', 'w'],1)

train.to_csv(path + '/data/train_featured.csv', index=False)
test.to_csv(path + '/data/test_featured.csv', index=False)





