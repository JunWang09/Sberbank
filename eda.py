from __future__ import division

import numpy as np 
import pandas as pd 
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import xgboost as xgb
import logging
logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s", level=logging.INFO, datefmt='%m/%d/%y %H:%M:%S')

from sklearn import model_selection, preprocessing
from sklearn.cross_validation import train_test_split, KFold
import xlsxwriter

import datetime
import matplotlib.dates as mdates

color = sns.color_palette()
pd.options.mode.chained_assignment = None

path = '/home/jun/Documents/Kaggle/Sberbank/'

train = pd.read_csv(path + 'data/train.csv', parse_dates=['timestamp']) 
test = pd.read_csv(path + 'data/test.csv', parse_dates=['timestamp'])
train.shape, test.shape

train['price_doc_log'] = np.log(train['price_doc'])

writer = pd.ExcelWriter(path + 'eda/eda03.xlsx')
workbook = writer.book
worksheet = workbook.add_worksheet('Charts')

options = {
    'width': 256,
    'height': 30,
    'x_offset': 10,
    'y_offset': 10,

    'font': {'color': 'white',
             'size': 14},
    'align': {'vertical': 'middle',
              'horizontal': 'center'
              },
    'gradient': {'colors': ['#DDEBCF',
                            '#9CB86E']}
}


logging.info("check missing data")
train_missing = (train.isnull().sum()/train.shape[0])*100
train_missing = train_missing.drop(train_missing[train_missing ==0].index).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8,12))
sns.barplot(x=train_missing, y=train_missing.index)
ax.set(title='Percent missing data', xlabel='% of missing')
fig.savefig("fig1.png")

worksheet.insert_textbox('A1', '1. Missing data in train', options)
worksheet.insert_image('A3',"fig1.png", {'x_scale':0.6, 'y_scale':0.6})


train['state'][train.state == 33] = train.state.mode().iloc[0]
train.state.value_counts()

train['build_year'][train.build_year == 20052009] = 2007

logging.info("Housing internal characteristics")
house_feature = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'kitch_sq', 'state', 'price_doc']     
corrMatrix = train[house_feature].corr()
fig, ax = plt.subplots(figsize=(10,7))
plt.xticks(rotation='45')
sns.heatmap(corrMatrix, square=True, linewidths=0.5, annot=True)
fig.savefig('fig2.png')
worksheet.insert_textbox('H1', '2. Housing internal characteristics', options)
worksheet.insert_image('H3', 'fig2.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info("Area of Home and Number of rooms")
fig, ax = plt.subplots(figsize=(10,7))
plt.scatter(x=train.full_sq, y=train.price_doc)
fig.savefig('fig3.png')
worksheet.insert_textbox('A40', '3. full_sq vs. price_doc', options)
worksheet.insert_image('A43', 'fig3.png', {'x_scale':0.6, 'y_scale':0.6})

fig.ax = plt.subplots(figsize=(10,7))
plt.scatter(x=train.full_sq[train.full_sq<1000], y=train.price_doc[train.full_sq<1000])
fig.savefig('fig4.png')
worksheet.insert_textbox('H40', '4. full_sq vs. price_doc', options)
worksheet.insert_image('H43', 'fig4.png', {'x_scale':0.6, 'y_scale':0.6})

(train.life_sq > train.full_sq).sum() #37

fig, ax = plt.subplots(figsize=(10,7))
sns.countplot(train.num_room)
ax.set(title = 'Distribution of room count', xlabel='num_room')
fig.savefig('fig5.png')
worksheet.insert_textbox('P40', '5. Distribution of room counts', options)
worksheet.insert_image('P43', 'fig5.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info("Product type")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8), sharey=True)
train.price_doc_log[train.product_type == 'Investment'].plot.kde(ax=ax[0])
train.price_doc_log[train.product_type == 'OwnerOccupier'].plot.kde(ax=ax[1])
ax[0].set(xlabel='price_log', title='Investment')
ax[1].set(xlabel='price_log', title='OwnerOccupier')
fig.savefig('fig6.png')
worksheet.insert_textbox('A65', '6. Product type', options)
worksheet.insert_image('A68', 'fig6.png', {'x_scale':0.6, 'y_scale':0.6})

logging.info('Build year')
fig, ax = plt.subplots(1,2,figsize=(16, 8))

fig.tight_layout()
plt.xticks(rotation='45')
by_year = train[(train.build_year>1800) & (train.build_year<2018)]
sns.countplot(by_year.build_year, ax=ax[0])
ax[0].set(xlabel='Build Year', title='Distribution of Build Year')

grouped = by_year.groupby(['build_year'], as_index=False)[['price_doc']].mean()
sns.regplot(x='build_year', y='price_doc', data=grouped, scatter=False, order=3, truncate=True,ax=ax[1])
plt.plot(grouped.build_year, grouped.price_doc, color='r')
ax[1].set(title='Average price by build year')

fig.savefig('fig7.png')
worksheet.insert_textbox('H65', '7. Distribution of Build Year', options)
worksheet.insert_image('H68', 'fig7.png', {'x_scale':0.6, 'y_scale':0.6}) 


logging.info('Timestamp')
fig, ax = plt.subplots(1,2,figsize=(20,8))
fig.tight_layout()
grouped = train.groupby(['timestamp'])[['price_doc']].median()
ax[0].plot(grouped.index, grouped.price_doc, color='r')
ax[0].set(title='Median price by timestamp')

years = mdates.YearLocator()
yearsFmt = mdates.DateFormatter('%Y')
sns.countplot(train.timestamp, ax=ax[1])
ax[1].xaxis.set_major_locator(years)
ax[1].xaxis.set_major_formatter(yearsFmt)
ax[1].set(title = 'Transaction volumn', ylabel='Number of transactions')
    
fig.savefig('fig8.png')
worksheet.insert_textbox('A95', '8. Timestamp', options)
worksheet.insert_image('A98', 'fig8.png', {'x_scale':0.6, 'y_scale':0.6}) 

logging.info('Month')
fig, ax = plt.subplots(figsize=(12,8))
month = train.groupby([train.timestamp.dt.month])[['price_doc']].median()
plt.plot(month.index, month.price_doc, color ='r')
ax.set(title='Price by month')
fig.savefig('fig9.png')
worksheet.insert_textbox('P95', '9. Price by Month of year', options)
worksheet.insert_image('P98', 'fig9.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('state')
fig, ax = plt.subplots(figsize=(12,8))
sns.violinplot(x='state', y='price_doc_log', data=train[train.state.notnull()], inner='box')
ax.set(title = 'Price by state of home', xlabel='state', ylabel='Log price')
fig.savefig('fig10.png')
worksheet.insert_textbox('A120', '10. Price by state', options)
worksheet.insert_image('A123', 'fig10.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('material')
fig, ax = plt.subplots(figsize=(12,9))
sns.violinplot(x='material', y='price_doc_log', data=train[train.material.notnull()], inner='box')
ax.set(title='Price by state of material', xlabel='material', ylabel='Log price')
fig.savefig('fig11.png')
worksheet.insert_textbox('H120', '11. Price by material', options)
worksheet.insert_image('H123', 'fig11.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('floor')
fig, ax = plt.subplots(1,3,figsize=(30,9))
fig.tight_layout()
ax[0].scatter(x=train.floor, y=train.price_doc_log, c='r', alpha=0.4)
sns.regplot(x='floor', y='price_doc_log', data=train, scatter=False, truncate=True, ax=ax[0])
ax[0].set(title='Price by floor', xlabel='Floor', ylabel='Log of price')

ax[1].scatter(x=train.max_floor, y=train.price_doc_log, c='r', alpha=0.4)
sns.regplot(x='max_floor', y='price_doc_log', data=train, scatter=False, truncate=True, ax=ax[1])
ax[1].set(title='Price by max floor', xlabel='Max Floor', ylabel='Log of price')

ax[2].scatter(x=train.floor, y=train.max_floor, c='r', alpha=0.4)
ax[2].plot([0,80],[0,80],color='.5')

fig.savefig('fig12.png')
worksheet.insert_textbox('A140', '12. Price by floor', options)
worksheet.insert_image('A143', 'fig12.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('demographic')
demo_vars = ['area_m', 'raion_popul', 'full_all', 'male_f', 'female_f', 'young_all', 'young_female', 
             'work_all', 'work_male', 'work_female', 'price_doc']
corrMat = train[demo_vars].corr()

fig, ax = plt.subplots(figsize=(12,8))
plt.xticks(rotation='45')
sns.heatmap(corrMat, square=True,linewidths=0.5, annot=True)
fig.savefig('fig13.png')# linear algebra
worksheet.insert_textbox('H140', '13. Demographic', options)
worksheet.insert_image('H143', 'fig13.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('Population density')
train['area_km'] = train.area_m/1000000
train['density'] = train.raion_popul/train.area_km
grouped = train.groupby(['sub_area'])[['density', 'price_doc']].median()
fig, ax = plt.subplots(figsize=(12,8))
sns.regplot(x='density', y='price_doc', data=grouped, scatter=True, order = 1, truncate=True)     
fig.savefig('fig14.png')
worksheet.insert_textbox('P140', '14. Density', options)
worksheet.insert_image('P143', 'fig14.png', {'x_scale':0.6, 'y_scale':0.6})
     

logging.info('share of working age population')
train['work_share'] = train.work_all/train.raion_popul
grouped = train.groupby(['sub_area'])[['work_share', 'price_doc']].median()
fig, ax = plt.subplots(figsize=(12,8))
sns.regplot(x='work_share', y='price_doc', data=grouped, scatter=True, order=4, truncate=True)
fig.savefig('fig15.png')
worksheet.insert_textbox('A165', '15. Share of working age population', options)
worksheet.insert_image('A168', 'fig15.png', {'x_scale':0.6, 'y_scale':0.6})     


logging.info('School')
school_chars = ['children_preschool', 'preschool_quota', 'preschool_education_centers_raion', 'children_school', 
                'school_quota', 'school_education_centers_raion', 'school_education_centers_top_20_raion', 
                'university_top_20_raion', 'additional_education_raion', 'additional_education_km', 'university_km', 'price_doc']
corrMat = train[school_chars].corr()
fig, ax = plt.subplots(figsize=(12,8))
plt.xticks(rotation='45')
sns.heatmap(corrMat, square=True, linewidths=0.5, annot=True)

fig.savefig('fig16.png')
worksheet.insert_textbox('H165', '16. School correlation', options)
worksheet.insert_image('H168', 'fig16.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('University top 20')
fig, ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='university_top_20_raion', y="price_doc", data=train, jitter=True, alpha=.2, color=".90")
sns.boxplot(x='university_top_20_raion', y="price_doc", data=train)
ax.set(title='Distribution of home price by # of top universities in Raion', xlabel='university_top_20_raion', ylabel='price_doc')

fig.savefig('fig17.png')
worksheet.insert_textbox('P165', '17. University top 20', options)
worksheet.insert_image('P168', 'fig17.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('Recreational Characteristics')
cult_chars = ['sport_objects_raion', 'culture_objects_top_25_raion', 'shopping_centers_raion', 'park_km', 'fitness_km', 
                'swim_pool_km', 'ice_rink_km','stadium_km', 'basketball_km', 'shopping_centers_km', 'big_church_km',
                'church_synagogue_km', 'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km', 'price_doc']
corrmat = train[cult_chars].corr()

f, ax = plt.subplots(figsize=(12, 7))
plt.xticks(rotation='45')
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)

f.savefig('fig18.png')
worksheet.insert_textbox('A190', '18. University top 20', options)
worksheet.insert_image('A193', 'fig18.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('# of sports objects in Raion')
f, ax = plt.subplots(figsize=(12, 7))
so_price = train.groupby('sub_area')[['sport_objects_raion', 'price_doc']].median()
sns.regplot(x="sport_objects_raion", y="price_doc", data=so_price, scatter=True, truncate=True)
ax.set(title='Median Raion home price by # of sports objects in Raion')

f.savefig('fig19.png')
worksheet.insert_textbox('H190', '19. # of sports objects in Raion', options)
worksheet.insert_image('H193', 'fig19.png', {'x_scale':0.6, 'y_scale':0.6})

co_price = train.groupby('sub_area')[['culture_objects_top_25_raion', 'price_doc']].median()
f, ax = plt.subplots(figsize=(12, 7))
sns.regplot(x="culture_objects_top_25_raion", y="price_doc", data=co_price, scatter=True, truncate=True)
ax.set(title='Median Raion home price by # of sports objects in Raion')
f.savefig('fig20.png')
worksheet.insert_textbox('P190', '20. # of sports objects in Raion', options)
worksheet.insert_image('P193', 'fig20.png', {'x_scale':0.6, 'y_scale':0.6})

f, ax = plt.subplots(figsize=(12,7))
sns.regplot(x='park_km', y='price_doc', data=train, scatter=True, truncate=True, scatter_kws={'color':'r', 'alpha':0.2})
ax.set(title='Median Raion home price by # of sports objects in Raion')
f.savefig('fig21.png')
worksheet.insert_textbox('A215', '21. # of sports objects in Raion', options)
worksheet.insert_image('A128', 'fig21.png', {'x_scale':0.6, 'y_scale':0.6})


logging.info('Infrastructure')
inf_features = ['nuclear_reactor_km', 'thermal_power_plant_km', 'power_transmission_line_km', 'incineration_km',
                'water_treatment_km', 'incineration_km', 'railroad_station_walk_km', 'railroad_station_walk_min', 
                'railroad_station_avto_km', 'railroad_station_avto_min', 'public_transport_station_km', 
                'public_transport_station_min_walk', 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km','bulvar_ring_km',
                'kremlin_km', 'price_doc']
corrmat = train[inf_features].corr()
f, ax = plt.subplots(figsize=(12, 7))
plt.xticks(rotation='45')
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
f.savefig('fig22.png')
worksheet.insert_textbox('H215', '22. # of sports objects in Raion', options)
worksheet.insert_image('H218', 'fig22.png', {'x_scale':0.6, 'y_scale':0.6})

f, ax = plt.subplots(figsize=(12,7))
sns.regplot(x='kremlin_km', y='price_doc', data=train, scatter=True, truncate=True, scatter_kws={'color':'r', 'alpha':0.2})
ax.set(title='Home price by distance to Kremlin')
f.savefig('fig23.png')
worksheet.insert_textbox('P215', '23. Home price by distance to Kremlin', options)
worksheet.insert_image('P218', 'fig23.png', {'x_scale':0.6, 'y_scale':0.6})




logging.info('Train vs Test')

# full_sq

f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
f.tight_layout()
np.log(train.full_sq+1).plot.kde(ax=ax[0])
ax[0].set(title='Train', xlabel='full_sq')
np.log(test.full_sq+1).plot.kde(ax=ax[1])
ax[1].set(title='Test', xlabel='full_sq')

f.savefig('fig24.png')
worksheet.insert_textbox('A240', '24. full_sq: train vs test', options)
worksheet.insert_image('A243', 'fig24.png', {'x_scale':0.6, 'y_scale':0.6})


# life_sq

f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
f.tight_layout()
np.log(train.life_sq+1).plot.kde(ax=ax[0])
ax[0].set(title='Train', xlabel='life_sq')
np.log(test.life_sq+1).plot.kde(ax=ax[1])
ax[1].set(title='Test', xlabel='life_sq')

f.savefig('fig25.png')
worksheet.insert_textbox('P240', '25. life_sq: train vs test', options)
worksheet.insert_image('P243', 'fig25.png', {'x_scale':0.6, 'y_scale':0.6})


# kitch_sq

f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
f.tight_layout()
np.log(train.kitch_sq+1).plot.kde(ax=ax[0])
ax[0].set(title='Train', xlabel='kitch_sq')
np.log(test.kitch_sq+1).plot.kde(ax=ax[1])
ax[1].set(title='Test', xlabel='kitch_sq')

f.savefig('fig26.png')
worksheet.insert_textbox('A265', '26. kitch_sq: train vs test', options)
worksheet.insert_image('A268', 'fig26.png', {'x_scale':0.6, 'y_scale':0.6})

#num room
f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
f.tight_layout()
sns.countplot(x=train.num_room, ax=ax[0])
ax[0].set(title='Train', xlabel='num_room')
sns.countplot(x=test.num_room, ax=ax[1])
ax[1].set(title='Test', xlabel='num_room')

f.savefig('fig27.png')
worksheet.insert_textbox('P265', '27. num_room: train vs test', options)
worksheet.insert_image('P268', 'fig27.png', {'x_scale':0.6, 'y_scale':0.6})

#floor

f,ax = plt.subplots(1,2,figsize=(12,8),sharey=True)
f.tight_layout()
train.floor.plot.kde(ax=ax[0])
ax[0].set(title='Train', xlabel='floor')
test.floor.plot.kde(ax=ax[1])
ax[1].set(title='Test', xlabel='floor')

f.savefig('fig28.png')
worksheet.insert_textbox('A290', '28. floor: train vs test', options)
worksheet.insert_image('A290', 'fig28.png', {'x_scale':0.6, 'y_scale':0.6})


#max floor

f,ax = plt.subplots(1,2,figsize=(12,8),sharey=True)
f.tight_layout()
train.max_floor.plot.kde(ax=ax[0])
ax[0].set(title='Train', xlabel='max_floor')
test.max_floor.plot.kde(ax=ax[1])
ax[1].set(title='Test', xlabel='max_floor')

f.savefig('fig29.png')
worksheet.insert_textbox('P290', '29. max_floor: train vs test', options)
worksheet.insert_image('P290', 'fig29.png', {'x_scale':0.6, 'y_scale':0.6})


f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
ax[0].scatter(x=train.floor, y=train.max_floor, c='r',alpha=0.4)
ax[0].plot([0,80], [0,80], color='0.5')
ax[0].set(title='Train', xlabel='floor', ylabel='max_floor')
ax[1].scatter(x=test.floor, y=test.max_floor, c='r',alpha=0.4)
ax[1].plot([0,80], [0,80], color='0.5')
ax[1].set(title='Test', xlabel='floor', ylabel='max_floor')

f.savefig('fig30.png')
worksheet.insert_textbox('A315', '30. floor/max_floor: train vs test', options)
worksheet.insert_image('A318', 'fig30.png', {'x_scale':0.6, 'y_scale':0.6})


# transactions by day
years = mdates.YearLocator()
yearsFmt = mdates.DateFormatter('%Y')
ts_train = train.timestamp.value_counts()
ts_test = test.timestamp.value_counts()
f, ax = plt.subplots(figsize=(12,6))
plt.bar(left=ts_train.index, height=ts_train)
plt.bar(left=ts_test.index, height=ts_test)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.set(title='Number of transactions', ylabel='count')

f.savefig('fig31.png')
worksheet.insert_textbox('P315', '31. Number of transactions', options)
worksheet.insert_image('P318', 'fig31.png', {'x_scale':0.6, 'y_scale':0.6})



#product type
f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
f.tight_layout()
sns.countplot(x=train.product_type, ax=ax[0])
ax[0].set(title='Train', xlabel='product type')
sns.countplot(x=test.product_type, ax=ax[1])
ax[1].set(title='Test', xlabel='product type')

f.savefig('fig32.png')
worksheet.insert_textbox('A340', '32. product type: train vs test', options)
worksheet.insert_image('A343', 'fig32.png', {'x_scale':0.6, 'y_scale':0.6})


#state
f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
f.tight_layout()
sns.countplot(x=train.state, ax=ax[0])
ax[0].set(title='Train', xlabel='state')
sns.countplot(x=test.state, ax=ax[1])
ax[1].set(title='Test', xlabel='state')

f.savefig('fig33.png')
worksheet.insert_textbox('P340', '33. state: train vs test', options)
worksheet.insert_image('P343', 'fig33.png', {'x_scale':0.6, 'y_scale':0.6})



#state
f, ax = plt.subplots(1,2,figsize=(12,8), sharey=True)
f.tight_layout()
sns.countplot(x=train.material, ax=ax[0])
ax[0].set(title='Train', xlabel='material')
sns.countplot(x=test.material, ax=ax[1])
ax[1].set(title='Test', xlabel='material')

f.savefig('fig34.png')
worksheet.insert_textbox('A365', '34. material: train vs test', options)
worksheet.insert_image('A368', 'fig34.png', {'x_scale':0.6, 'y_scale':0.6})




writer.save()






