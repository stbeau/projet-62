# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:39:57 2020

@author: josal
"""
#%%
# to handle datasets
import pandas as pd
import numpy as np

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to build the models
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# to evaluate the models
from sklearn.metrics import mean_squared_error
from math import sqrt

# to persist the model and the scaler
#from sklearn.externals import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)
#%% md ### Load Raw Data

#%%
# Load Raw Data
data_in = pd.read_csv('../../data/interim/raw_useful_ftrs.csv')
data_in = pd.read_csv('../../data/interim/raw_useful_ftrs.csv')
data = data_in.dropna()
print(data.shape)
data.head(1).T

#%%
# load dataset tags on features to use
features_csv= pd.read_csv('../../data/interim/features_to_use_summary.csv', index_col=0)
print(features_csv.shape)

features_csv[['use_ftr']].sort_values(by=['use_ftr'], ascending=False).head(10)
ftrs = list(features_csv.loc[features_csv.use_ftr==True,'use_ftr'].index)
ftrs

#%%
# Filter useful ftrs
leave_out_ftrs = ['nature_culture', 'code_postal', 'code_commune']
data2 = data.loc[:,[c for c in ftrs if c not in leave_out_ftrs]]
data2.info() 

data2 = data_in.dropna()
print(data.shape)
data.tail(1).T

#%%
# load External data 
big_cities = pd.read_csv('../../data/external/france_big_cities.csv', sep = ';')
big_cities.head()


#%%4
## add columns with radians for latitude and longitude
import sklearn

cities_radians = data2.loc[:,['latitude', 'longitude']].apply(np.radians)
big_cities_radians = big_cities.apply({'latitude':np.radians, 'longitude':np.radians})
big_cities_radians.tail()


dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')


# for i,r in data2.head(2).iterrows():
#     ith = pd.DataFrame((cities_radians.loc[i, ['latitude','longitude']].values.reshape(1,-1))
#                        , index = [0], columns =['latitude','longitude'])

#     distances = np.ravel(dist.pairwise(ith, big_cities[['latitude','longitude']]))* 6371

#     closest_dist = distances[np.argmin(distances)]
#     farest_dist =  distances[np.argmax(distances)]
    
#     closest_name = big_cities.city[np.argsort(distances)[0]]
#     farest_name = big_cities.city[np.argsort(distances)[::-1][0]]
#     print(distances)
#     print(closest_name, closest_dist, farest_name, farest_dist )
#     #data2['close_big_city_dist'] = clostes
    
dist_matrix = pd.DataFrame(dist.pairwise
    (cities_radians[['latitude','longitude']],
     big_cities_radians[['latitude','longitude']])*6371 # 6371 kms is average radius of the earth
    ,index=cities_radians.index, columns = big_cities.city
)



data2['close_big_city_dist'] = dist_matrix.apply(np.min, axis = 1)
data2.tail()

#%%

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,8))
data2.close_big_city_dist.plot.hist(ax = ax, bins = 70)
plt.xlim(0,125)
plt.show()

#%%


