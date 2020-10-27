# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Machine Learning Model Pipeline: Wrapping up for Deployment
#
#
# Here, we will summarise, the key pieces of code, that we need to take forward, for this particular project, to put our model in production.
#
#

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
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)
# -

# ### Setting the seed
#
#
# Important note **Always set the seeds**.
#
# Load Libraries

#%%
# Set constant variables

SEED = 123

#%%
# ## Load data
#
# We need the training data to train our model in the production environment. 

# load dataset
ROOT = "../"
data_in = pd.read_csv(ROOT +'/data/interim/raw_useful_ftrs.csv')
data = data_in.dropna()
print(data.shape)
data.head()

data.info()

# Recode code postal as object (This should be done on EDA Preproc Script)

data.loc[:,'code_postal'] = data.loc[:,'code_postal'].astype('str')
data.info()

#%%
# load dataset tags on features to use
features_csv= pd.read_csv(ROOT + '/data/interim/features_to_use_summary.csv', index_col=0)
print(features_csv.shape)

features_csv[['use_ftr']].sort_values(by=['use_ftr'], ascending=False).head(10)
ftrs = list(features_csv.loc[features_csv.use_ftr==True,'use_ftr'].index)
ftrs

#%%
leave_out_ftrs = ['nature_culture', 'code_postal', 'code_commune']
data2 = data.loc[:,[c for c in ftrs if c not in leave_out_ftrs]]
data2.info() 


#%%

# load External data 
big_cities = pd.read_csv(ROOT +'/data/external/france_big_cities.csv', sep = ';')
big_cities.head()

#%%
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



#data2['close_big_city_dist'] = dist_matrix.apply(np.min, axis = 1)
#data2.tail()
## -
#
## %matplotlib inline
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(figsize=(12,8))
#data2.close_big_city_dist.plot.hist(ax = ax, bins = 70)
#plt.xlim(0,125)
#plt.show()

#%%
# Load libraries and set up default values
from sklearn import (model_selection, preprocessing, linear_model, naive_bayes
                     , metrics)

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn import decomposition, ensemble


#import xgboost # A lot more optimized than sci-kit learn one
import string
import numpy as np
import pandas as pd
import scipy
import os
import io
#import aux_functions_mod  #Custom own functions
# -

# #### Num Ftrs

#%%
# Set Ftr Types
num_ftrs = list(data2.select_dtypes('number').columns)
num_ftrs.remove('valeur_fonciere')
num_ftrs 


# -

# #### Cat Ftrs

#%%
# let's capture the categorical variables first
cat_ftrs = [var for var in data2.columns if data2[var].dtype == 'O']
cat_ftrs

# Label encoder needs cat ftrs to be string.
for f in cat_ftrs:
    data2[f] = data2[f].astype(str)
# -

# ### Separate dataset into train and test
#
# Before beginning to engineer our features, it is important to separate our data intro training and testing set. This is to avoid over-fitting. There is an element of randomness in dividing the dataset, so remember to set the seed.

data2.loc[:,data2.columns[:-2]].columns
data2.columns

# Let's separate into train and test set
# Remember to seet the seed (random_state for this sklearn function)
# 
X_train, X_test, y_train, y_test = train_test_split(data2.loc[:,data2.columns[:-1]], data2.valeur_fonciere,
                                                    test_size=0.1,
                                                    random_state=SEED) # we are setting the seed here
X_train.shape, X_test.shape
X_train.info()


from catboost import CatBoostRegressor

# !pip install lightgbm

#%%
# =======================Training Pipeline=================================
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
from tempfile import mkdtemp
from joblib import Memory
from sklearn.feature_selection import chi2, SelectPercentile #SelectFromModel
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor



FIX_RAND_STATE = SEED

# Set temp storage to cache first pipe transformations
#cachedir = mkdtemp()
#memory = Memory(location=cachedir, verbose=0)


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_ftrs),
        ('cat', categorical_transformer, cat_ftrs)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
cached_pipe = Pipeline(steps=[('preprocessor', preprocessor)
                      ,#('classifier', LogisticRegression())
                             ])



## Delete the temporary cache
#before exiting
#from shutil import rmtree
#rmtree(cachedir)


params_grid = {

               #, 'ftr_select_percent__percentile': [5,10,20,50] #, 10, 20, 40Retain nth perc
               #, 'feature_select__threshold': [.05, .1] # Drops features with weights under threshold
             }


# Set models and hyperparameters
models = [
          #  ('nb', naive_bayes.MultinomialNB())
           ('ridge', linear_model.Ridge( random_state=FIX_RAND_STATE))
          , ('lasso', linear_model.Ridge( random_state=FIX_RAND_STATE))           
          #, ('knn',  KNeighborsRegressor())                                     
          #,('tree', DecisionTreeRegressor( random_state=FIX_RAND_STATE))
          # ,('rf', RandomForestRegressor( random_state=FIX_RAND_STATE )) #random_state=FIX_RAND_STATE
          #, ('sk_xgb', ensemble.GradientBoostingClassifier(random_state=FIX_RAND_STATE))
          #, ('xgb', xgboost.XGBClassifier(random_state=FIX_RAND_STATE))
           #('catgb'. CatBoostRegressor())
          , ('lgbmr', LGBMRegressor())
          ]

modelsDic = dict(models)


model_params = {
              #   'knn__n_neighbors': [5]
               'ridge__alpha': [1] #.001,.01, .1, 1, 10, 100, 1000, 10000, 100000
              , 'lasso__alpha': [1] #.001,.01, .1, 1, 10, 100, 1000, 10000, 100000
              #, 'tree__max_features':['sqrt']
              #,  'lasso__penalty':['l2'] #'l1',
              #, 'svm__C': [1] #.001,.01, .1, 1, 10, 100, 1000, 10000, 100000
              #, 'svm__kernel':  ['linear', 'rbf']
              # ,'rf__max_features':['sqrt'] #, 'log2', 'auto'
              # , 'catgb__n?estimators' : [100]
              #, 'rf__n_estimators': [1000]
              #, 'sk_xgb__max_depth':[2,3]
              #, 'sk_xgb__subsample': [1.0]
              #, 'sk_xgb__max_features':['auto', 'sqrt', 'log2']
              #, 'sk_xgb__n_estimators':[500]
              #, 'xgb__learning_rate':[0.01]
              #, 'xgb__max_depth':[3] #2,3,4
              #, 'xgb__colsample_bytree':[.7] #
              #, 'xgb__subsample': [.8]
              #, 'xgb__objective': ['binary:logistic']
              #, 'xgb__gamma':[1]
              #, 'xgb__n_estimators': [1000]
               ,'lgbmr__n_estimators':[100] #, 'log2', 'auto'    
        }


preproc_hparams_count = np.prod([len(params_grid[k]) for k in params_grid.keys()])
model_params_count = np.prod([len(model_params[k]) for k in model_params.keys()])
combinations = preproc_hparams_count * model_params_count
print('Total grid search space size: %.d combinations to test'% ( combinations) )

scores = {}
ftr_compress = False
folds = 3


 
import time #Measure gridSearchCV execution time
start = time.time()
print(modelsDic.keys())
for model_name in list(modelsDic.keys()):
    print('Training model %s ...' % model_name)
    pipe = cached_pipe
    pipe.steps.append((model_name, modelsDic[model_name]))
    
    #Add only model_name parameters to the grid dict
    filtered_params = params_grid.copy() #Initialize with preprocess variations to test
    for key in model_params.keys():
        if model_name in key:
            filtered_params[key]=model_params[key]
            
    cv_splitter = model_selection.KFold(n_splits=folds
                                  , random_state=FIX_RAND_STATE)
    
    scorer = metrics.make_scorer(metrics.mean_squared_error)#, average = score_avg
    performance_metric_name = scorer.__str__().rstrip(')').split('(')[1] # Extract scorer metric name

    
    gs = model_selection.GridSearchCV(estimator=pipe, param_grid = filtered_params
                                  , cv = cv_splitter, n_jobs=-1,  scoring = scorer )
    print(filtered_params)
    
    gs.fit(X_train, np.log1p(y_train))
    scores[model_name] = None
    scores[model_name]={'best_score':  gs.best_score_}
    print("Best {}: {:.4f} with params: {}: ".format( performance_metric_name
                                          , gs.best_score_, gs.best_params_))
    
    pipe.steps.pop()   #Pop model in turn from pipe

    # Store best model results to plot i.e  those corresponding to best model params
    cv_results_df = pd.DataFrame(gs.cv_results_)
    # Get the best candidate parameter setting.
    scores[model_name]['cv_optim_result'] = cv_results_df.loc[gs.best_index_,:]
    scores[model_name]['best_estimator'] = gs.best_estimator_
        
end = time.time()
total_elapsed_seconds = end - start
print('Grid search elapsed time: %.2f minutes'% (total_elapsed_seconds/60))



#%%
first_model = list(scores.keys())[0] # Just to get columns of results summary

results_pd = pd.DataFrame(None, columns = ['model'] + list(scores[first_model]['cv_optim_result'].index))
results_pd
# -

models_performance = [(k,scores[k]['best_score']) for k in scores.keys()]
best_model = sorted(models_performance, key=lambda x: x[1], reverse=False)[0][0]
best_model

scores[best_model]['best_estimator'][-1]



#%%
y_pred = np.expm1(scores[best_model]['best_estimator'].predict(X_test))
print(list(zip(y_test,y_pred))[0:10])



#%%
# Base estimator is the overall mean
m = y_test.median()
y_pred =[m for m in range(len(y_test))]

performance = np.sqrt(mean_squared_error(y_test, y_pred))
print('Base model performance: {:.2f} '.format(performance))

#%%
y_pred = np.expm1(scores[best_model]['best_estimator'].predict(X_test))

performance = np.sqrt(mean_squared_error(y_test, y_pred))
print('Base model performance: {:.2f}'.format(performance))

#%%

summary_ftrs_subset = ['mean_fit_time', 'std_fit_time', 'mean_test_score', 'std_test_score']

models_summary_pd = pd.DataFrame(None, columns = [f for f in scores[best_model]['cv_optim_result'].index if f in summary_ftrs_subset]
                       , index = scores.keys())



for model in scores.keys():
    print(scores[model]['cv_optim_result'][summary_ftrs_subset].values)
    models_summary_pd.loc[model] =  scores[model]['cv_optim_result'][summary_ftrs_subset]
    
models_summary_pd

#%% md
#### Drawand Store Metrics

#%%
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
mean_test_score = models_summary_pd
std_test_score =  scores[best_model]['cv_optim_result'].std_test_score*3

mdls = np.array([m for m in models_summary_pd.index])
performances = np.array([p for p in models_summary_pd.mean_test_score])
stds = np.array([p for p in models_summary_pd.std_test_score])

fig, ax = plt.subplots(figsize=(12,8))
#ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
models_summary_pd.T.loc['mean_test_score',:].plot(kind='bar'
                       , yerr=models_summary_pd.T.loc['std_test_score',:].values *3
                       , alpha = 0.3, error_kw=dict(ecolor='k'))

#ax = plt.errorbar(range(len(mdls)), np.expm1(performances), xerr=0., yerr=stds, color='blue', fmt='o')
axes = plt.gca()
llim = np.min(performances-np.min((stds*4)))
ulim = np.max(performances+np.max((stds*4)))
axes.set_ylim([llim,ulim])
plt.title("Compare Models Performance")


#%%
# Write out metrics to be tracked
best_model_cv_summary = scores[best_model]['cv_optim_result'].drop(['params'])
best_model_cv_summary.to_json('metrics/best_model_cv_summary.json', orient='columns')
plt.savefig('../reports/figures/mdl_performance.png')

#%%
#%%
from joblib import dump, load

dump(scores[best_model][ 'best_estimator'], ROOT + '/models/model.joblib') 
clf3 = load(ROOT + 'models/model.joblib') 
new_vals = X_test[0:2]
new_preds = np.expm1(clf3.predict(new_vals))
print(new_preds)
y_test[0:2]


# ### Export first test data as example for frontend app

#%%

X_test.tail(1).to_csv(ROOT + '/data/ui/input_data_sample.csv', index=False)

X_test.tail(1)
