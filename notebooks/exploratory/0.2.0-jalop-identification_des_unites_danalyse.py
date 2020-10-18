# -*- coding: utf-8 -*-

#%% md  #
# Exploratory Data Analysis

# Prix de Vente des Proprietées


#%%
import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import quickda

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

import seaborn as sns
import quickda

#import warnings
#warnings.filterwarnings('ignore')

#%%
# Set Fix Parame
    
SEED = 1234 # Seed for random  number geneartors

#%% md 
### Load  data


#%%


#ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
data_folder = '../../data/raw/'


filenames = [  
               #'dvf_2018.gz',
              'dvf_2019.gz'
             ]

#%% 
 # Util Fn to read write out a data sample from origina raw data files
 def pd_read_save(filename, sample_size, output_path):
 #    n = sum(1 for line in open(filename, errors = "ignore")) - 1 #number of records in file (excludes header)
 #    print("total rows")
 #    s = sample_size #desired sample sizfre
 #    skip = sorted(np.random.random_integers(1,n+1,n-s)) #the 0-indexed header will not be included in the skip list
     df = pd.read_csv(filename, error_bad_lines = False) #, skiprows=skip
     df_sample =  df.sample(sample_size, random_state = SEED)
     df_sample.to_csv(output_path, sep = ";", index = False)
     print('Writing file of shape: ', df_sample.shape)
     return df


 for f in filenames:
     data = data_folder + f
     print(data_folder  +'samples/'+f.split('.')[0]  +'_sample.csv.gz')
     pd_read_save( data, sample_size=400000, output_path = data_folder  +'samples/'+f.split('.')[0]  +'_sample.csv.gz')


#%%
# Read saved data samples
    
dsets = {} #f.split('.')[0]: pd.DataFrame() for f in filenames

for f in filenames:
    base_name = f.split('.')[0]
    dsets[base_name] = pd.read_csv(  data_folder  + 'samples/' + base_name + '_sample.csv.gz'
                                   , error_bad_lines = False, sep = ';'
                                   )
    print(base_name, dsets[base_name].shape)



#%% md 
#### Merge the Data Sets 


#%%
# Merge the data sets

data_in = pd.concat([dsets[d] for d in dsets.keys()])
print(data_in.shape) 
print(data_in.tail(2).T)

# Initial pre Analysis showed that  there are many duplicated rows.
data_in.drop_duplicates(inplace=True)
print(data_in.shape) 

# =============================================================================
#%% md 
#### Explore Data 

#%%

data_in.info()

#%%
# Get Proportion of Null values on Columns

#nulls_df = pd.DataFrame( {'Nulls': data_in.isnull().sum().values
#                        , 'Ptj':data_in.isnull().sum() /  data_in.shape[0]
#                        }
#                       )
#print(nulls_df)


# =============================================================================
#%% md 
### Data Summary
#Summary of the Row Data Set 

#%%
from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *


summary = explore(data_in, method="summarize")
#%%

summary  # Display Raw Data Summary

#summary

#%% md 
### Assign Features on Categories


#%%
#%md Useful FtrsDrop Reason: They can not have missing values
#valeur_fonciere           (target cant have nulls)


#%md High null pct ftrs

#adresse_suffixe                 0.958
#ancien_code_commune             0.991
#ancien_id_parcelle              0.999
#ancien_nom_commune              0.991
#code_nature_culture_speciale    0.954
#code_type_local                 0.476
#lot1_numero                     0.687
#lot1_surface_carrez             0.912
#lot2_numero                     0.935
#lot2_surface_carrez             0.979
#lot3_numero                     0.989
#lot3_surface_carrez             0.998
#lot4_numero                     0.996
#lot4_surface_carrez             0.999
#lot5_numero                     0.998
#lot5_surface_carrez             1.000
#nature_culture_speciale         0.954
#numero_volume                   0.997


#%md Useful FtrsRedundant Features
#                               dtypes   count  null_sum  null_pct  nunique
#ancien_code_commune           float64    7004    778519     0.991      532   
#code_commune                   object  785523         0     0.000    31251   
#code_departement               object  785523         0     0.000       97   
#code_nature_culture            object  535959    249564     0.318       27   
#code_nature_culture_speciale   object   35761    749762     0.954      113   
#code_type_local               float64  411728    373795     0.476        4  


#%md Useful Ftrs
#surface_reelle_bati           118074  0.590370  NaN 0
#nombre_pieces_principales      93279  0.466395  NaN 0
#nature_culture                 62293  0.311465  NaN "N/A"
#code_type_local NAN  "N/A"
#longitude                       4048  0.020240  k-neighb estim
#latitude                        4048  0.020240  k-neighb estim




#%% md 
### Setting Unuseful featues for analysis   Using above summary results



#%% # Assigning Features to differet categorie types

unuseful_ftrs = ['id_mutation', 'adresse_nom_voie', 'adresse_numero']

useful_high_ptc_ftrs = [  'adresse_numero'
                        , 'nombre_pieces_principales'
                        , 'type_local'
                        , 'surface_reelle_bati' 
                        , 'type_local'
                        ]

#Un inputable high nulls ftrs
ftrs_high_null_ptc = [f for f in summary.loc[summary.null_pct > 0.40,:].index  if f not in useful_high_ptc_ftrs]
#summary.loc[ftrs_high_null_ptc, 'null_pct']

useful_code_ftrs =  [
                     'code_postal'
                    ]

redundant_ftrs = [c for c in summary.index if 'code' in c and c not in useful_code_ftrs]
redundant_ftrs


unuseful_ftrs += sorted(list(set(ftrs_high_null_ptc).union(set(redundant_ftrs))))
unuseful_ftrs


ftrs = list(sorted([f for f in data_in.columns if f not in unuseful_ftrs])) 



#%%
# Inspect left useful ftrs
smmry4 = ['dtypes', 'count', 'null_pct', 'nunique']
summary.loc[ftrs, smmry4]





#%% md 
### Set useful Feature Groups


#%%
# Set useful Feature Groups

ftrs = list(sorted([f for f in data_in.columns if f not in unuseful_ftrs]))
target = 'valeur_fonciere'
ftrs.remove(target)

num_ftrs = list(summary.loc[summary.index.isin(ftrs) & (summary['dtypes'] != 'object'),:].index)
num_ftrs = [f for f in num_ftrs if f not in ['code_postal']] # Remove wrogly assigned num ftrs and target
data_in.loc[:,'code_postal'] = data_in.loc[:,'code_postal'].astype('str')

summary.loc[num_ftrs,smmry4]

date_ftrs = ['date_mutation']

cat_ftrs = list(summary.loc[summary.index.isin(ftrs)
                 & (summary['dtypes'] == 'object')
                 & (summary['nunique'] > 2 )
                 ,:].index
                )
cat_ftrs.append('code_postal')
bool_ftrs = []

# Assert all columns have bee assigned to a ftr class
all_solumns_assigned = len(set(data_in.columns).symmetric_difference(
        set(unuseful_ftrs + num_ftrs + cat_ftrs + date_ftrs + bool_ftrs + [target])))==0
assert(all_solumns_assigned)


data_in2 = data_in.loc[:, [target] + ftrs ]


for dt in [date_ftrs, num_ftrs, cat_ftrs]:
    print(summary.loc[dt, smmry4])

data_in2.tail(1).T


#%% md 
##### Check Duplicates



#%% md #
#It seems like duplicates are sales with id_parcelle
#and valeur_fonciere when they are same more than once 


#%%
# Explore number of duplicated id parcells, This, inf fact should be a dropped field
by_id_parcells = data_in.groupby('id_parcelle')
#by_id_parcells.agg({'id_mutation': 'count'}).id_mutation.value_counts()#.plot.bar() #head()
by_parcell_mutations = by_id_parcells.agg({'id_mutation': 'count'}).reset_index()
# print(by_parcell_mutations.head())

nuni_ue_mutations_parcels = by_parcell_mutations.loc[
                                    by_parcell_mutations.id_mutation>1,'id_parcelle'] #filter parcels wit 2 or more mutations

nuni_ue = (data_in.loc[data_in.id_parcelle.isin(nuni_ue_mutations_parcels.values),:]
                    .sort_values(by=['id_parcelle']
                    )
          )
nuni_ue.tail(4).T

#%% 

# Drop duplicate properties
id_ftrs = ['id_parcelle'
           #, 'id_mutation', 'date_mutation', 'numero_disposition', 'nature_mutation'
           , 'valeur_fonciere'
           ]
data_unique = data_in2.drop_duplicates(subset=id_ftrs, keep='last')

print(data.shape)


#%% 

# Select only Sell operations

data = data_unique.loc[data_unique.nature_mutation == 'Vente',:]
data.shape

#%% 
# Update usable ftrs categories Nature mutation will be a not usable ftrs
cat_ftrs.remove('nature_mutation')
ftrs.remove('nature_mutation')
unuseful_ftrs.append('nature_mutation')




#%%md 
### Explore Target Feature
###### First Plotting target_value in thousands to facilitate visualization


#%%md 
###### Original target values are very left skewed

#%%
###### Add sale value in thousands for easiest management
###### Now plot log of target value
df_plot = data.sample(10000, random_state=SEED)

fig, ax = plt.subplots(figsize=(12,8))
(df_plot.valeur_fonciere/1000).plot.hist(ax = ax, bins = 10)
#sns.histplot(df_plot.valeur_fonciere/1000, ax=ax,  kde=False)
plt.show()



#%% md
#Applying log transformations gives a very centered distribution, There is, however a small separated group of very low valuew  which can, maybe be inspected to see if they are outliers.




#%%
# We add the scaled target value to use

data['valeur_fonciere_log'] = np.log1p(data['valeur_fonciere'])

# Now plot log of target value
df_plot = data.sample(10000, random_state=SEED)

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(np.log1p(df_plot.valeur_fonciere_log.values), ax=ax,  kde=True)
plt.show()

#%% md # Potentian outlier vals are sell values under 90 and are approx 1% of data

#%%


outlier_thr =  np.expm1(4.5)
print('Outlier threshold:', outlier_thr)
print(  data.loc[data['valeur_fonciere_log']<4.5,:].shape
      , data.loc[data['valeur_fonciere']<outlier_thr,:].shape)

print('Prop of outliers: {:.2f}'.format(
        data.loc[data['valeur_fonciere_log']<4.5,:].shape[0]/ data.shape[0]))

#%%
# Log distrib is centered but there is a small cluster on values close to 0. Let's explore it

identity_ftrs = ['id_parcelle', 'latitude', 'longitude',  'nature_culture']
atypic_low = data.loc[data.valeur_fonciere_log<1, :].sort_values(by=['valeur_fonciere', 'id_parcelle'])
#atypic_low.show()
atypic_low.shape[0] / data.shape[0]

atypic_low.shape


#%%
data.loc[data.valeur_fonciere_log<1, :].sort_values(by=['id_parcelle', 'valeur_fonciere']).head(100).nature_culture.value_counts()



#%% md 
### Exploratoty Analysis


#%% md 
###  Categorical features vs Target

#%%

# Set low cardinality features subset
summary.loc[summary.index.isin(cat_ftrs),:]

cat_ftrs_nunique_low = summary.loc[  summary.index.isin(cat_ftrs) & (summary['nunique'] < 28)
                                       , summary.columns[0:5]].index



#%% md 
### Plot each categorical value vs Target
# We see that for nature_mutation and culture_nature categories seem to influence target while type_local ones don't seem to have an impact. 


#%%
# Plot each categorical value vs Target
for f in cat_ftrs_nunique_low:
    data.loc[:,f] = data[f].astype('category')
    fig, ax = plt.subplots(figsize = (14,10))
    for cat in data.loc[:,f].cat.categories:
#        print(f,cat)
        # Select the category type
        subset = data.loc[data[f] == cat, :]
        
        # Density plot of Energy Star scores
        sns.kdeplot(subset['valeur_fonciere_log'].dropna(),
                   label = cat, shade = False, alpha = 0.8);
    
    # label the plot
    ax.legend()
    plt.xlabel('Value by {} ftr'.format(f), size = 20); plt.ylabel('Density', size = 20); 
    plt.title('Density Plot of {} Scores by {}'.format('valeur_foncier', f), size = 28);
    plt.show()



#%% #### md 
### Inspect Numeric Features
    
#By inspecting continuous features against the target we found next discoveries:
#1. Nombre de lots, surface reel batie et nombre de pieces principales semblent avoir
#2. Longitude et Latitude aussi semblen avoir une influence cépendant il existe un  petit cluster separé des autres dans lon[5,15] et lat [-60, -50]. Possibly delete these and not make predictions for outside Euope Continent France Territories
#Other numeric features dont seem to have an important contribution on value

#%%

for f in num_ftrs:
    data_plt = data.sample(1000, )
    sns.jointplot(x=f, y='valeur_fonciere_log', data=data_plt, kind="reg")
    plt.show()


#% On the other hand, features are not highly correlated to each other
#%%
#eda_num(data.loc[:,num_ftrs], method='correlation')
    
corr =  data.loc[:,num_ftrs].corr(method = 'pearson')
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="Greens")
plt.show()


#%% md 
### Predictive Capacity of Features
# Finally we inspect the predictive capacity of the features

    
#%%
pred_mtx = eda_numcat(data, x=None, y=None, method="pps")
#%%
#pred_mtx #This is only displayed in the notebook not in script execution

#%% md 
#### Export Summary of features to use


#%%


#%%
# Make sunmary of all original colunns and left rows


summary_out = explore(data_in.loc[data.index,:], method="summarize")

summary_out

#%%
# Add column with Boolean of usable or not ftr

summary['use_ftr'] = [True if f in ftrs else False for f in summary.index]
summary.loc[ftrs,smmry4]

summary['use_ftr']

#%%
summary.to_csv('../../data/interim/features_to_use_summary.csv', index = False) # Out summary to be used on modelassertion to know which ftrs to use

#%% md
### Export Filtered Row Data Set

#%% 
data.tail(1).T
summary.to_csv('../../data/interim/raw_useful_ftrs.csv', index = False) 



