#!/usr/bin/env python
# coding: utf-8

# In[193]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


# In[194]:


# Data Cleaning
df_1 = pd.read_csv("Bengaluru_House_Price_Data.csv")
df_1.head()


df_1.groupby('area_type')['area_type'].agg('count')

df_2 = df_1.drop(['area_type', 'availability', 'society', 'balcony', 'availability'], axis='columns')
df_2.head()


# In[197]:


df_2.isnull().sum()

df_3 = df_2.dropna()


# In[199]:


df_3['size'].unique()


# In[200]:


df_4=df_3.copy()
df_4['BHK']=df_3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[201]:


df_5=df_4.drop(['size'],axis='columns')


# In[202]:


df_5.head()


# In[203]:


df_5['BHK'].unique()
df_6=df_5[df_5['BHK']<17]


# In[232]:


df_6.head()


# In[233]:


df_6.total_sqft.unique()


# In[234]:


def convert_sqft_to_num(x):
    try:
        return float(x)
    except:
        tokens=x.split('-')
        if len(tokens)==2:
            return (float(tokens[0])+float(tokens[1]))/2
    else:
        return None
        


# In[235]:


df_7=df_6.copy()
df_7['total_sqft']=df_6['total_sqft'].apply(convert_sqft_to_num)


# In[236]:


df_7.head()


# In[237]:


df_8=df_7.copy()
df_8['price_per_sqft']=df_8['price']*100000/df_7['total_sqft']
df_8.head()


# In[238]:


len(df_8.location.unique())


# In[239]:


df_8['location']=df_8['location'].apply(lambda x: x.strip())

location_stats=df_8.groupby('location')['location'].agg('count').sort_values(ascending=False)



# In[240]:


len(location_stats[location_stats<10])


# In[241]:


location_stats_less_than_10=location_stats[location_stats<10]
location_stats_less_than_10


# In[242]:


df_8['location']=df_8.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df_8.location.unique())


# In[243]:


df_8.head(10)


# In[244]:


df_8[df_8.total_sqft/df_8.BHK<300].head()


# In[245]:


df_9=df_8[~(df_8.total_sqft/df_8.BHK<300)]
df_9.head()


# In[246]:


df_9.price_per_sqft.describe()


# In[247]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        sd=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-sd)) & (subdf.price_per_sqft<(m+sd))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df_10=remove_pps_outliers(df_9)
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location) & (df.BHK==2)]
    bhk3=df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,marker='o',color='blue',label='2bhk',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3bhk',s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()


# In[249]:


plot_scatter_chart(df_10,"Hebbal")


# In[250]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for BHK, bhk_df in location_df.groupby('BHK'):
            bhk_stats[BHK]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'sd':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for BHK,bhk_df in location_df.groupby('BHK'):
            stats=bhk_stats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')               

df_11=remove_bhk_outliers(df_10)
df_11.shape


# In[251]:


plot_scatter_chart(df_11,"Hebbal")


# In[252]:


matplotlib.rcParams['figure.figsize']=(20,10)
plt.hist(df_11.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[253]:


df_11.bath.unique()


# In[254]:


df_11[df_11.bath>10]


# In[255]:


df_12=df_11[df_11.bath<df_11.BHK+2]
df_12.shape


# In[256]:


df_13=df_12.drop('price_per_sqft',axis='columns')
df_13.head()


# In[257]:


dummies=pd.get_dummies(df_13.location)
dummies.head()


# In[258]:


df_14=pd.concat([df_13,dummies.drop('other',axis='columns')],axis='columns')
df_14.head()

df_15=df_14.drop('location',axis='columns')
df_15.head()


# In[261]:


df=df_15

X=df.drop('price',axis='columns')
X.head()

y=df.price
y.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[265]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[266]:


model=LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[267]:


cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)

cross_val_score(LinearRegression(),X,y,cv=cv)


# In[268]:


def best_model_using_gridsearchcv(X,y):
    algorithms={
        'linear_regression':{
            'model':make_pipeline(StandardScaler(with_mean=False), LinearRegression()),
            'params':{}
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decision_tree':{
            'model':DecisionTreeRegressor(criterion='squared_error'),
            'params':{
                'splitter':['best','random']
            }
        }
    }

    scores=[]
    cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algorithms.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model', 'best_score', 'best_params'])

best_model_using_gridsearchcv(X,y)

def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
    
    return model.predict([x])[0]

print(predict_price('1st Phase JP Nagar',1000,2,2))

print(predict_price('Indira Nagar',1000,2,2))
