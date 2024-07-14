#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('Bengaluru_House_Data.csv')


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df=df.drop(['area_type','availability','balcony','society'],axis=1)


# In[8]:


df


# In[9]:


df.isna().sum()


# In[10]:


df=df.dropna()


# In[11]:


df.isna().sum()


# In[12]:


df.shape


# In[13]:


df['size'].unique()


# In[15]:


df['BHK']=df['size'].apply(lambda x: int(x.split(' ')[0]))


# In[16]:


df.head()


# In[17]:


df['BHK'].unique()


# In[18]:


df[df.BHK>20]


# In[19]:


df.total_sqft.unique()


# In[20]:


def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


# In[21]:


df[~df['total_sqft'].apply(isfloat)].head(10)


# In[22]:


def convert_sqft_tonum(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[23]:


df=df.copy()
df['total_sqft']=df['total_sqft'].apply(convert_sqft_tonum)


# In[24]:


df.head(10)


# In[25]:


df.loc[30]


# In[26]:


df1=df.copy()
df1['price_per_sqft']=df1['price']*1000000/df1['total_sqft']
df1.head()


# In[27]:


len(df1.location.unique())


# In[28]:


df1.location=df1.location.apply(lambda x: x.strip())
location_stats=df1.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[29]:


len(location_stats[location_stats<=10])


# In[30]:


locationlessthan10=location_stats[location_stats<=10]
locationlessthan10


# In[32]:


len(df1.location.unique())


# In[33]:


df1.location=df1.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
len(df1.location.unique())


# In[34]:


df1.head(10)


# In[35]:


df1[df1.total_sqft/df1.BHK<300].head()


# In[36]:


df2=df1[~(df1.total_sqft/df1.BHK<300)]
df2.head(10)


# In[37]:


df2.shape


# In[38]:


df2["price_per_sqft"].describe().apply(lambda x:format(x,'f'))


# In[39]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df3=remove_pps_outliers(df2)
df3.shape


# In[40]:


import matplotlib.pyplot as plt
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.BHK==2)]
    bhk3=df[(df.location==location)&(df.BHK==3)]
    plt.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='Blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='green',marker='+',label='3 BHK',s=50)
    plt.xlabel('Total Square Foot')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
plot_scatter_chart(df3,"Rajaji Nagar")


# In[41]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats={}
        for BHK,BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'std':np.std(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats=bhk_sats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df4=remove_bhk_outliers(df3)
df4.shape


# In[42]:


plot_scatter_chart(df4,"Rajaji Nagar")


# In[44]:


plt.rcParams['figure.figsize']=(20,15)
plt.hist(df4.price_per_sqft,rwidth=0.6)
plt.xlabel("Price Per Square Foor")
plt.ylabel("Count")


# In[45]:


df4.bath.unique()


# In[46]:


df4[df4.bath>10]


# In[47]:


plt.rcParams['figure.figsize']=(20,15)
plt.hist(df4.bath,rwidth=0.6)
plt.xlabel("Number Of Bathroom")
plt.ylabel("Count")


# In[48]:


df4[df4.bath>df4.BHK+2]


# In[49]:


df5=df4[df4.bath<df4.BHK+2]
df5.shape


# In[50]:


df6=df5.drop(['size','price_per_sqft'],axis='columns')
df6


# In[51]:


dummies=pd.get_dummies(df6.location)
dummies.head(10)


# In[52]:


df7=pd.concat([df6,dummies.drop('other',axis='columns')],axis='columns')
df7.head()


# In[53]:


df8=df7.drop('location',axis='columns')
df8.head()


# In[54]:


df8.shape


# In[55]:


X=df8.drop('price',axis='columns')
X.head()


# In[56]:


y=df8.price


# In[57]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[58]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[59]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[61]:


def price_predict(location,sqft,bath,BHK):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=BHK
    if loc_index >=0:
        x[loc_index]=1
    return model.predict([x])[0]


# In[62]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[63]:


price_predict('1st Phase JP Nagar',1000,2,3)


# In[64]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[65]:


price_predict('Indira Nagar',1000,2,2)


# In[ ]:




