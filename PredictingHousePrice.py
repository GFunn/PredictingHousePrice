
# coding: utf-8

# # 读入数据

# In[84]:

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

pd.options.display.max_rows = 999
pd.set_option('display.max_columns', 999)


# In[85]:

data = pd.read_csv('AmesHousing.txt', sep='\t')

train_raw = data.iloc[:1460]
test_raw = data.iloc[1460:]

train_subset = train_raw
train_subset.shape


# # 主方程部分：

# In[86]:

def transform_features(df):
    
    ### Dropping null values ###
    
    null_thd = len(df) / 2
    null_total = df.isnull().sum()
    for col in null_total.index:
        if null_total[col] > null_thd:
            df = df.drop(col, axis=1)
    
    
    ### Processing object columns ### 
    
    txt_train = df.select_dtypes(include=['object'])
            
    
    ### Processing numerical columns ###
    ## Variance values processing
    num_train = df.select_dtypes(include=['int', 'float'])
    num_var = num_train / num_train.max()
    var_series = num_var.var().sort_values()
    
    for col in var_series.index:
        if var_series[col] < 0.015:
            num_train = num_train.drop(col, axis=1)    

    return txt_train, num_train


# In[87]:

def select_features(df):
    
    txt_train, num_train = transform_features(df)

    
    ### Turn your model here ###
    
    txt_train = df.select_dtypes(include=['object'])
    
    txt_useless = ['Sale Type', 'Sale Condition']
    txt_train = txt_train.drop(txt_useless, axis=1)
    
    for col in txt_train.columns:
        txt_train[col] = txt_train[col].astype('category')
        col_dummies = pd.get_dummies(txt_train[col], prefix=col)
        txt_train = pd.concat([txt_train, col_dummies], axis=1)
        del txt_train[col]

    
    num_cat = []
    for col in num_train.columns:
        val_counts = len(num_train[col].value_counts())
        if val_counts < 15:
            num_cat.append(col)
    
    for col in num_cat:
        num_train[col] = num_train[col].astype('category')
        col_dummies = pd.get_dummies(num_train[col], prefix=col)
        num_train = pd.concat([num_train, col_dummies], axis=1)
        del num_train[col]

    num_train = num_train.fillna(num_train.mean())

    ### Turn your model here ###
    
    
    train_clean = pd.concat([txt_train, num_train], axis=1)
    return train_clean


# In[89]:

def train_and_test(df):
    data_sets = select_features(df)
    try:
        train_set = data_sets.drop('SalePrice', axis=1)
        target_set = data_sets[['SalePrice']]
    except:
        train_set = data_sets
        target_set = df[['SalePrice']]
    
    kf = KFold(5, shuffle=True, random_state=1)
    model = LinearRegression()
    
    mses = cross_val_score(model, train_set, target_set,
                           scoring='neg_mean_absolute_error', cv=kf)
    rmses = [np.sqrt(np.abs(mse)) for mse in mses]
    avg_rmse = sum(rmses)/len(rmses)
    return rmses, avg_rmse

result = train_and_test(data)
print(result)


# ### 结果如上方所示

# 

# 

# 

# # 实验测试部分：

# ### 处理null value

# In[29]:

null_thd = len(train_subset) / 2
null_total = train_subset.isnull().sum()

print('null门限设置为数据集长度一半：', null_thd)

for col in null_total.index:
    if null_total[col] > null_thd:
        train_subset = train_subset.drop(col, axis=1)
    
print('处理后数据集中null最多的有：', train_subset.isnull().sum().sort_values().max())


# ### 删除数字部分变化率很小的列

# In[30]:

num_train = train_subset.select_dtypes(include=['int', 'float'])
num_var = num_train / num_train.max()
var_series = num_var.var().sort_values()

for col in var_series.index:
    if var_series[col] < 0.015:
        num_train = num_train.drop(col, axis=1)
        var_series = var_series.drop(col)
print(var_series.apply(lambda x: '%.8f' %x))


# In[44]:

num_test = num_train
for col in num_test.columns:
    print(col)
    val_counts = len(num_test[col].value_counts())
    print(val_counts)


# In[31]:

num_clean = num_train
num_cat = ['Full Bath', 'Bedroom AbvGr', 'Overall Qual', 'Garage Cars', 'Bsmt Full Bath',
          'Fireplaces', 'MS SubClass', 'Bsmt Half Bath', 'Half Bath']


num_useless = ['PID', 'Mo Sold', 'Order']

for col in num_cat:
    num_clean[col] = num_clean[col].astype('category')
    col_dummies = pd.get_dummies(num_clean[col], prefix=col)
    num_clean = pd.concat([num_clean, col_dummies], axis=1)
    del num_clean[col]
for col in num_useless:
    del num_clean[col]

num_clean = num_clean.fillna(num_clean.mean())
print(num_clean.columns)


# In[32]:

txt_train = train_subset.select_dtypes(include=['object'])

for col in txt_train.columns:
    txt_train[col] = txt_train[col].astype('category')
    col_dummies = pd.get_dummies(txt_train[col], prefix=col)
    txt_train = pd.concat([txt_train, col_dummies], axis=1)
    del txt_train[col]
print(txt_train.columns)


# In[ ]:




# In[33]:

train_clean = pd.concat([num_clean, txt_train], axis=1)
print(train_clean.columns)


# In[34]:

train_clean.isnull().sum().sort_values()


# 

# 

# In[35]:

def transform_features():
    train = train_clean
    return train


# In[36]:

def select_features():
    train = transform_features()
    return train


# In[37]:

def train_and_test():
    df = select_features()
    train_set = df.drop('SalePrice', axis=1)
    target_set = df[['SalePrice']]
    
    kf = KFold(5, shuffle=True, random_state=1)
    model = LinearRegression()
    
    mses = cross_val_score(model, train_set, target_set,
                           scoring='neg_mean_absolute_error', cv=kf)
    rmses = [np.sqrt(np.abs(mse)) for mse in mses]
    avg_rmse = sum(rmses)/len(rmses)
    return rmses, avg_rmse

result = train_and_test()
print(result)

