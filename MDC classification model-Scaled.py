#!/usr/bin/env python
# coding: utf-8

# # PR Assignment -4 MDC classification model(with Scaling)

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# importing data from txt file into dataframe
df = pd.read_csv("D:/ISIBangalore/MS-QMScourse/2nd sem/Pattern Recognition/Assignment-4/Vowel Data.txt", sep = '\s+', header=None, names=['class','x1','x2','x3'])


# In[3]:


df.iloc[:,1:4] = (df.iloc[:,1:4] - df.iloc[:,1:4].min())/(df.iloc[:,1:4].max() - df.iloc[:,1:4].min())


# In[4]:


# grouping data classwise
dflist = list(df.groupby('class'))


# In[5]:


# segregating dataframe class wise
df1 = dflist[0][1]
df2 = dflist[1][1]
df3 = dflist[2][1]
df4 = dflist[3][1]
df5 = dflist[4][1]
df6 = dflist[5][1]


# In[6]:


# segregating input features and output feature for each class
df1_x = df1.drop(columns=['class'])
df1_y = df1.iloc[:,0]
df2_x = df2.drop(columns=['class'])
df2_y = df2.iloc[:,0]
df3_x = df3.drop(columns=['class'])
df3_y = df3.iloc[:,0]
df4_x = df4.drop(columns=['class'])
df4_y = df4.iloc[:,0]
df5_x = df5.drop(columns=['class'])
df5_y = df5.iloc[:,0]
df6_x = df6.drop(columns=['class'])
df6_y = df6.iloc[:,0]


# In[7]:


"""# splitting each data frame into training and test data
df1_x_train, df1_x_test, df1_y_train, df1_y_test = train_test_split(df1_x, df1_y, test_size = testsize[i])
df2_x_train, df2_x_test, df2_y_train, df2_y_test = train_test_split(df2_x, df2_y, test_size = testsize[i])
df3_x_train, df3_x_test, df3_y_train, df3_y_test = train_test_split(df3_x, df3_y, test_size = testsize[i])
df4_x_train, df4_x_test, df4_y_train, df4_y_test = train_test_split(df4_x, df4_y, test_size = testsize[i])
df5_x_train, df5_x_test, df5_y_train, df5_y_test = train_test_split(df5_x, df5_y, test_size = testsize[i])
df6_x_train, df6_x_test, df6_y_train, df6_y_test = train_test_split(df6_x, df6_y, test_size = testsize[i])
        
scaler1 = StandardScaler()
scaler1.fit(df1_x_train)
df1_x_train = pd.DataFrame(scaler1.transform(df1_x_train))
df1_x_test = pd.DataFrame(scaler1.transform(df1_x_test))
scaler2 = StandardScaler()
scaler2.fit(df2_x_train)
df2_x_train = pd.DataFrame(scaler2.transform(df2_x_train))
df2_x_test = pd.DataFrame(scaler2.transform(df2_x_test))
scaler3 = StandardScaler()
scaler3.fit(df3_x_train)
df3_x_train = pd.DataFrame(scaler3.transform(df3_x_train))
df3_x_test = pd.DataFrame(scaler3.transform(df3_x_test))
scaler4 = StandardScaler()
scaler4.fit(df4_x_train)
df4_x_train = pd.DataFrame(scaler4.transform(df4_x_train))
df4_x_test = pd.DataFrame(scaler4.transform(df4_x_test))
scaler5 = StandardScaler()
scaler5.fit(df5_x_train)
df5_x_train = pd.DataFrame(scaler5.transform(df5_x_train))
df5_x_test = pd.DataFrame(scaler5.transform(df5_x_test))
scaler6 = StandardScaler()
scaler6.fit(df6_x_train)
df6_x_train = pd.DataFrame(scaler6.transform(df6_x_train))
df6_x_test = pd.DataFrame(scaler6.transform(df6_x_test))
        
# combining training and test data from each class
x_train = pd.concat([df1_x_train, df2_x_train, df3_x_train, df4_x_train, df5_x_train, df6_x_train], axis=0)
y_train = pd.concat([df1_y_train, df2_y_train, df3_y_train, df4_y_train, df5_y_train, df6_y_train], axis=0)
x_test = pd.concat([df1_x_test, df2_x_test, df3_x_test, df4_x_test, df5_x_test, df6_x_test], axis=0)
y_test = pd.concat([df1_y_test, df2_y_test, df3_y_test, df4_y_test, df5_y_test, df6_y_test], axis=0)

scaler1 = StandardScaler()
scaler1.fit(x_train)
x_train = pd.DataFrame(scaler1.transform(x_train))
x_test = pd.DataFrame(scaler1.transform(x_test))

# centroid from training data of each class
cent1 = np.array(df1_x_train.mean())
cent2 = np.array(df2_x_train.mean())
cent3 = np.array(df3_x_train.mean())
cent4 = np.array(df4_x_train.mean())
cent5 = np.array(df5_x_train.mean())
cent6 = np.array(df6_x_train.mean())
# converting training & test input features dataframe to array form
#x_train_arr = x_train.to_numpy()
x_test_arr = x_test.to_numpy()
# predicting output class for training data
y_pred_train = []
for k in range(x_train['x1'].count()):
    x = x_train_arr[k]
    d1 = np.linalg.norm(x - cent1)
    d2 = np.linalg.norm(x - cent2)
    d3 = np.linalg.norm(x - cent3)
    d4 = np.linalg.norm(x - cent4)
    d5 = np.linalg.norm(x - cent5)
    d6 = np.linalg.norm(x - cent6)
    d = [d1, d2, d3, d4, d5, d6]
    y_pred_train.append(d.index(min(d)) + 1)
    ypred_train = np.array(y_pred_train)
    acc_table_train = accuracy_score(y_train, ypred_train)
# predicting output class for test data
y_pred_test = []
for k in range(x_test['x1'].count()):
    x = x_test_arr[k]
    d1 = np.linalg.norm(x - cent1)
    d2 = np.linalg.norm(x - cent2)
    d3 = np.linalg.norm(x - cent3)
    d4 = np.linalg.norm(x - cent4)
    d5 = np.linalg.norm(x - cent5)
    d6 = np.linalg.norm(x - cent6)
    d = [d1, d2, d3, d4, d5, d6]
    y_pred_test.append(d.index(min(d)) + 1)
    
ypred_test = np.array(y_pred_test)
acc_table_test = accuracy_score(y_test, ypred_test)
#print(acc_table_train)
print(acc_table_test)"""


# In[8]:


testsize = [0.1, 0.2, 0.3, 0.4]
acc_table_train = np.empty((4, 10))
acc_table_test = np.empty((4, 10))
for i in range(4):
    for j in range(10):
        # splitting each data frame into training and test data
        df1_x_train, df1_x_test, df1_y_train, df1_y_test = train_test_split(df1_x, df1_y, test_size = testsize[i])
        df2_x_train, df2_x_test, df2_y_train, df2_y_test = train_test_split(df2_x, df2_y, test_size = testsize[i])
        df3_x_train, df3_x_test, df3_y_train, df3_y_test = train_test_split(df3_x, df3_y, test_size = testsize[i])
        df4_x_train, df4_x_test, df4_y_train, df4_y_test = train_test_split(df4_x, df4_y, test_size = testsize[i])
        df5_x_train, df5_x_test, df5_y_train, df5_y_test = train_test_split(df5_x, df5_y, test_size = testsize[i])
        df6_x_train, df6_x_test, df6_y_train, df6_y_test = train_test_split(df6_x, df6_y, test_size = testsize[i])
        # combining training and test data from each class
        x_train = pd.concat([df1_x_train, df2_x_train, df3_x_train, df4_x_train, df5_x_train, df6_x_train], axis=0)
        y_train = pd.concat([df1_y_train, df2_y_train, df3_y_train, df4_y_train, df5_y_train, df6_y_train], axis=0)
        x_test = pd.concat([df1_x_test, df2_x_test, df3_x_test, df4_x_test, df5_x_test, df6_x_test], axis=0)
        y_test = pd.concat([df1_y_test, df2_y_test, df3_y_test, df4_y_test, df5_y_test, df6_y_test], axis=0)
        # centroid from training data of each class
        cent1 = np.array(df1_x_train.mean())
        cent2 = np.array(df2_x_train.mean())
        cent3 = np.array(df3_x_train.mean())
        cent4 = np.array(df4_x_train.mean())
        cent5 = np.array(df5_x_train.mean())
        cent6 = np.array(df6_x_train.mean())
        # converting training & test input features dataframe to array form
        x_train_arr = x_train.to_numpy()
        x_test_arr = x_test.to_numpy()
        # predicting output class for training data
        y_pred_train = []
        for k in range(x_train['x1'].count()):
            x = x_train_arr[k]
            d1 = np.linalg.norm(x - cent1)
            d2 = np.linalg.norm(x - cent2)
            d3 = np.linalg.norm(x - cent3)
            d4 = np.linalg.norm(x - cent4)
            d5 = np.linalg.norm(x - cent5)
            d6 = np.linalg.norm(x - cent6)
            d = [d1, d2, d3, d4, d5, d6]
            y_pred_train.append(d.index(min(d)) + 1)
        ypred_train = np.array(y_pred_train)
        acc_table_train[i][j] = accuracy_score(y_train, ypred_train)
        # predicting output class for test data
        y_pred_test = []
        for k in range(x_test['x1'].count()):
            x = x_test_arr[k]
            d1 = np.linalg.norm(x - cent1)
            d2 = np.linalg.norm(x - cent2)
            d3 = np.linalg.norm(x - cent3)
            d4 = np.linalg.norm(x - cent4)
            d5 = np.linalg.norm(x - cent5)
            d6 = np.linalg.norm(x - cent6)
            d = [d1, d2, d3, d4, d5, d6]
            y_pred_test.append(d.index(min(d)) + 1)
        ypred_test = np.array(y_pred_test)
        acc_table_test[i][j] = accuracy_score(y_test, ypred_test)

acc_table_train_df = pd.DataFrame(np.transpose(np.round(acc_table_train*100,2)), columns=['90-10', '80-20', '70-30', '60-40'], index=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th'])
acc_table_test_df  = pd.DataFrame(np.transpose(np.round(acc_table_test*100, 2)), columns=['90-10', '80-20', '70-30', '60-40'], index=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th'])


# In[9]:


acc_table_train_df.loc['Mean acc_train'] = acc_table_train_df.mean()
acc_table_train_df


# In[10]:


acc_table_test_df.loc['Mean acc_test'] = acc_table_test_df.mean()
acc_table_test_df


# In[11]:


pd.concat([acc_table_train_df.iloc[[10]], acc_table_test_df.iloc[[10]]], axis = 0).to_csv('D:/ISIBangalore/MS-QMScourse/2nd sem/Pattern Recognition/Assignment-4/accuracy table_MDC_scaled.csv', sep=',', header=True, index=True)


# In[ ]:




