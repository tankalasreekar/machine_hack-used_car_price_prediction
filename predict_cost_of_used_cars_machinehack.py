#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[35]:


import pandas as pd
import numpy as np
data_train = pd.read_excel("Data_Train.xlsx")
data_test = pd.read_excel('Data_Test.xlsx')


# In[36]:


data_train.drop('New_Price',axis = 1,inplace = True)
data_test.drop('New_Price' , axis=1 , inplace = True)


# New price column is dropped since most of the car's new price is unknown

# In[37]:


y_train_data = data_train.iloc[:,-1].values
data_train = data_train.iloc[:,:-1]


# In[38]:


data_train[data_train.isnull().any(axis=1)]
data_test[data_test.isna().any(axis=1)]


# A function 'change' is used to convert the process the string values present in the series.

# In[39]:


def change(series):
    shifted = []
    for i,item in enumerate(series):
        if (type(item)==float) or item.split()[0]=='null':
            shifted.append(np.nan)
        else:
            shifted.append(float(item.split()[0]))
    return shifted


# In[40]:


str_col = ['Engine','Mileage','Power']
for item in str_col:
    data_train[item] = pd.Series(change(data_train[item]))
    data_test[item] = pd.Series(change(data_test[item]))


# The 'brand_model_split' splits the name of the car column to two seperate series : 'Brand' and 'Model'

# In[41]:


def brand_model_split(series):
    brand = []
    model = []
    for i,item in enumerate(series.Name):
        brand.append(item.split()[0])
        model.append(item[len(item.split()[0]):])
    series.insert(loc = 0,value = pd.Series(brand), column = 'Brand')
    series.insert(loc = 1,value = pd.Series(model), column = 'Model')
    series.drop('Name',axis=1,inplace=True)


# In[42]:


brand_model_split(data_train)
brand_model_split(data_test)


# Label Encoding the categorical variables

# In[43]:


from sklearn.preprocessing import LabelEncoder
enc_ind = [0,1,2,5,6,7]
le_enc = []
for j in enc_ind:
    le = LabelEncoder()
    le_enc.append(le)
    le.fit(pd.concat([data_train.iloc[:,j],data_test.iloc[:,j]],ignore_index = True))
    data_train.iloc[:,j] = le.transform(data_train.iloc[:,j])
    data_test.iloc[:,j] = le.transform(data_test.iloc[:,j])    


# Imputing the missing values in columns : Seats, Power, Engine, Mileage

# In[44]:


from sklearn.impute import SimpleImputer
imp_seats = SimpleImputer(missing_values = np.nan , strategy = 'constant' , fill_value = 4)
imp_power_engine_mileage = SimpleImputer(missing_values = np.nan , strategy = 'mean')
data_train['Seats'] = imp_seats.fit_transform(data_train['Seats'].values.reshape(-1,1))
data_train.iloc[:,8:11] = imp_power_engine_mileage.fit_transform(data_train.iloc[:,8:11])
data_test['Seats'] = imp_seats.transform(data_test['Seats'].values.reshape(-1,1))
data_test.iloc[:,8:11] = imp_power_engine_mileage.transform(data_test.iloc[:,8:11])


# from sklearn.preprocessing import OneHotEncoder
# enco_ind = ['Brand','Model','Location','Fuel_Type','Transmission','Owner_Type']
# ohe_enc = []
# for j in enco_ind:
#     oe = OneHotEncoder()
#     ohe_enc.append(oe)
#     oe.fit([pd.concat([data_train[j],data_test[j]],ignore_index = True).values])
#     data_train[j] = oe.transform([data_train[j]]).toarray()
#     data_test[j] = oe.transform(data_test[j]).toarray()
#     data_train = data_train[:,1:]
#     data_test = data_test[:,1:]

# In[45]:


data_train


# Scaling the series to avoid the effect of variable larger vales

# In[46]:


from sklearn.preprocessing import StandardScaler
data_train = data_train[['Brand','Model','Location','Year','Fuel_Type','Transmission','Owner_Type','Kilometers_Driven','Mileage','Engine','Power','Seats']]
data_test = data_test[['Brand','Model','Location','Year','Fuel_Type','Transmission','Owner_Type','Kilometers_Driven','Mileage','Engine','Power','Seats']]
sc = StandardScaler()
sc.fit(data_train.iloc[:,7:11])
data_train.iloc[:,7:11] = sc.transform(data_train.iloc[:,7:11])
data_test.iloc[:,7:11] = sc.transform(data_test.iloc[:,7:11])


# In[47]:


data_train


# In[ ]:


from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV

import xgboost as xgb


# In[ ]:


from sklearn.model_selection import train_test_split
X_train , X_val , y_train  , y_val = train_test_split(data_train,y_train_data,test_size=0.2)


# In[ ]:


xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=0)
xgb_model.fit(X_train, y_train)


# In[ ]:


y_pred = xgb_model.predict(X_val)
mse=mean_squared_error(y_val, y_pred)
print(np.sqrt(mse))


# In[ ]:


y_pred2 = xgb_model.predict(data_test)
y_pred2 = pd.DataFrame(y_pred2 , columns = ['Price'])
#y_pred2.to_excel("output3.xlsx",index = False)


# In[ ]:


from sklearn.metrics import mean_squared_log_error
print(np.sqrt(mean_squared_log_error(y_val,y_pred)))


# In[ ]:


y_pred2.Price


# In[ ]:





# In[ ]:





# In[ ]:




