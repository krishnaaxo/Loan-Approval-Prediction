import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.metrics import f1_score

import xgboost as xgb

import pickle
import os


def divide_per_type(data, dtype):
  cols = data.select_dtypes([dtype]).columns
  return data[cols], cols


def get_null_cols(data, dtype):
  null_value_columns =[y for x, y in zip(data.isnull().sum(), data.isnull().sum().index) if x > 0]
  _, cols = divide_per_type(data, dtype)
  null_cols = [x for x in cols if x in null_value_columns]
  
  return data[null_cols]

def handle_missing_values(data):
  cat_data = get_null_cols(data, 'object')
  num_data = get_null_cols(data, 'number')

  if not cat_data.empty:
    imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit(data[cat_data.columns])
    dat_cat = imputer_cat.transform(cat_data)
    for i, col in enumerate(cat_data.columns): 
      data[col] = dat_cat[:,i].ravel()

  if not num_data.empty:
    imputer_num = SimpleImputer(missing_values=np.nan, strategy='median').fit(data[num_data.columns])
    dat_num = imputer_num.transform(num_data)
    for i, col in enumerate(num_data.columns):
      data[col] = dat_num[:,i].ravel()

def map_dependents(data):
  map_dependents = {'0': 0, '1': 1, '2':2, '3+':3}
  data['Dependents'] = data['Dependents'].map(map_dependents)

def need_hot_encoding(data):
  _ , cat_cols = divide_per_type(data, 'object')
  for cols in cat_cols:
    print("*"*10 + cols + "*"*10)
    print(data[cols].value_counts().shape[0] == 2)

def handle_cat_data(data):
  enc = LabelBinarizer()
  enc_h = OneHotEncoder(handle_unknown='ignore')
  _ , cat_cols = divide_per_type(data, 'object')

#   print(cat_cols)

  for col in cat_cols:
    # print(col)
    if data[col].value_counts().shape[0] <= 2:
      enc.fit(data[col])
      df = pd.DataFrame(enc.transform(data[col]), columns=[col])
      df.index = data.index
      data = data.drop(columns=col)
      data = pd.concat([data, df], axis=1)
 
    elif data[col].value_counts().shape[0] > 2:
      values = enc_h.fit_transform(data[[col]]).toarray()
      cols = [col+'_'+str(x) for x in range(values.shape[1])]
      df = pd.DataFrame(data= values, columns=cols)
      df.index = data.index
      data = data.drop(columns=col)
      data = pd.concat([data, df], axis=1)
  
  return data





if __name__ == "__main__":
    
    # Get data
    loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )
    data = loan_data.drop(columns=['Unnamed: 0', 'Loan_ID'])
    X = data.drop(columns='Loan_Status')
    y = data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=123, stratify=y, shuffle=True)

    # Preprocessing data
    handle_missing_values(X_train)
    map_dependents(X_train)
    X_train = handle_cat_data(X_train)

    handle_missing_values(X_test)
    map_dependents(X_test)
    X_test = handle_cat_data(X_test)

    # Build model
    xgc_model = xgb.XGBClassifier(n_estimators=500, max_depth=5, random_state=123, use_label_encoder=False)
    xgc_model.fit(X_train, y_train)

    # Test model
    # print("f1-score {} ".format(f1_score(y_test, xgc_model.predict(X_test))))

    # Save model
    filename = 'loan_model.pkl'
    pickle.dump(xgc_model, open(filename,'wb'))
