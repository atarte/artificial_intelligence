import utils as utl

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer



# il faudra ajouter une fonction fpour ajouter les colone en plus``

USELESS_COLUMNS_TO_DROP = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeID']
ETHICAL_COLUMNS_TO_DROP = ['Age', 'Gender', 'DistanceFromHome', 'MaritalStatus']




transform_pipeline = Pipeline([
    # ('imputer', SimpleImputer(strategy="median")),
    ('imputer', KNNImputer(n_neighbors=5)),
    ('std_scaler', StandardScaler()),
])

def Get_Columns_To_HotEncoder(data, columns_list):
    OneHotEncoderColumns = []
    OrdinalEncoderColumns = []
    
    for column in columns_list:
        count = data[column].value_counts()

        if count.size > 2:
            OneHotEncoderColumns.append(column)
        else:
            OrdinalEncoderColumns.append(column)

    return (OneHotEncoderColumns, OrdinalEncoderColumns)

# def Get_Columns_To_OrdinalEncoder(data, columns_list):

def TransformData():

    # Load csv data file
    general_data = pd.read_csv(utl.GENERAL_CSV)
    manager_survey_data = pd.read_csv(utl.MANAGER_SURVEY_CSV)
    employee_survey_data = pd.read_csv(utl.EMPLOYEE_SURVEY_CSV)
    # in_data = pd.read_csv(utl.IN_TIME_CSV)
    # out_data = pd.read_csv(utl.OUT_TIME_CSV)
    # print(general_data.head())

    # merge data together
    full_data = pd.merge(general_data, manager_survey_data, on='EmployeeID')
    full_data = pd.merge(full_data, employee_survey_data, on='EmployeeID')
    print(full_data.head())

    # Drop some columns
    full_data.drop(ETHICAL_COLUMNS_TO_DROP, axis=1, inplace=True)
    full_data.drop(USELESS_COLUMNS_TO_DROP, axis=1, inplace=True)
    print(full_data.head())

    general_num = full_data.select_dtypes(include=[np.number]) 
    # print("general num")
    # print(general_num)
    # print()

    num_attribs = list(general_num)
    # print("num attribs")
    # print(num_attribs)
    # print(type(num_attribs))

    general_string = full_data.select_dtypes(include=[object])
    string_attribs = list(general_string)
    # print(string_attribs)

    encoded_atribs = Get_Columns_To_HotEncoder(full_data, string_attribs)
    # print(encoded_atribs)


    full_pipeline = ColumnTransformer([
        ('transform', transform_pipeline, num_attribs),
        ('ordinal', OrdinalEncoder(), encoded_atribs[1]),
        ('one_hot', OneHotEncoder(), encoded_atribs[0]),
    ])

    treated_data = full_pipeline.fit_transform(full_data)


    print(treated_data[0])
    print(len(treated_data[0]))
    print(type(treated_data))

    utl.Save_pipeline_data(treated_data)

    return treated_data