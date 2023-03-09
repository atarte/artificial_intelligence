import utils as utl

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer



# il faudra ajouter une fonction fpour ajouter les colone en plus``

COLUMNS_TO_DROP = ['Age', 'Gender', 'DistanceFromHome', 'MaritalStatus', 'Over18']




transform_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
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

    general_data = utl.Load_data(utl.GENERAL_CSV)
    print(general_data.head())
    # general_data = Drop_Columns(general_data)
    general_data.drop(COLUMNS_TO_DROP, axis=1, inplace=True)
    print(general_data.head())

    general_num = general_data.select_dtypes(include=[np.number]) 
    # print("general num")
    # print(general_num)
    # print()

    num_attribs = list(general_num)
    # print("num attribs")
    # print(num_attribs)
    # print(type(num_attribs))

    general_string = general_data.select_dtypes(include=[object])
    string_attribs = list(general_string)
    print(string_attribs)

    encoded_atribs = Get_Columns_To_HotEncoder(general_data, string_attribs)
    print(encoded_atribs)


    full_pipeline = ColumnTransformer([
        ('transform', transform_pipeline, num_attribs),
        ('ordinal', OrdinalEncoder(), encoded_atribs[1]),
        ('one_hot', OneHotEncoder(), encoded_atribs[0])
    ])

    treated_data = full_pipeline.fit_transform(general_data)

    print(treated_data[0])
    print(len(treated_data[0]))
    print(type(treated_data))

    utl.Save_pipeline_data(treated_data)



# class FullPipeline:
#     def __init__(self, data):
#         pass

#     def transform(self):
#         pass

#     transform_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy="median")),
#         ('std_scaler', StandardScaler()),
#     ])

#     housing_num = housing.select_dtypes(include=[np.number]) 
#     num_attribs = list(housing_num)

#     full_pipeline = ColumnTransformer([
#         ('transform', transform_pipeline, num_attribs),

#     ])