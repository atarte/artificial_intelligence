import utils as utl
import time
import datetime
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

USELESS_COLUMNS_TO_DROP = ['Attrition', 'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeID']
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

# def MeanDataframe(data):
#     # mean = 0
#     nb_row = len(in_data)
#     nb_column = len(in_data.columns)

#     for row in range(nb_row):
#         mean_sum = 0
#         mean_nb = 0

#         for column in range(1, nb_column):
#             print(row)
#             print(column)
#             value_str = str(in_data.iat[row, column])

#             if value_str != "nan":
#                 value = time.mktime(datetime.datetime.strptime(value_str, "%Y-%m-%d %H:%M:%S").timetuple())
#                 print(value)
            
#                 mean_nb += 1
#                 mean_sum += value

#         mean = mean_sum / mean_nb
#         # print(mean)

def InOutMean():
    in_data = pd.read_csv(utl.IN_TIME_CSV)
    out_data = pd.read_csv(utl.OUT_TIME_CSV)

    nb_row = len(in_data)
    nb_column = len(in_data.columns)

    mean_array = []

    for row in range(nb_row):
        mean_sum_in = 0
        mean_sum_out = 0
        
        mean_nb_in = 0
        mean_nb_out = 0

        for column in range(1, nb_column):
            value_str_in = str(in_data.iat[row, column])
            value_str_out = str(out_data.iat[row, column])

            if value_str_in != "nan":
                value_in = time.mktime(datetime.datetime.strptime(value_str_in, "%Y-%m-%d %H:%M:%S").timetuple())
            
                mean_nb_in += 1
                mean_sum_in += value_in

            if value_str_out != "nan":
                value_out = time.mktime(datetime.datetime.strptime(value_str_out, "%Y-%m-%d %H:%M:%S").timetuple())
            
                mean_nb_out += 1
                mean_sum_out += value_out


        # mean_in = mean_sum_in / mean_nb_in
        # mean_out = mean_sum_out / mean_nb_out
        # standart_day = 28800

        mean = (mean_sum_out / mean_nb_out) - (mean_sum_in / mean_nb_in) - 28800
        mean_array.append(mean)
        # print(mean)

    data_dict = {'EmployeeID': list(range(1, nb_row+1)), 'MeanHours': mean_array}
    data = pd.DataFrame(data=data_dict)

    return data

def TransformData():

    # Load csv data file
    general_data = pd.read_csv(utl.GENERAL_CSV)
    manager_survey_data = pd.read_csv(utl.MANAGER_SURVEY_CSV)
    employee_survey_data = pd.read_csv(utl.EMPLOYEE_SURVEY_CSV)
    in_out_data = InOutMean()

    # merge data together
    full_data = pd.merge(general_data, manager_survey_data, on='EmployeeID')
    full_data = pd.merge(full_data, employee_survey_data, on='EmployeeID')
    full_data = pd.merge(full_data, in_out_data, on='EmployeeID')
    print(full_data.head())

    # Drop some columns
    full_data.drop(ETHICAL_COLUMNS_TO_DROP, axis=1, inplace=True)
    full_data.drop(USELESS_COLUMNS_TO_DROP, axis=1, inplace=True)
    print(full_data.head())

    # Extract the target data ('Attrition') form the data set
    target_value = general_data.copy()

    attris_without_attrition = list(target_value.columns.values)
    attris_without_attrition.remove('Attrition')
    target_value.drop(attris_without_attrition, axis=1, inplace=True)

    enc = OrdinalEncoder()
    target_value = enc.fit_transform(target_value)

    # Get the collumns containing int
    general_num = full_data.select_dtypes(include=[np.number]) 
    num_attribs = list(general_num)

    # Get the columns containing string
    general_string = full_data.select_dtypes(include=[object])
    string_attribs = list(general_string)

    # Split the columns to into two set of columns (the one to hot encod, the one one to ordinal encod)
    encoded_atribs = Get_Columns_To_HotEncoder(full_data, string_attribs)
    # print(encoded_atribs)


    full_pipeline = ColumnTransformer([
        ('transform', transform_pipeline, num_attribs),
        ('ordinal', OrdinalEncoder(), encoded_atribs[1]),
        ('one_hot', OneHotEncoder(), encoded_atribs[0]),
    ])

    # Execute the pipeline
    treated_data = full_pipeline.fit_transform(full_data)


    print(treated_data[0])
    print(len(treated_data[0]))
    print(type(treated_data))

    # utl.Save_pipeline_data(treated_data)

    return treated_data, target_value