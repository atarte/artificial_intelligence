import os
import matplotlib.pyplot as plt
import pandas as pd

from numpy import savetxt

GENERAL_CSV = "./data/general_data.csv"
MANAGER_SURVEY_CSV = "./data/manager_survey_data.csv"
EMPLOYEE_SURVEY_CSV = "./data/employee_survey_data.csv"
IN_TIME_CSV = "./data/in_time.csv"
OUT_TIME_CSV = "./data/out_time.csv"
GRAPH_DIR = "./graph"

# Save graph
def Save_graph(graph_id, tight_layout=True, fig_extension="png", resolution=300):
    """
    This function is use to save graph into a png format
    """

    path = os.path.join(GRAPH_DIR, graph_id + "." + fig_extension)
    print("Saving figure:", graph_id)

    if tight_layout:
        plt.tight_layout()
    
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Load data
def Load_data(csv_path):
    """
    This function is use to load data from a csv file and return the data as a panda Dataframe
    """

    return pd.read_csv(csv_path)

def Save_pipeline_data(data):
    """
    This function is use to save the 
    """

    if not os.path.isdir('./pipeline'):
        os.mkdir('./pipeline')
        
    savetxt('./pipeline/data.csv', data, delimiter=',')

# Save model

