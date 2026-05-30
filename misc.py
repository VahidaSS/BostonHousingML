import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
     
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    df = pd.DataFrame(data)
    df["MEDV"] = target
    return df
     
def split_data(df):
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
     
def evaluate(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
