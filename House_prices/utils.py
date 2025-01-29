import pandas as pd

problem_title = "House price prediction"
_target_column_name = "SalePrice"

def get_train_data(path="data/train.csv"):
    data = pd.read_csv(path)
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis = 1)
    return X_df, y_array