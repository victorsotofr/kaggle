import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# DATA ENCODING
# ==============================================================================
def _encode(X,y=None):
    """
    We will do Feature Engineering related to the values of our dataframe to:
    - encode categorical variables
    - apply a log transformation to the prediction variable

    Parameters:
    X (pd.DataFrame): Input DataFrame with a numerical and categorical columns.

    Returns:
    pd.DataFrame: Transformed DataFrame with encoded categorical features.
    """
    
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             categorical_cols)
        ],
        remainder='passthrough'  # Retain non-categorical columns
    )

    X_transformed = preprocessor.fit_transform(X)

    transformed_columns = (
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    )

    column_names = list(transformed_columns) + list(X.select_dtypes(include = [np.number]).columns)

    if y is not None:
        return pd.DataFrame(X_transformed, columns=column_names), np.log(y)
    else:
        return pd.DataFrame(X_transformed, columns=column_names)