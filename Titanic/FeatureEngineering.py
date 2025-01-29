import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# DATA ENCODING
# ==============================================================================
def _encode(X, preprocessor=None, y=None):
    """
    We will do Feature Engineering related to the values of our dataframe to:
    - encode categorical variables

    Parameters:
    X (pd.DataFrame): Input DataFrame with a numerical and categorical columns.

    Returns:
    pd.DataFrame: Transformed DataFrame with encoded categorical features.
    """
    
    X = X.drop(["PassengerId", "Name"], axis=1)

    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                categorical_cols)
                ],
            remainder='passthrough'  # Retain non-categorical columns
        )
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    transformed_columns = (
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    )

    column_names = list(transformed_columns) + list(X.select_dtypes(include = [np.number]).columns)

    if y is not None:
        return pd.DataFrame(X_transformed, columns=column_names), y, preprocessor
    else:
        return pd.DataFrame(X_transformed, columns=column_names)