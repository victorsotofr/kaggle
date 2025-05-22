import numpy as np
import pandas as pd

from utils import get_train_data
from FeatureEngineering import _encode

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================================================================

# DATA
X, y = get_train_data()


# FEATURE ENGINEERING
X_encoded, y_log = _encode(X, y)


# MODEL
# Best parameters from Optuna
model = XGBRegressor(
        n_estimators=291,
        learning_rate=0.07256489894558181,
        max_depth=3,
        random_state=42
    )

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_log, 
                                                  test_size=0.2, 
                                                  random_state=42)

model.fit(X_train, y_train)

# PREDICTION
test_data = pd.read_csv('data/test.csv')
test_data_encoded = _encode(test_data).reindex(columns=X_encoded.columns, 
                                               fill_value=0)

test_prediction = model.predict(test_data_encoded)
predictions = np.exp(test_prediction)

# OUTPUT
results = pd.DataFrame(
    dict(
        Id=test_data['Id'],
        SalePrice=predictions,
    )
)
results.to_csv("submission_XGB_vF.csv", index=False)