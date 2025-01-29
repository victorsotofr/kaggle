import numpy as np
import pandas as pd

from utils import get_train_data

from FeatureEngineering import _encode

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ==============================================================================
# DATA
X, y = get_train_data()

# ==============================================================================
# FEATURE ENGINEERING & MODEL FIT
X_encoded, y, preprocessor = _encode(X, y=y)

pipeline = make_pipeline(
    # StandardScaler(),
    XGBClassifier(
        eta = 0.13931600539184902,
        gamma = 1.6973396780852883,
        max_depth = 4,
        min_child_weight = 4,
        subsample = 0.5885034417642654,
        colsample_bytree = 0.9034020022420274,
        seed = 42,
    )
)

pipeline.fit(X_encoded, y)

# ==============================================================================
# PREDICTION
test_data = pd.read_csv('data/test.csv')
test_data_encoded = _encode(test_data, preprocessor=preprocessor).reindex(columns=X_encoded.columns, 
                                                fill_value=0)

predictions = pipeline.predict(test_data_encoded)

# ==============================================================================
# OUTPUT
results = pd.DataFrame(
    dict(
        PassengerId=test_data['PassengerId'],
        Survived=predictions,
    )
)

results.to_csv('submission_XGB_vF_woScaling.csv', index=False)