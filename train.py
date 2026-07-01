import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np
df = pd.read_csv("data/insurance.csv")
X=df.drop("charges",axis=1)
y=df["charges"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
categorical_features = [
    "sex",
    "smoker",
    "region"
]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first"),
            categorical_features
        )
    ],
    remainder="passthrough"
)
model = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="log2",
    max_depth=None,
    random_state=42
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)
print("Final Model Performance")
print("-" * 30)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
joblib.dump(
    pipeline,
    "model/model.pkl"
)
print("\nModel saved successfully!")