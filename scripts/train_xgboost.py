import os
import json
import numpy as np
import xgboost as xgb

# Synthetic training dataset representing Human (0) vs AI (1) stylometric metrics
# Features: [TTR, Avg Sentence Length, Burstiness, Punctuation Density]
X_human = np.array([
    [0.65, 14.2, 12.5, 0.12],
    [0.71, 11.0, 15.1, 0.10],
    [0.58, 28.5, 32.1, 0.14],
    [0.62, 22.1, 24.8, 0.15],
    [0.54, 30.2, 35.4, 0.13]
])
y_human = np.zeros(len(X_human))

X_ai = np.array([
    [0.45, 18.2, 3.1, 0.08],
    [0.48, 17.8, 2.4, 0.07],
    [0.42, 19.5, 4.0, 0.09],
    [0.50, 16.9, 2.1, 0.08],
    [0.46, 18.0, 3.5, 0.07]
])
y_ai = np.ones(len(X_ai))

X = np.vstack([X_human, X_ai])
y = np.hstack([y_human, y_ai])

dtrain = xgb.DMatrix(X, label=y)
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1
}

bst = xgb.train(params, dtrain, num_boost_round=50)
bst.save_model("models/stylometry_xgb.json")
print("Successfully trained and saved model to models/stylometry_xgb.json")
