# Run this inside Google Colab with GPU enabled
# !pip install xgboost spacy numpy pandas
# !python -m spacy download en_core_web_sm

import numpy as np
import xgboost as xgb
import json

# Multi-class target map: 0 = Human, 1 = GPT-4, 2 = Claude-3.5
# Features: [TTR, Avg Sentence Length, Burstiness, Punctuation Density]

X_train = np.array([
    # Human Samples
    [0.65, 24.2, 18.5, 0.12],
    [0.71, 21.0, 22.1, 0.10],
    # GPT-4 Samples (Moderate length, low burstiness)
    [0.48, 17.8, 2.4, 0.07],
    [0.50, 16.9, 2.1, 0.08],
    # Claude Samples (Longer sentences, high TTR, medium burstiness)
    [0.58, 26.5, 8.2, 0.11],
    [0.61, 28.1, 9.0, 0.13]
])

y_train = np.array([0, 0, 1, 1, 2, 2])

dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 4,
    "eta": 0.1
}

model = xgb.train(params, dtrain, num_boost_round=100)
model.save_model("stylometry_multiclass_xgb.json")
print("Multi-class LLM fingerprint model trained successfully!")
