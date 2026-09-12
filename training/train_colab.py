# ==============================================================================
# GOOGLE COLAB TRAINING SCRIPT FOR AI VS HUMAN STYLOMETRIC CLASSIFIER
# Run this entire script in a Google Colab GPU/CPU instance.
# ==============================================================================

import re
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class StylometryExtractor:
    def extract_features(self, text: str):
        words = re.findall(r'\w+', text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        total_words = len(words)
        total_sentences = len(sentences)
        
        if total_words == 0 or total_sentences == 0:
            return [0.0, 0.0, 0.0, 0.0]

        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = float(np.mean(sentence_lengths))
        sent_len_var = float(np.var(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0
        unique_words = set(words)
        ttr = float(len(unique_words) / total_words)
        punctuation_count = len(re.findall(r'[,;:\-"\']', text))
        punct_density = float(punctuation_count / total_words)

        return [avg_sent_len, sent_len_var, ttr, punct_density]

# 1. Synthetic Dataset Setup (Replace with actual Human vs AI datasets in Colab)
human_samples = [
    "I was walking down the street yesterday when suddenly it started raining. I didn't have an umbrella, so I ran into a local café!",
    "Honestly, the meeting could have been an email. We spent two hours discussing things that were already decided last week.",
    "The project timeline seems tight, but if we prioritize the core features, we can definitely make the deadline without much issue."
] * 100

ai_samples = [
    "In conclusion, the integration of advanced technologies plays a pivotal role in optimizing operational workflows and enhancing productivity.",
    "Furthermore, it is essential to consider the various factors that influence the overall performance of the proposed system architecture.",
    "To address this challenge effectively, a comprehensive analysis of the underlying mechanisms must be conducted prior to implementation."
] * 100

extractor = StylometryExtractor()
X_data, y_data = [], []

for text in human_samples:
    X_data.append(extractor.extract_features(text))
    y_data.append(0) # 0 = Human

for text in ai_samples:
    X_data.append(extractor.extract_features(text))
    y_data.append(1) # 1 = AI

X = np.array(X_data)
y = np.array(y_data)

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss')
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("=== Model Training Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

# 5. Export Model Artifact for Backend Integration
model.save_model("stylometry_xgb.json")
print("✅ Saved model to 'stylometry_xgb.json'. Download this file and place it in your local backend root!")
