import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

# Load and balance dataset (same logic as before)
df = pd.read_csv("../datasets/finalized_dataset.csv")
df.drop(columns=["ID", "Source"], inplace=True)

df_dark = df[df["Category (Safe/Dark)"] == "Dark"]
df_safe = df[df["Category (Safe/Dark)"] == "Safe"]
df_safe_oversample = df_safe.sample(df_dark.shape[0], replace=True)
df_balanced = pd.concat([df_safe_oversample, df_dark], axis=0)

df_balanced["Dark"] = df_balanced["Category (Safe/Dark)"].apply(lambda x: 0 if x == "Safe" else 1)
texts = df_balanced["Policy"].values
labels = df_balanced["Dark"].values

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)

# K-Fold with Random Forest
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in kfold.split(X, labels):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    fold += 1

# Save final model and vectorizer
joblib.dump(rf_model, "../models/random_forest_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
print("TF-IDF + Random Forest model and vectorizer saved.")
