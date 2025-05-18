import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

# Load and prepare dataset
df = pd.read_csv("../datasets/finalized_dataset.csv")
df.drop(columns=["ID", "Source"], inplace=True)

df_dark = df[df["Category (Safe/Dark)"] == "Dark"]
df_safe = df[df["Category (Safe/Dark)"] == "Safe"]
df_safe_oversample = df_safe.sample(df_dark.shape[0], replace=True)
df_balanced = pd.concat([df_safe_oversample, df_dark], axis=0)

df_balanced["Dark"] = df_balanced["Category (Safe/Dark)"].apply(lambda x: 0 if x == "Safe" else 1)
texts = df_balanced["Policy"].values
labels = df_balanced["Dark"].values

# Load BERT layers
bert_preprocessor = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
bert_encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-12-h-768-a-12/4")

# Model builder function
def build_model():
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="policy")
    
    # Preprocess input text using BERT preprocessor
    preprocessed = bert_preprocessor(input_layer)
    
    # Use BERT encoder
    encoded = bert_encoder(preprocessed)
    
    # Use pooled output from the encoder for downstream classification
    x = tf.keras.layers.Dropout(0.1)(encoded["pooled_output"])
    x = tf.keras.layers.Dense(128, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Build and compile the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

# K-Fold training
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in kfold.split(texts, labels):
	print(f"\n--- Fold {fold} ---")
	x_train, x_test = texts[train_index], texts[test_index]
	y_train, y_test = labels[train_index], labels[test_index]

	model = build_model()
	model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1)

	print("\nEvaluating:")
	model.evaluate(x_test, y_test)

	y_pred = model.predict(x_test).flatten()
	y_pred_binary = np.where(y_pred > 0.5, 1, 0)

	print("\nConfusion Matrix:")
	print(confusion_matrix(y_test, y_pred_binary))

	print("\nClassification Report:")
	print(classification_report(y_test, y_pred_binary))

	fold += 1

# Optionally save the last model
model.save("../models/kfold_model_final.keras")
print("Final model saved after last fold.")
