import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("../datasets/finalized_dataset.csv")
df.drop(columns=["ID", "Source"], inplace=True)

df_dark = df[df["Category (Safe/Dark)"] == "Dark"]
df_safe = df[df["Category (Safe/Dark)"] == "Safe"]

df_safe_oversample = df_safe.sample(df_dark.shape[0], replace=True)
df_balanced = pd.concat([df_safe_oversample, df_dark], axis=0)

df_balanced["Dark"] = df_balanced["Category (Safe/Dark)"].apply(lambda x: 0 if x == "Safe" else 1)

x_train, x_test, y_train, y_test = train_test_split(
    df_balanced["Policy"],
    df_balanced["Dark"],
    stratify=df_balanced["Dark"]
)

bert_preprocessor = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
bert_encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-12-h-768-a-12/4")

policy_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="policy")
preprocessed_policy = bert_preprocessor(policy_input)
output_policy = bert_encoder(preprocessed_policy)

layer = tf.keras.layers.Dropout(0.1, name="dropout1")(output_policy["pooled_output"])
layer = tf.keras.layers.Dense(128, activation="swish", name="hidden_swish")(layer)
layer = tf.keras.layers.Dropout(0.1, name="dropout2")(layer)

output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(layer)
bert_model = tf.keras.Model(inputs=[policy_input], outputs=[output])

Metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall")
]

bert_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=Metrics
)

print("Training Values:")
bert_model.fit(x_train, y_train, epochs=500)

print("\nTesting Values:")
bert_model.evaluate(x_test, y_test)

print("\nPredicting Y:")
y_predicted = bert_model.predict(x_test)
y_predicted = y_predicted.flatten()
y_predicted = np.where(y_predicted > 0.5, 1, 0)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_predicted)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_predicted))

print("\nSaving Model:")
bert_model.save('../models/enhanced_bert_model.keras')
print("Model Saved Successfully")
