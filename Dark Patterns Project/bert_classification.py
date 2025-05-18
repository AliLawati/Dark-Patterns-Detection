import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

loaded_model = tf.keras.models.load_model("models/enhanced_bert_model.keras", custom_objects={'KerasLayer': hub.KerasLayer})

def classify_with_bert(policy_text):
    predictions = loaded_model.predict([policy_text], verbose=0)

    score = float(predictions[0][0])
    label = int(score > 0.5)

    if label == 0:
        return "Safe"
    return "Dark"
