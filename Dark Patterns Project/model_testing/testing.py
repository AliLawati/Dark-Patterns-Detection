import time

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Load Dataset
df = pd.read_csv("../datasets/finalized_dataset.csv")
df.drop(columns=["ID", "Source"], inplace=True)

texts = df['Policy'].astype(str).tolist()
labels = df['Category (Safe/Dark)'].tolist()

true_labels = [label for label in labels]

"""
Testing bert model
"""

def test_bert():
	import tensorflow as tf
	import tensorflow_hub as hub
	import tensorflow_text as text

	tf.get_logger().setLevel('ERROR')

	### 1. BERT MODEL ###
	print("Evaluating BERT model...")
	bert_model = tf.keras.models.load_model(
		"../models/kfold_bert.keras",
		custom_objects={"KerasLayer": hub.KerasLayer}
	)

	# Load preprocessing model from TF Hub — adjust URL based on what was used in training
	preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
	bert_preprocess_model = hub.load(preprocess_url)

	# Preprocess text
	def preprocess_bert_inputs(texts):
		# Pass the entire list of texts at once (batch prediction)
		predictions = bert_model.predict(texts, verbose=0)

		# Assuming output shape is (n_samples, 1), where prediction > 0.5 is 'Dark'
		labels = ["Dark" if score > 0.5 else "Safe" for score in predictions.flatten()]

		return labels

	# Predict
	bert_pred_labels = preprocess_bert_inputs(texts)

	# Evaluate
	print("BERT Accuracy:", accuracy_score(true_labels, bert_pred_labels))
	print(classification_report(true_labels, bert_pred_labels))

"""
Testing the tfidf model with random forest
"""

def test_tfidf():
	import joblib

	start_time = time.time()

	### 2. TF-IDF + RANDOM FOREST ###
	print("\nEvaluating TF-IDF + Random Forest...")
	tfidf = joblib.load('../models/tfidf_vectorizer_5.pkl')
	rf = joblib.load('../models/random_forest_model_5.pkl')

	X_tfidf = tfidf.transform(texts)
	rf_preds = rf.predict(X_tfidf)

	rf_label_map = {0: "Safe", 1: "Dark"}
	rf_pred_labels = [rf_label_map[pred] for pred in rf_preds]

	print("Random Forest Accuracy:", accuracy_score(true_labels, rf_pred_labels))
	print(classification_report(true_labels, rf_pred_labels))
	print(f"Took: {time.time() - start_time} seconds to complete")

"""
Testing the llama models, Mistral and Deepseek-R1
"""

def test_llama_models():
	import httpx
	import re
	import asyncio

	### 3. OLLAMA MODELS (Mistral and DeepSeek) ###
	async def call_ollama(prompt, model):
		async with httpx.AsyncClient() as client:
			response = await client.post(
				"http://localhost:11434/api/generate",
				json={
					"model": model,
					"prompt": prompt,
					"stream": False
				},
				timeout=60.0
			)
			response.raise_for_status()
			result = response.json()
			return re.sub(r'<think>.*?</think>', '', result["response"].strip(), flags=re.DOTALL)

	async def run_async_ollama_evaluation(model_name, ids, policies):
		completed = 0
		total = len(ids)
		results = {}
		start_time = time.time()
		sem = asyncio.Semaphore(3)

		async def limited_task(task):
			async with sem:
				return await task

		async def process_single_policy(idx, policy):
			nonlocal completed
			prompt = f"Classify the following policy as either 'Safe' or 'Dark':\n\"{policy}\"\n\nAnswer with only 'Safe' or 'Dark'."
			try:
				classification = await call_ollama(prompt, model_name)
				if "safe" in classification.lower():
					label = "Safe"
				elif "dark" in classification.lower():
					label = "Dark"
				else:
					label = "Unknown"
			except Exception as e:
				print(f"[{model_name}] Error on ID {idx}: {e}")
				label = "Unknown"
			results[idx] = label
			completed += 1

		async def progress_reporter():
			while completed < total:
				await asyncio.sleep(10)
				percent = completed / total * 100
				print(f"[{model_name}] Progress: {completed}/{total} ({percent:.2f}%)")

		tasks = [limited_task(process_single_policy(idx, text)) for idx, text in zip(ids, policies)]
		await asyncio.gather(progress_reporter(), *tasks)

		print(f"[{model_name}] Completed all {total} classifications in {int(time.time() - start_time)}s")
		return results

	def evaluate_ollama_model(model_name, ids, texts, true_labels):
		print(f"\nEvaluating Ollama model: {model_name}")

		predictions_dict = asyncio.run(run_async_ollama_evaluation(model_name, ids, texts))
		pred_labels = [predictions_dict.get(idx, "Unknown") for idx in ids]

		print(f"{model_name} Accuracy:", accuracy_score(true_labels, pred_labels))
		print(classification_report(true_labels, pred_labels))

	# Run evaluations for Mistral and DeepSeek
	ids = df.index.tolist()

	evaluate_ollama_model("Mistral", ids, texts, true_labels)
	evaluate_ollama_model("deepseek-r1:7b", ids, texts, true_labels)

if __name__ == "__main__":
	test_bert()
	test_tfidf()
	test_llama_models()
