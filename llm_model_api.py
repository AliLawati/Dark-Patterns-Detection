import json
import asyncio
import os
import re
from starlette.responses import StreamingResponse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from bert_classification import classify_with_bert
from tfidf_classification import classify_with_tf_idf

app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

def clean_policies(policies):
	return [policy.strip() for policy in policies if policy.strip()]

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

		return result["response"].strip()

# Queue-based policy processing
async def process_policy_stream(policies, classification_model, model="mistral", brief=True):
	async def process_single_policy(policy):
		result = classification_model(policy)
		prompt = (
			f"Explain clearly in 1 line"
			if brief else
			f"Explain clearly in detail"
		)
		prompt += (
			f" why the following privacy policy is classified as '{result}' only based on the context of the policy itself:\n{policy}\n\n"
			f"Start your answer in the format: 'This policy is considered '{result}' because...'."
		)

		# Fetch response from Ollama
		response_text = await call_ollama(prompt, model)

		# Remove <think> tags and clean the result
		response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

		# Yield the policy and the explanation as a JSON string
		return json.dumps({
			"policy": policy,
			"explanation": response_text.strip(),
			"classification": result
		}) + "\n\n\n"

	tasks = [process_single_policy(policy) for policy in policies]

	for coro in asyncio.as_completed(tasks):
		result = await coro
		yield result

@app.post("/brief")
async def explain_policies_classification_briefly(request: Request):
	received_data = await request.json()
	policies_text = received_data["text"]
	policies = clean_policies(
		policies_text
		.split("\n")
	)

	return StreamingResponse(process_policy_stream(policies, classify_with_tf_idf, brief=True),
							 media_type="application/json")

@app.post("/detailed")
async def explain_policies_classification_in_detail(request: Request):
	received_data = await request.json()
	policies_text = received_data["text"]
	policies = clean_policies(
		policies_text
		.split("\n")
	)

	return StreamingResponse(process_policy_stream(policies, classify_with_tf_idf, brief=False),
							 media_type="application/json")

if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="localhost", port=8000)
