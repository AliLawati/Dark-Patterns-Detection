import joblib

tfidf = joblib.load("models/tfidf_vectorizer_5.pkl")
rf = joblib.load("models/random_forest_model_5.pkl")

def classify_with_tf_idf(policy_text):
	X_tfidf = tfidf.transform([policy_text])
	rf_preds = rf.predict(X_tfidf)

	rf_label_map = {0: "Safe", 1: "Dark"}
	rf_pred_label = [rf_label_map[pred] for pred in rf_preds][0]

	return rf_pred_label