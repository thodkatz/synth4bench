from main import CODE_TO_CLASS, build_record_table, load_model, preprocess_dataset

pre, model = load_model("saved_models/model_20260427.joblib")

records = build_record_table("./datasets/dataset.tsv", sample_rows=10)
dataset = preprocess_dataset(records)

X_transformed = pre.transform(dataset.X)
scores = model.predict_proba(X_transformed)
predictions = [CODE_TO_CLASS[code] for code in scores.argmax(axis=1)]

print(scores)
print(predictions)
