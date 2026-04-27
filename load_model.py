from main import build_emitted_record_table, load_model, preprocess_dataset

pre, model = load_model("saved_models/model_20260427.joblib")

records = build_emitted_record_table("~/synth4bench/dataset.tsv", sample_rows=10)
dataset = preprocess_dataset(records)

X_transformed = pre.transform(dataset.X)
scores = model.predict_proba(X_transformed)[:, 1]

print(scores)
