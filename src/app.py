from flask import Flask, request, jsonify
import joblib
import pandas as pd

MODEL_PATH = 'model_pipeline.pkl'
app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route('/health')
def health():
    return jsonify({'status':'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # data can be a single record or list
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    # model expects the original feature cols; assume user provides same col names
    preds = model.predict(df)
    probs = model.predict_proba(df)
    classes = model.classes_.tolist()
    results = []
    for p, pr in zip(preds, probs):
        results.append({'prediction': p, 'probabilities': dict(zip(classes, pr))})
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)