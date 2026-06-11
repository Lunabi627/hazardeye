from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('road_safety_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = pd.DataFrame([[
        data['month'], data['hour'],
        str(data['road_class']), str(data['zone_type']),
        str(data['road_feature']), str(data['road_condition']),
        str(data['weather']), str(data['light_condition']),
        str(data['vehicle_type']), str(data['cause_of_accident']),
        data['speed'], data['speed_limit'],
        str(data['overspeeding']), data['persons_injured'],
        str(data['district']), str(data['day_of_week'])
    ]], columns=[
        'month', 'hour_of_day', 'road_class', 'zone_type',
        'road_feature', 'road_condition', 'weather', 'light_condition',
        'vehicle_type', 'cause_of_accident', 'speed_kmh', 'speed_limit_kmh',
        'overspeeding', 'persons_injured', 'district', 'day_of_week'
    ])
    prediction = model.predict(features)
    result = "HIGH" if prediction[0] == 1 else "LOW"
    return jsonify({'severity': result})

if __name__ == '__main__':
    app.run()