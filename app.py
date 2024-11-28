from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('autism_ensemble_model.pkl')

selected_features = ['A1', 'A3', 'A5', 'A7', 'A9'] 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        form_data = request.json
        input_data = [float(form_data[feature]) for feature in selected_features]  # Extract only selected features
        
        input_array = np.array(input_data).reshape(1, -1)
        
        prediction = model.predict(input_array)
        print(prediction)
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == '__main__':
    app.run(debug=True)
