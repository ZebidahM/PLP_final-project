from flask import Flask, render_template, request, jsonify
from app.models import WastePredictor, RouteOptimizer
from app.utils import process_image
import os

app = Flask(__name__)

# Initialize models
predictor = WastePredictor('models/waste_model.pkl')
optimizer = RouteOptimizer()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = predictor.predict(
        data['area'], 
        data['day_of_week'], 
        data['weather']
    )
    return jsonify({'prediction': prediction})

@app.route('/optimize-route', methods=['POST'])
def optimize_route():
    bins = request.json['bins']
    optimized_route = optimizer.optimize(bins)
    return jsonify({'route': optimized_route})

@app.route('/detect-waste', methods=['POST'])
def detect_waste():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    image_file = request.files['image']
    image_path = os.path.join('static/uploads', image_file.filename)
    image_file.save(image_path)
    
    result = process_image(image_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
