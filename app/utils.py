import cv2
import numpy as np
from tensorflow.keras.models import load_model

def process_image(image_path):
    """Use OpenCV and a trained model to detect waste in an image"""
    # Load a pre-trained model (would be trained separately)
    model = load_model('models/waste_detection.h5')
    
    # Preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img)
    classes = ['empty', 'half_full', 'full', 'overflowing']
    result = classes[np.argmax(prediction)]
    
    return {
        'status': result,
        'confidence': float(np.max(prediction))
    }

def send_sms_alert(phone, message):
    """Simulate sending SMS alerts to waste collectors"""
    # In a real app, integrate with an SMS API like Twilio
    print(f"Sending SMS to {phone}: {message}")
    return True
