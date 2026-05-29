# ml_engine.py
import pandas as pd
import tensorflow as tf
import joblib

def load_ai_model():
    """Завантажує модель та скейлер у кеш"""
    model = tf.keras.models.load_model('crop_mlp_model.h5')
    scaler = joblib.load('scaler.pkl')
    classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 
               'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 
               'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 
               'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
    return model, scaler, classes

def predict_crops(model, scaler, n, p, k, temp, hum, ph, rain):
    """Робить прогноз на основі даних"""
    input_vector = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], 
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    input_scaled = scaler.transform(input_vector)
    probs = model.predict(input_scaled, verbose=0)[0]
    return probs