import flask
import io
import pandas as pd
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request

# Load model pengenalan gambar ResNet50
model = tf.keras.models.load_model('my_model.h5')

app = Flask(__name__)

# def recommend_food(food_id, food_name):
#     # Implementasikan logika untuk merekomendasikan makanan berdasarkan food_id dan food_name
#     # Return daftar rekomendasi makanan teratas

#     # Contoh implementasi sederhana:
#     recommendations = ['3423', 'Burger Australia']
#     return recommendations

def recommend_food(model, data, food_id, food_name, num_recommendations=10):
    # Find the row in the data based on the food_id
    food_row = data[data['foodId'] == food_id]
    
    # Create a DataFrame with the input features
    input_features = pd.DataFrame({
        'foodId': [food_id],
        'Nama': [food_name],
        'Tipe': [food_row['Tipe'].values[0]]
    })
    type=f
    # Preprocess the input features
    input_features['tipe_encoded'] = label_encoder.transform(input_features['Tipe'])
    input_features_encoded = onehot_encoder.transform(input_features['tipe_encoded'].values.reshape(-1, 1))
    
    # Predict the food type
    prediction = model.predict(input_features_encoded)
    
    # Decode the predicted food type
    predicted_food_type = label_encoder.inverse_transform([prediction.argmax()])[0]
    
    # Get recommended foods of the predicted type
    recommended_foods = data[data['Tipe'] == predicted_food_type]['Nama'].tolist()
    
    # Remove the input food from the recommended list
    recommended_foods = [food for food in recommended_foods if food != food_name]
    
    # Return the top recommendations
    top_recommendations = recommended_foods[:num_recommendations]
    
    return top_recommendations

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    if 'food_id' not in request.form or 'food_name' not in request.form:
        return "Please provide food_id and food_name"
    
    food_id = request.form.get('food_id')
    food_name = request.form.get('food_name')
    df=pd.read_csv('tes.csv')
    top_recommendations = recommend_food(model=model, data=df, food_id=food_id, food_name=food_name)

    return jsonify({'top_recommendations': top_recommendations})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
