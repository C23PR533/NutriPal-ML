from ast import For
import flask
import io
import pandas as pd
import string
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from flask import Flask, jsonify, request

# Load model
model = tf.keras.models.load_model('./my_model.h5')

app = Flask(__name__)

# Definisi variabel global
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

# def recommend_food(food_id, food_name):
#     # Implementasikan logika untuk merekomendasikan makanan berdasarkan food_id dan food_name
#     # Return daftar rekomendasi makanan teratas

#     # Contoh implementasi sederhana:
#     recommendations = ['3423', 'Burger Australia']
#     return recommendations

def preprocess_data(data):
    # Mengubah kolom tipe menjadi angka menggunakan LabelEncoder
    label_encoder = LabelEncoder()
    data['tipe_encoded'] = label_encoder.fit_transform(data['Tipe'])

    # Melakukan preprocessing pada dataset
    onehot_encoder = OneHotEncoder(sparse=False)
    tipe_encoded = data['tipe_encoded'].values.reshape(-1, 1)
    tipe_encoded = onehot_encoder.fit_transform(tipe_encoded)
    tipe_encoded_df = pd.DataFrame(tipe_encoded, columns=['tipe_' + str(i) for i in range(tipe_encoded.shape[1])])
    data = pd.concat([data, tipe_encoded_df], axis=1)

    # Memisahkan fitur dan label
    X = data.drop(['foodId', 'Nama', 'Tipe', 'Tipe 1', 'Tipe 2', 'tipe_encoded'], axis=1)
    y = data['tipe_encoded']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def recommend_food(model, data, food_id, food_name, num_recommendations=10):
    # Find the row in the data based on the food_id
    food_row = data[data['foodId'] == food_id]
    
    # Create a DataFrame with the input features
    input_features = pd.DataFrame({
        'foodId': [food_id],
        'Nama': [food_name],
        'Tipe': [food_row['Tipe'].values[0]]
    })
    type=For
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
    try:
        if 'food_id' not in request.form or 'food_name' not in request.form:
            return "Please provide food_id and food_name"

        food_id = request.form.get('food_id')
        food_name = request.form.get('food_name')
        df = pd.read_csv('./tes.csv')
        top_recommendations = recommend_food(model=model, data=df, food_id=food_id, food_name=food_name)

        return jsonify({'top_recommendations': top_recommendations})
    except Exception as e:
        # Log the error message
        import traceback
        traceback.print_exc()
        return "An error occurred"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')