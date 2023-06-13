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

# Load dataset
df = pd.read_csv('./tes.csv')

# Mengubah kolom tipe menjadi angka menggunakan LabelEncoder
label_encoder = LabelEncoder()
df['tipe_encoded'] = label_encoder.fit_transform(df['Tipe'])

# Melakukan preprocessing pada dataset
onehot_encoder = OneHotEncoder(sparse=False)
tipe_encoded = df['tipe_encoded'].values.reshape(-1, 1)
tipe_encoded = onehot_encoder.fit_transform(tipe_encoded)
tipe_encoded_df = pd.DataFrame(tipe_encoded, columns=['tipe_' + str(i) for i in range(tipe_encoded.shape[1])])
df = pd.concat([df, tipe_encoded_df], axis=1)

# Memisahkan fitur dan label
X = df.drop(['foodId', 'Nama', 'Tipe', 'Tipe 1', 'Tipe 2', 'tipe_encoded'], axis=1)
y = df['tipe_encoded']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y.nunique(), activation='softmax')
])

# Compile model dengan optimizer 'adam', loss function 'sparse_categorical_crossentropy', dan metrics ['accuracy']
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Melatih model dengan model.fit dengan menggunakan X_train, y_train, X_test, y_test,  dan menampilkan akurasi dan loss
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Menyimpan model ke dalam format TF-Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Menyimpan model ke dalam file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load model TF-Lite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    try:
        if 'food_id' not in request.form or 'food_name' not in request.form:
            return "Please provide food_id and food_name"

        food_id = request.form.get('food_id')
        food_name = request.form.get('food_name')

        # Preprocess the input features
        input_features = pd.DataFrame({
            'foodId': [food_id],
            'Nama': [food_name],
            'Tipe': ['Unknown']
        })

        # Handle previously unseen labels
        input_features['tipe_encoded'] = label_encoder.transform(input_features['Tipe'].fillna('Unknown'))

        # Check if 'Unknown' label is present in the label encoder's classes
        if 'Unknown' not in label_encoder.classes_:
            label_encoder.classes_ = np.concatenate([label_encoder.classes_, ['Unknown']])

        # Encode the input features
        input_features_encoded = onehot_encoder.transform(input_features['tipe_encoded'].values.reshape(-1, 1))


        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_features_encoded.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # Decode the predicted food type
        predicted_food_type = label_encoder.inverse_transform([np.argmax(output)])[0]

        # Get recommended foods of the predicted type
        recommended_foods = df[df['Tipe'] == predicted_food_type]['Nama'].tolist()

        # Remove the input food from the recommended list
        recommended_foods = [food for food in recommended_foods if food != food_name]

        # Return the top recommendations
        top_recommendations = recommended_foods[:10]

        return jsonify({'top_recommendations': top_recommendations})
    except Exception as e:
        # Log the error message
        import logging
        logging.basicConfig(filename='error.log', level=logging.ERROR)
        logging.error(str(e))
        return "An error occurred"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
