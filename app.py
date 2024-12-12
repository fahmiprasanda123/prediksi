import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Path ke file Excel
file_path = 'gabungan.xlsx'  # Ganti dengan path file Excel Anda

# Membaca dan memproses data
def load_data(sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print("Kolom yang ditemukan dalam DataFrame:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None

# Normalisasi dan persiapan data
def prepare_data(df):
    if df is None or df.empty:
        raise ValueError("DataFrame kosong atau tidak valid.")

    df.drop(columns=['No'], inplace=True, errors='ignore')  # Hapus kolom 'No' jika ada
    df = df.melt(id_vars=['Komoditas (Rp)'], var_name='Tanggal', value_name='Harga')  # Pastikan 'Komoditas (Rp)' ada di id_vars
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df['Harga'] = df['Harga'].str.replace(',', '').astype(float)
    df.dropna(subset=['Tanggal', 'Harga'], inplace=True)

    if df.empty:
        raise ValueError("DataFrame kosong setelah menghapus NaN.")

    df.set_index('Tanggal', inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Harga']])
    return scaled_data, scaler, df.index

# Membangun model RNN
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Memuat data dan melatih model
df = load_data('beras')  # Ganti dengan sheet default jika diperlukan
if df is not None:
    scaled_data, scaler, dates = prepare_data(df)

    look_back = 3
    X, Y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        Y.append(scaled_data[i + look_back])

    X, Y = np.array(X), np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=0)

# Fungsi untuk memprediksi beberapa langkah ke depan
def predict_future(model, data, scaler, steps):
    predictions = []
    current_data = data[-look_back:]
    for _ in range(steps):
        current_data_reshaped = np.reshape(current_data, (1, look_back, data.shape[1]))
        predicted_scaled = model.predict(current_data_reshaped)
        predicted = scaler.inverse_transform(predicted_scaled)
        predictions.append(predicted[0])
        current_data = np.append(current_data[1:], predicted_scaled, axis=0)
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        end_date = pd.to_datetime(data['end_date'], format='%d/%m/%Y', errors='coerce')
        start_date = pd.to_datetime(data['start_date'], format='%d/%m/%Y', errors='coerce')
        sheet_name = data.get('sheet_name')  # Ambil sheet_name dari permintaan
        province = data.get('province')  # Ambil province dari permintaan

        if pd.isna(start_date) or pd.isna(end_date) or not sheet_name or not province:
            return jsonify({'error': 'Invalid date format or missing parameters'}), 400

        # Memuat data dari sheet yang ditentukan
        df = load_data(sheet_name)
        scaled_data, scaler, dates = prepare_data(df)

        # Filter data berdasarkan provinsi
        df_province = df[df['Komoditas (Rp)'] == province]
        if df_province.empty:
            return jsonify({'error': 'Province not found'}), 404

        # Hitung jumlah langkah berdasarkan end_date
        steps = (end_date - start_date).days

        last_data = scaled_data[-look_back:]
        predictions_future = predict_future(model, last_data, scaler, steps)

        # Konversi prediksi ke tipe float biasa
        predictions_future = [float(price) for price in predictions_future]

        future_dates = [start_date + pd.Timedelta(days=i) for i in range(steps)]
        future_prices = {date.strftime('%d/%m/%Y'): price for date, price in zip(future_dates, predictions_future)}

        return jsonify(future_prices)

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON'}), 400
    except KeyError as e:
        return jsonify({'error': f'Missing key in JSON: {e}'}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running!"})

if __name__ == '__main__':
    app.run(debug=False)  # Nonaktifkan debug mode
