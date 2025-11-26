import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Muat model yang sudah dilatih dengan penanganan error
try:
    model = load_model('stok_terpakai_model.h5')  # Pastikan model yang sudah dilatih berada di path yang benar
except Exception as e:
    st.error(f"Model tidak dapat dimuat: {e}")
    st.stop()

# Fungsi untuk memprediksi stok terpakai untuk n hari ke depan
def predict_stok(n_days, model, data, seq_length=7):
    # Menormalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['stok_terpakai']])

    # Ambil sequence terakhir untuk prediksi
    last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)

    predictions = []
    for i in range(n_days):
        next_day_pred = model.predict(last_sequence)
        predictions.append(next_day_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_day_pred[0, 0]

    # Transformasi kembali prediksi ke skala asli
    predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions_rescaled

# Antarmuka pengguna dengan Streamlit
st.title('Peramalan Stok Terpakai')
st.write('Aplikasi ini digunakan untuk memprediksi penggunaan stok dalam periode tertentu berdasarkan data yang sudah ada.')

# Membaca file CSV
try:
    df = pd.read_csv('dataset_kp.csv')  # Ganti dengan path file yang benar jika perlu
    df['tanggal'] = pd.to_datetime(df['tanggal'])
except Exception as e:
    st.error(f"File CSV tidak dapat dibaca: {e}")
    st.stop()

# Memilih data yang relevan dan mengurutkan berdasarkan tanggal
data = df[['tanggal', 'stok_terpakai']].sort_values('tanggal')

# Input untuk jumlah hari yang ingin diprediksi
n_days = st.number_input('Masukkan jumlah hari untuk diprediksi', min_value=1, max_value=30, value=0)

# Tombol untuk memulai prediksi
if st.button('Prediksi'):
    # Prediksi stok untuk n_days
    predictions_rescaled = predict_stok(n_days, model, data)

    # Tampilkan hasil prediksi
    st.subheader(f'Prediksi untuk {n_days} Hari Ke Depan:')
    predicted_dates = pd.date_range(start=data['tanggal'].max() + pd.Timedelta(days=1), periods=n_days, freq='D')
    predicted_df = pd.DataFrame({
        'Tanggal': predicted_dates,
        'Prediksi Stok Terpakai': predictions_rescaled.flatten()
    })
    st.write(predicted_df)

    # Plot hasil prediksi
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_df['Tanggal'], predicted_df['Prediksi Stok Terpakai'], label='Prediksi Stok Terpakai', color='orange')
    plt.xlabel('Tanggal')
    plt.ylabel('Stok Terpakai')
    plt.title(f'Prediksi Stok Terpakai untuk {n_days} Hari Ke Depan')
    plt.legend()
    st.pyplot()



