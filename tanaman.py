
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Crop Yield Prediction App")

# Load the trained pipeline (model and preprocessor)
try:
    with open('model.pkl', 'rb') as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

st.title('🌱 Prediksi Hasil Panen (Crop Yield Prediction)')
st.write('Aplikasi ini memprediksi hasil panen berdasarkan fitur input.')

# --- Input Form ---
st.header('Masukkan Detail Pertanian')

# Get unique values for dropdowns from the original DataFrame (or define them manually)
# For a real deployment, these lists should be derived from your training data or known categories.
# Since `df` is not available directly in app.py context, we define them here.
# In a more robust app, these lists would be saved alongside the model or hardcoded if static.

# These lists are based on the unique values observed in the original notebook's df.unique() outputs.
crops = [
    'Arecanut (Pinang)', 'Arhar/Tur (Kacang Gude)', 'Castor seed (Biji Jarak)', 'Coconut (Kelapa)', 'Cotton(lint) (Kapas)',
    'Dry chillies (Cabai Kering)', 'Gram (Buncis)', 'Jute (Rami)', 'Linseed (Biji Rami)', 'Maize (Jagung)',
    'Mesta (Mesta)', 'Niger seed (Biji Niger)', 'Onion (Bawang Merah)', 'Other Rabi pulses (Kacang-kacangan Rabi Lainnya)',
    'Potato (Kentang)', 'Rapeseed &Mustard (Rapeseed & Sawi)', 'Rice (Padi)', 'Sesamum (Wijen)',
    'Small millets (Jelai Kecil)', 'Sugarcane (Tebu)', 'Sweet potato (Ubi Jalar)', 'Tapioca (Singkong)',
    'Tobacco (Tembakau)', 'Turmeric (Kunyit)', 'Wheat (Gandum)', 'Bajra (Milet Mutiara)', 'Black pepper (Lada Hitam)',
    'Cardamom (Kapulaga)', 'Coriander (Ketumbar)', 'Garlic (Bawang Putih)', 'Ginger (Jahe)',
    'Groundnut (Kacang Tanah)', 'Horse-gram (Kacang Kuda)', 'Jowar (Jawawut)', 'Ragi (Jelai)',
    'Cashewnut (Mete)', 'Banana (Pisang)', 'Soyabean (Kedelai)', 'Barley (Jelai)',
    'Khesari (Kacang Khesari)', 'Masoor (Lentil)', 'Moong(Green Gram) (Kacang Hijau)',
    'Other Kharif pulses (Kacang-kacangan Kharif Lainnya)', 'Safflower (Kesumba)', 'Sannhamp (Rami San)',
    'Sunflower (Bunga Matahari)', 'Urad (Kacang Urad)', 'Peas & beans (Pulses) (Kacang Polong & Kacang-kacangan)',
    'other oilseeds (Biji Minyak Lainnya)', 'Other Cereals (Sereal Lainnya)', 'Cowpea(Lobia) (Kacang Tunggak)',
    'Oilseeds total (Total Biji Minyak)', 'Guar seed (Biji Guar)', 'Other Summer Pulses (Kacang-kacangan Musim Panas Lainnya)',
    'Moth (Kacang Moth)'
]

seasons = [
    'Whole Year (Sepanjang Tahun)','Autumn (Musim Gugur)', 'Summer (Musim Panas)', 'Winter (Musim Dingin)'
]

with st.form('prediction_form'):
    col1, col2 = st.columns(2)
    with col1:
        crop_display = st.selectbox('Crop (Jenis Tanaman)', options=crops)
        year = st.slider('Year (Tahun)', min_value=1997, max_value=2025, value=2023)
        season_display = st.selectbox('Season (Musim)', options=seasons)
    with col2:
        area = st.number_input('Area (Luas Lahan Hektar)', min_value=0.0, max_value=1000000.0, value=1000.0)
        production = st.number_input('Production (Produksi Ton)', min_value=0.0, max_value=1000000.0, value=2000.0)
        fertilizer = st.number_input('Fertilizer (Pupuk Kg)', min_value=0.0, max_value=100000000.0, value=500.0)
        pesticide = st.number_input('Pesticide (Pestisida Kg)', min_value=0.0, max_value=1000000.0, value=100.0)

    submitted = st.form_submit_button('Prediksi Hasil Panen')

    if submitted:
        # Extract original English values for prediction
        crop_selected = crop_display.split(' (')[0]
        season_selected = season_display.split(' (')[0]

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([{
            'crop': crop_selected,
            'year': year,
            'season': season_selected,
            'area': area,
            'production': production,
            'fertilizer': fertilizer,
            'pesticide': pesticide
        }])

        try:
            prediction = pipeline.predict(input_data)[0]
            st.success(f"Prediksi Hasil Panen: **{prediction:.2f} (satuan hasil panen)**")
            st.info("Catatan: Hasil prediksi adalah nilai `yield` yang telah diproses. Interpretasi satuan bergantung pada dataset asli.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
            st.write("Harap periksa kembali input Anda dan pastikan file model telah dimuat dengan benar.")