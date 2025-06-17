import streamlit as st
import pandas as pd
import numpy as np
import joblib


def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_model_and_encoders():
    model = joblib.load('model_adaboost.pkl')
    le_home = joblib.load('le_home.pkl')
    le_away = joblib.load('le_away.pkl')
    le_ftr = joblib.load('le_ftr.pkl')
    return model, le_home, le_away, le_ftr


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('data_laliga.csv')
    df.columns = df.columns.str.strip()
    return df


st.set_page_config(page_title="Prediksi La Liga", layout="wide")
local_css("style.css")

model, le_home, le_away, le_ftr = load_model_and_encoders()
df = load_data()

# Header dengan logo dan judul
from PIL import Image
laliga_logo = Image.open("LaLiga-Logo-PNG-Official-Symbol-for-Football-League-Transparent.png")
col1, col2 = st.columns([1, 5])
with col1:
    st.image(laliga_logo, width=200)
with col2:
    st.markdown("<h1 style='font-size:3rem; text-transform:uppercase; color:#74b9ff;'>Prediksi Hasil Pertandingan La Liga</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; max-width:720px; margin:0 auto 40px auto; color:#d0e7ff;'>"
    "Selamat datang di aplikasi prediksi hasil pertandingan La Liga! Pilih tim dan dapatkan prediksi hasilnya."
    "</p>",
    unsafe_allow_html=True,
)

# Inject style fixing dropdown text color in Streamlit
st.markdown("""
<style>
div[role="listbox"] > div {
    color: #0b2545 !important;
}
select {
    color: #0b2545 !important;  /* warna teks dropdown agar terlihat */
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        tim_home = st.selectbox("🔍 Pilih Tim Home", sorted(df['HomeTeam'].unique()))
    with col2:
        available_away = [team for team in sorted(df['AwayTeam'].unique()) if team != tim_home]
        tim_away = st.selectbox("🔍 Pilih Tim Away", available_away)
    st.markdown('</div>', unsafe_allow_html=True)

feature_columns = [
    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HTHG', 'HTAG'
]

historical_matches = df[(df['HomeTeam'] == tim_home) & (df['AwayTeam'] == tim_away)]

if len(historical_matches) == 0:
    feature_values = df[feature_columns].mean()
else:
    feature_values = historical_matches[feature_columns].mean()

feature_values = feature_values.fillna(0)

if st.button("🔮 Prediksi Hasil"):
    try:
        # Transform labels and predict
        home_enc = le_home.transform([tim_home])[0]
        away_enc = le_away.transform([tim_away])[0]
        input_vector = np.array([home_enc, away_enc] + feature_values.tolist()).reshape(1, -1)
        pred_encoded = model.predict(input_vector)
        pred_label = le_ftr.inverse_transform(pred_encoded)[0]

        # Show prediction result
        st.markdown(
            f"""
            <div class="result-container">
                <h2>Hasil Prediksi</h2>
                <p class="prediction">{pred_label}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hitung total fitur untuk Tim Home dan Tim Away di seluruh dataset 
        teamA_mask = (df['HomeTeam'] == tim_home) | (df['AwayTeam'] == tim_home)
        teamB_mask = (df['HomeTeam'] == tim_away) | (df['AwayTeam'] == tim_away)

        teamA_sums = df.loc[teamA_mask, feature_columns].sum()
        teamB_sums = df.loc[teamB_mask, feature_columns].sum()

        # Buat DataFrame untuk menampilkan perbandingan
        comparison_df = pd.DataFrame({
            'Fitur': feature_columns,
            f'Total {tim_home}': teamA_sums.values,
            f'Total {tim_away}': teamB_sums.values,
        })

        # Hitung persentase Team A dibandingkan Team B
        # Jika total Team B 0, beri nilai 0 untuk menghindari pembagian nol
        comparison_df['Persentase %'] = np.where(
            comparison_df[f'Total {tim_away}'] == 0,
            0,
            (comparison_df[f'Total {tim_home}'] / comparison_df[f'Total {tim_away}']) * 100
        ).round(2)

        st.subheader("Perbandingan Jumlah Fitur Antara Tim")
        st.write(f"Perbandingan total nilai fitur dari tim {tim_home} terhadap tim {tim_away} dalam persentase:")
        st.table(comparison_df.set_index('Fitur'))

    except Exception as e:
        st.error(f"❌ Kesalahan saat prediksi: {e}")

st.markdown("---")
st.markdown("<footer>Tugas Akir UAS [Kelompok 7] | 2025</footer>", unsafe_allow_html=True)

