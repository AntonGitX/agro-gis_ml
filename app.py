# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np

# Імпортуємо НАШІ нові модулі
from ml_engine import load_ai_model, predict_crops
from gis_engine import analyze_polygon

st.set_page_config(page_title="Agro GIS AI", layout="wide")
st.title("🌱 Гібридна ГІС-система планування агровиробництва")

# Завантаження моделі
model, scaler, crop_classes = load_ai_model()
row_crops = ['maize', 'cotton', 'jute', 'watermelon', 'muskmelon', 'papaya']

# Інтерфейс
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🗺️ Крок 1. Оберіть поле на карті")
    m = folium.Map(location=[49.42, 26.98], zoom_start=11, tiles="CartoDB positron")
    folium.plugins.Draw(export=True).add_to(m)
    map_data = st_folium(m, width=800, height=400)

with col2:
    st.subheader("🧪 Крок 2. Агрохімія ґрунту")
    n_val = st.number_input("Азот (N)", value=90)
    p_val = st.number_input("Фосфор (P)", value=42)
    k_val = st.number_input("Калій (K)", value=43)
    ph_val = st.number_input("Кислотність (pH)", value=6.5, step=0.1)

if map_data["all_drawings"]:
    st.markdown("---")
    st.header("📊 Результати аналізу")
    
    # Виклик ГІС-модуля
    geom = map_data["all_drawings"][0]["geometry"]
    area_2d, area_3d, slope, temp, hum, rain = analyze_polygon(geom)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Площа 2D", f"{area_2d:.2f} га")
    c2.metric("Площа 3D", f"{area_3d:.2f} га")
    c3.metric("Схил", f"{slope}°")
    c4.metric("Температура", f"{temp} °C")

    # Виклик ML-модуля
    base_probs = predict_crops(model, scaler, n_val, p_val, k_val, temp, hum, ph_val, rain)
    final_probs = base_probs.copy()
    
    # ГІС-Фільтр
    if slope > 15:
        st.error("🚨 КРИТИЧНИЙ СХИЛ (>15°)! Землеробство заборонено.")
        final_probs *= 0 
    elif slope >= 5:
        st.warning("⚠️ УВАГА! Схил 5-15°. Просапні культури виключені.")
        for i, crop in enumerate(crop_classes):
            if crop in row_crops: final_probs[i] *= 0.0 
    else:
        st.success("✅ Рівнинна ділянка. Базова оранка дозволена.")

    if np.sum(final_probs) > 0:
        final_probs = final_probs / np.sum(final_probs)
        st.markdown("### 🏆 Оптимальні культури:")
        top_idx = final_probs.argsort()[-3:][::-1] 
        for i, idx in enumerate(top_idx):
            if final_probs[idx] > 0.01:
                st.progress(float(final_probs[idx]))
                st.write(f"**#{i+1} {crop_classes[idx].capitalize()}** — {final_probs[idx]*100:.1f}%")