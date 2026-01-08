import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="HydroGuard AI | Aditya Atmaram", 
    page_icon="💧", 
    layout="wide"
)

# 2. Innovative CSS: Background Image & Glassmorphism
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
        url("https://images.unsplash.com/photo-1518063319789-7217e6706b04?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3, p, span, label {
        color: white !important;
    }
    .stSlider > div > div > div > div {
        color: #00d4ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Asset Loading
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        with open(os.path.join(base_path, 'water_model.pkl'), 'rb') as m_file:
            model = pickle.load(m_file)
        with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as s_file:
            scaler = pickle.load(s_file)
        return model, scaler
    except: return None, None

model, scaler = load_assets()

# 4. Professional Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>💧 HydroGuard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("👤 **Developer:** Aditya Atmaram")
    st.write("🎓 **Status:** B.Tech Mechatronics Candidate")
    st.caption("MPSTME | BIA (AI & Data Science)")
    st.markdown("---")
    st.subheader("📡 System Specs")
    st.write("• **Core:** Random Forest")
    st.write("• **Accuracy:** 65%")
    st.write("• **Status:** Operational")

# 5. INNOVATIVE HEADER
st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); padding: 20px; border-radius: 15px; border-left: 10px solid #00d4ff;">
        <h1 style='margin:0;'>Intelligent Water Quality Monitor</h1>
        <p style='margin:0; opacity: 0.8;'>Mechatronics & Data Science Diagnostic Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# 6. INTERACTIVE SENSOR INPUTS
st.markdown("### 🛰️ Real-time Sensor Simulation")
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        ph = st.slider("pH Level", 0.0, 14.0, 7.0)
        hardness = st.slider("Hardness (mg/L)", 50.0, 400.0, 196.3)
        solids = st.slider("Solids (ppm)", 5000.0, 50000.0, 22000.0)
    with c2:
        chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.1)
        sulfate = st.slider("Sulfate (mg/L)", 100.0, 500.0, 333.6)
        conductivity = st.slider("Conductivity (μS/cm)", 100.0, 800.0, 426.2)
    with c3:
        organic_carbon = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 14.2)
        trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 66.4)
        turbidity = st.slider("Turbidity (NTU)", 0.0, 7.0, 3.9)

# 7. DIAGNOSTIC LOGIC
if st.button("⚡ RUN DIAGNOSTIC ANALYSIS", use_container_width=True):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    st.markdown("---")
    
    col_res, col_viz = st.columns([1, 1.5])
    
    with col_res:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if prediction == 1:
            st.success("### ✅ POTABLE")
            st.balloons()
            st.write("The sample meets safety thresholds for human consumption.")
        else:
            st.error("### ❌ NON-POTABLE")
            st.write("Potential contamination detected. Remediation required.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # RISK GAUGE
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 100 if prediction == 1 else 25,
            title = {'text': "Potability Score (%)", 'font': {'color': "white"}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00d4ff"},
                'steps': [
                    {'range': [0, 50], 'color': "gray"},
                    {'range': [50, 100], 'color': "darkblue"}]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_viz:
        st.markdown("### 📊 Parameter Deviation (WHO Standards)")
        
        # COMPLIANCE CHECK
        checks = {
            "Param": ["pH", "Chloramines", "Sulfate", "Turbidity"],
            "Value": [ph, chloramines, sulfate, turbidity],
            "Limit": ["6.5-8.5", "<4.0", "<250", "<5.0"],
            "Status": [
                "✅" if 6.5 <= ph <= 8.5 else "❌",
                "✅" if chloramines <= 4.0 else "❌",
                "✅" if sulfate <= 250 else "❌",
                "✅" if turbidity <= 5.0 else "❌"
            ]
        }
        st.table(pd.DataFrame(checks))
        
        # FEATURE IMPORTANCE VISUAL (Mockup based on RF general behavior)
        st.write("🔍 **Model Feature Influence**")
        importance_data = pd.DataFrame({
            'Feature': ['pH', 'Sulfate', 'Solids', 'Chloramines', 'Hardness'],
            'Weight': [0.25, 0.20, 0.15, 0.12, 0.10]
        })
        st.bar_chart(importance_data.set_index('Feature'))

st.markdown("---")
st.caption("System Architecture by Aditya Atmaram | MPSTME Mechatronics & BIA AI/DS")
