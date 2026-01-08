import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="HydroGuard AI | Aditya Atmaram", page_icon="💧", layout="wide")

# 2. CSS Styling (Refined for the new Header)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("https://images.unsplash.com/photo-1518063319789-7217e6706b04?q=80&w=2000&auto=format&fit=crop"); background-size: cover; }
    [data-testid="stSidebar"] { background-color: #0e1117 !important; border-right: 2px solid #00d4ff; }
    .glass-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; border: 1px solid rgba(0,212,255,0.3); margin-bottom: 20px;}
    
    /* Unique Header Styling */
    .hero-section {
        background: linear-gradient(90deg, rgba(0,212,255,0.2) 0%, rgba(0,0,0,0) 100%);
        padding: 30px;
        border-left: 5px solid #00d4ff;
        border-radius: 0px 15px 15px 0px;
        margin-bottom: 30px;
    }
    .system-badge {
        background-color: #00d4ff;
        color: #0e1117;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div.stButton > button { background-color: #00d4ff !important; color: #0e1117 !important; font-weight: bold; width: 100%; height: 3em; border-radius: 10px; border: none; }
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .stSlider label { color: #00d4ff !important; font-weight: bold; }
    .stat-box { background: rgba(0, 212, 255, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.3); margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Assets
@st.cache_resource
def load_assets():
    path = os.path.dirname(__file__)
    try:
        with open(os.path.join(path, 'water_model.pkl'), 'rb') as f1: m = pickle.load(f1)
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f2: s = pickle.load(f2)
        return m, s
    except: return None, None

model, scaler = load_assets()

# 4. SIDEBAR (Untouched as requested)
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;'>💧 HydroGuard</h2>", unsafe_allow_html=True)
    st.write("👤 **Aditya Atmaram**")
    st.caption("B.Tech Mechatronics | MPSTME")
    st.caption("AI & Data Science | BIA")
    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    st.markdown("""
    <div class="stat-box"><small>Algorithm</small><br><b>Random Forest</b></div>
    <div class="stat-box"><small>Test Accuracy</small><br><b>65%</b></div>
    <div class="stat-box"><small>Logic</small><br><b>AI + Safety Override</b></div>
    """, unsafe_allow_html=True)
    st.success("System: Operational")

# 5. UNIQUE DUAL-CREDENTIAL HEADER
st.markdown("""
    <div class="hero-section">
        <div style="display: flex; gap: 10px; margin-bottom: 10px;">
            <span class="system-badge">System ID: HG-2026-V1</span>
            <span class="system-badge" style="background-color: #ffaa00; color: #000;">DOP: JAN-2026</span>
        </div>
        <h1 style='margin:0; font-size: 42px; letter-spacing: -1px;'>HydroGuard <span style='color:#00d4ff;'>Intelligence</span></h1>
        <p style='font-size: 18px; margin-top: 5px; opacity: 0.9;'>
            Neural-Network Assisted Water Potability Analysis & Sensor Diagnostics
        </p>
        <div style='border-top: 1px solid rgba(0,212,255,0.3); margin-top: 15px; padding-top: 10px;'>
            <p style='font-size: 14px; margin:0;'>
                <b style='color:#00d4ff;'>Mechatronics Engineering:</b> Mukesh Patel School of Technology Management and Engineering (MPSTME)
            </p>
            <p style='font-size: 14px; margin:0;'>
                <b style='color:#ffaa00;'>AI & Data Science:</b> Boston Institute of Analytics (BIA)
            </p>
            <p style='font-size: 13px; opacity: 0.7; margin-top: 5px;'>Developer: Aditya Atmaram</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 6. Sensor Inputs
st.markdown("### 🛰️ Real-Time Sensor Array")
c1, c2, c3 = st.columns(3)
with c1:
    v1 = st.slider("pH Level", 0.0, 14.0, 7.0)
    v2 = st.slider("Hardness (mg/L)", 50.0, 400.0, 196.0)
    v3 = st.slider("Solids (TDS)", 5000.0, 50000.0, 22000.0)
with c2:
    v4 = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.0)
    v5 = st.slider("Sulfate (mg/L)", 100.0, 500.0, 333.0)
    v6 = st.slider("Conductivity (μS/cm)", 100.0, 800.0, 426.0)
with c3:
    v7 = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 14.0)
    v8 = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 66.0)
    v9 = st.slider("Turbidity (NTU)", 0.0, 7.0, 3.9)

# 7. Diagnostic Logic
if st.button("⚡ EXECUTE SYSTEM DIAGNOSTIC"):
    if model is not None and scaler is not None:
        input_array = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9]])
        scaled_features = scaler.transform(input_array)
        
        ai_p = model.predict(scaled_features)[0]
        probs = model.predict_proba(scaled_features)[0]
        conf_val = round(probs[1] * 100, 2)
        
        issues = []
        if not (6.5 <= v1 <= 8.5): issues.append(f"pH Imbalance: {v1} (Safe: 6.5-8.5)")
        if v4 > 4.0: issues.append(f"Chloramine Toxicity: {v4} ppm (Safe: <4.0)")
        if v5 > 250.0: issues.append(f"Sulfate Excess: {v5} mg/L (Safe: <250)")
        if v9 > 5.0: issues.append(f"High Turbidity: {v9} NTU (Safe: <5.0)")
        
        final_res = 0 if issues else ai_p

        st.markdown("---")
        res_col, viz_col = st.columns([1, 1.5])
        
        with res_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if final_res == 1:
                st.success("### ✅ STATUS: POTABLE")
                st.balloons()
            else:
                st.error("### ❌ STATUS: UNSAFE")
            st.markdown('</div>', unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=conf_val, 
                title={'text': "AI Reliability Index", 'font': {'size': 14}}, 
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00d4ff"}}
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=230, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        with viz_col:
            if final_res == 0:
                st.markdown("### 🔍 Root Cause Analysis")
                if issues:
                    st.warning("Hardware Override: Sensor violations detected")
                    for i in issues:
                        st.write(f"🛑 {i}")
                else:
                    st.info("AI Anomaly: Pattern recognized as hazardous.")
                st.markdown("---")
            
            st.markdown("### 📋 Validation Matrix")
            df_status = pd.DataFrame({
                "Sensor Node": ["pH", "Sulfate", "Chloramine", "Turbidity"],
                "Actual": [v1, v5, v4, v9],
                "Setpoint": ["6.5-8.5", "<250", "<4.0", "<5.0"],
                "Compliance": ["✅ PASS" if 6.5<=v1<=8.5 else "🛑 FAIL", 
                               "✅ PASS" if v5<=250 else "🛑 FAIL", 
                               "✅ PASS" if v4<=4 else "🛑 FAIL", 
                               "✅ PASS" if v9<=5 else "🛑 FAIL"]
            })
            st.table(df_status)
    else:
        st.error("Diagnostic Assets Offline. Check Repository.")

st.markdown("---")
st.caption("Aditya Atmaram | Mechatronics & AI Engineering | MPSTME-BIA 2026")

