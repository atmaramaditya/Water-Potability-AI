import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="HydroGuard AI", page_icon="💧", layout="wide")

# 2. CSS Styling
st.markdown("""
    <style>
    .stApp { background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("https://images.unsplash.com/photo-1518063319789-7217e6706b04?q=80&w=2000&auto=format&fit=crop"); background-size: cover; }
    [data-testid="stSidebar"] { background-color: #0e1117 !important; border-right: 2px solid #00d4ff; }
    .glass-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; border: 1px solid rgba(0,212,255,0.3); margin-bottom: 20px;}
    div.stButton > button { background-color: #00d4ff !important; color: #0e1117 !important; font-weight: bold; width: 100%; height: 3em; border-radius: 10px; }
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

# 4. Sidebar
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;'>💧 HydroGuard</h2>", unsafe_allow_html=True)
    st.write("👤 **Aditya Atmaram**")
    st.caption("B.Tech Mechatronics | MPSTME")
    st.caption("AI & Data Science | BIA")
    st.markdown("---")
    st.markdown("### 📊 System Specs")
    st.markdown("""<div class="stat-box"><small>Model</small><br><b>Random Forest</b></div>
    <div class="stat-box"><small>Accuracy</small><br><b>65%</b></div>""", unsafe_allow_html=True)
    st.success("System: Operational")

# 5. Header
st.markdown("<div style='border-left:8px solid #00d4ff; padding:10px; background:rgba(0,212,255,0.1);'><h1>Water Quality AI</h1><p>Intelligent Root Cause Diagnostics</p></div>", unsafe_allow_html=True)

# 6. Sensor Inputs
st.markdown("### 🛰️ Sensor Data")
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
if st.button("⚡ RUN SYSTEM DIAGNOSTIC"):
    if model and scaler:
        arr = np.array([[v1,v2,v3,v4,v5,v6,v7,v8,v9]])
        scaled_data = scaler.transform(arr)
        
        ai_pred = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0]
        confidence = round(prob[1] * 100, 2)
        
        # ROOT CAUSE ANALYSIS LOGIC
        critical_issues = []
        if not (6.5 <= v1 <= 8.5): critical_issues.append(f"pH Level ({v1}) is outside the safe range of 6.5-8.5.")
        if v4 > 4.0: critical_issues.append(f"Chloramines ({v4} ppm) exceed the safety limit of 4.0 ppm.")
        if v5 > 250.0: critical_issues.append(f"Sulfate ({v5} mg/L) exceeds the 250 mg/L limit.")
        if v9 > 5.0: critical_issues.append(f"Turbidity ({v9} NTU) is too high for safe consumption.")
        
        final_result = 0 if len(critical_issues) > 0 else ai_pred

        st.markdown("---")
        res, viz = st.columns([1, 1.5])
        
        with res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if final_result == 1:
                st.success("### ✅ RESULT: POTABLE")
                st.balloons()
            else:
                st.error("### ❌ RESULT: UNSAFE")
            st.markdown('</div>', unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(mode="gauge+number", value=confidence, title={'text': "AI Safety Confidence", 'font': {'size': 14}}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00d4ff"}}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=230, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        with viz:
            if final_result == 0:
                st.markdown("### 🔍 Root Cause Diagnostic")
                st.info("The following parameters are responsible for the 'Unsafe' classification:")
                
                if critical_issues:
                    for issue in critical_issues:
                        st.markdown(f"⚠️ **{issue}**")
                else:
                    st.warning("No single WHO limit was exceeded, but the AI detected a dangerous combination of borderline chemical levels (Multivariate Interaction).")
                
                st.markdown("---")
            
            st.markdown("### 📋 Sensor Compliance Table")
            df = pd.DataFrame({
                "Sensor": ["pH", "Sulfate", "Chloramine", "Turbidity"],
                "Reading": [v1, v5, v4, v9],
                "Standard": ["6.5-8.5", "<250", "<4.0", "<5.0"],
                "Status": ["✅ Pass" if 6.5<=v1<=8.5 else "🛑 Critical", "✅ Pass" if v5<=250 else "🛑 Critical", "✅ Pass" if v4<=4 else "🛑 Critical", "✅ Pass" if v9<=5 else "🛑 Critical"]
            })
            st.table(df)
    else: st.error("Assets missing!")

st.markdown("---")
st.caption("Aditya Atmaram | MPSTME Mechatronics Portfolio | 2026")
