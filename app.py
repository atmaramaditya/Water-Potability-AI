import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="HydroGuard AI", page_icon="💧", layout="wide")

# 2. High-Contrast CSS
st.markdown("""
    <style>
    .stApp { background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("https://images.unsplash.com/photo-1518063319789-7217e6706b04?q=80&w=2000&auto=format&fit=crop"); background-size: cover; }
    [data-testid="stSidebar"] { background-color: #0e1117 !important; border-right: 2px solid #00d4ff; }
    .glass-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; border: 1px solid rgba(0,212,255,0.3); }
    div.stButton > button { background-color: #00d4ff !important; color: #0e1117 !important; font-weight: bold; width: 100%; height: 3em; border-radius: 10px; }
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .stSlider label { color: #00d4ff !important; }
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
    st.info("System: Operational")

# 5. Header
st.markdown("<div style='border-left:8px solid #00d4ff; padding:10px; background:rgba(0,212,255,0.1);'><h1>Water Quality AI</h1><p>Diagnostic Dashboard</p></div>", unsafe_allow_html=True)

# 6. Inputs
st.markdown("### 🛰️ Sensor Data")
c1, c2, c3 = st.columns(3)
with c1:
    v1 = st.slider("pH Level", 0.0, 14.0, 7.0)
    v2 = st.slider("Hardness", 50.0, 400.0, 196.0)
    v3 = st.slider("Solids", 5000.0, 50000.0, 22000.0)
with c2:
    v4 = st.slider("Chloramines", 0.0, 15.0, 7.0)
    v5 = st.slider("Sulfate", 100.0, 500.0, 333.0)
    v6 = st.slider("Conductivity", 100.0, 800.0, 426.0)
with c3:
    v7 = st.slider("Carbon", 0.0, 30.0, 14.0)
    v8 = st.slider("Trihalomethanes", 0.0, 130.0, 66.0)
    v9 = st.slider("Turbidity", 0.0, 7.0, 3.9)

# 7. Logic
if st.button("⚡ RUN DIAGNOSTIC"):
    if model and scaler:
        arr = np.array([[v1,v2,v3,v4,v5,v6,v7,v8,v9]])
        pred = model.predict(scaler.transform(arr))[0]
        st.markdown("---")
        res, viz = st.columns([1, 1.5])
        
        with res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if pred == 1: st.success("### ✅ RESULT: POTABLE")
            else: st.error("### ❌ RESULT: UNSAFE")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # THE GAUGE
            fig = go.Figure(go.Indicator(mode="gauge+number", value=(100 if pred==1 else 25), gauge={'bar':{'color':"#00d4ff"}}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Confidence Meter:** This gauge represents the model's safety rating. 100% indicates the sample is classified as safe for consumption.")
            
        with viz:
            st.markdown("### 📋 Compliance Check")
            df = pd.DataFrame({
                "Parameter": ["pH Balance", "Sulfate Level", "Chlorine", "Clarity"],
                "Value": [v1, v5, v4, v9],
                "WHO Limit": ["6.5 - 8.5", "< 250 mg/L", "< 4.0 ppm", "< 5.0 NTU"],
                "Status": ["✅ Pass" if 6.5<=v1<=8.5 else "🛑 Fail", "✅ Pass" if v5<=250 else "🛑 Fail", "✅ Pass" if v4<=4 else "🛑 Fail", "✅ Pass" if v9<=5 else "🛑 Fail"]
            })
            st.table(df)
            st.write("**Why this table?** While the AI looks at all 9 sensors together, this table highlights the most critical safety benchmarks set by the WHO. A 'Fail' status indicates the specific chemical causing the safety risk.")
    else: st.error("Assets missing!")

st.markdown("---")
st.caption("Aditya Atmaram | Mechatronics Portfolio | 2026")
