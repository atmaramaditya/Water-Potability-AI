import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(
    page_title="HydroGuard AI | Aditya Atmaram", 
    page_icon="💧", 
    layout="wide"
)

# 2. Modern Glassmorphism CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stSlider { padding-bottom: 20px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    
    /* Custom Card Design */
    .status-card {
        padding: 20px;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-top: 5px solid #007bff;
    }
    
    /* Metric styling */
    .metric-text { font-weight: bold; color: #1f77b4; font-size: 1.2rem; }
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

# 4. SIDEBAR - Professional Branding
with st.sidebar:
    st.markdown("## 🛰️ System Status")
    st.success("Model: Random Forest (Active)")
    st.markdown("---")
    st.markdown("### 👨‍🎓 Developer")
    st.write("**Aditya Atmaram**")
    st.caption("B.Tech Mechatronics Candidate @ MPSTME")
    st.caption("AI & Data Science @ BIA")
    st.markdown("---")
    
    # Model Specs in Sidebar
    with st.expander("📈 Model Evaluation Metrics"):
        st.write("**Accuracy:** 65%")
        st.write("**F1-Score:** 0.64")
        st.write("**Precision (Class 0):** 0.69")

# 5. HEADER
col_title, col_anim = st.columns([3, 1])
with col_title:
    st.title("HydroGuard: AI Water Analysis")
    st.info("Interactive Laboratory Simulation Dashboard")

# 6. INTERACTIVE INPUTS (Using Sliders for better UX)
st.markdown("### 🎛️ Digital Sensor Inputs")
with st.container():
    c1, c2, c3 = st.columns(3)
    
    with c1:
        ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1, help="WHO Standard: 6.5 - 8.5")
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

# 7. ANALYSIS LOGIC
if st.button("🚀 RUN SYSTEM DIAGNOSTIC", use_container_width=True):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    st.markdown("---")
    
    # Create Layout for Results
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.markdown('<div class="status-card" style="border-top: 5px solid #28a745;">', unsafe_allow_html=True)
            st.header("✅ POTABLE")
            st.write("Safe for human consumption.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card" style="border-top: 5px solid #dc3545;">', unsafe_allow_html=True)
            st.header("❌ UNSAFE")
            st.write("Potentially contaminated.")
            st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        st.subheader("📋 Compliance Checklist")
        
        # Real-time parameter check table
        check_data = {
            "Parameter": ["pH Level", "Chloramines", "Sulfate", "Turbidity"],
            "Value": [ph, chloramines, sulfate, turbidity],
            "WHO Limit": ["6.5 - 8.5", "< 4.0 ppm", "< 250 mg/L", "< 5.0 NTU"],
            "Status": [
                "✅ Pass" if 6.5 <= ph <= 8.5 else "🛑 Fail",
                "✅ Pass" if chloramines <= 4.0 else "🛑 Fail",
                "✅ Pass" if sulfate <= 250.0 else "🛑 Fail",
                "✅ Pass" if turbidity <= 5.0 else "🛑 Fail"
            ]
        }
        st.table(pd.DataFrame(check_data))

# 8. EDUCATIONAL SECTION (Innovative Expanders)
st.markdown("---")
with st.expander("📖 Learn about Water Quality Parameters"):
    st.write("""
    Understanding these parameters is vital for Mechatronics applications in water treatment plants:
    * **Turbidity:** Measures water clarity using light scattering.
    * **Conductivity:** Indicates the amount of dissolved salts.
    * **Trihalomethanes:** Chemicals often found in water treated with chlorine.
    """)



st.caption("Developed by Aditya Atmaram | MPSTME Mechatronics Portfolio")
