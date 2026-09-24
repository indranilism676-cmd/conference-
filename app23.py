import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from catboost import CatBoostRegressor

# ---------------------------------------------------------
# Page Configuration & Load Model
# ---------------------------------------------------------
st.set_page_config(page_title="Capillary Pressure Upscaling", page_icon="🛢️", layout="wide")
st.title("🛢️ Capillary Pressure Upscaling using Machine Learning")

st.markdown("""
Upload a Core Capillary Pressure curve and enter the core, reservoir, and lithology properties.
The CatBoost model predicts the Leverett scaling factor accounting for pore heterogeneity and generates the upscaled capillary pressure curve.
""")

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("ScalingFactorCatBoost4.cbm")
    return model

model = load_model()

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("Rock Properties")

lithology = st.sidebar.selectbox("Lithology", ["Sandstone", "Carbonate", "Shaly-Sand"])

phi_core = st.sidebar.number_input("Core Porosity", min_value=0.0001, value=0.18, format="%.4f")
k_core = st.sidebar.number_input("Core Permeability (mD)", min_value=0.0001, value=100.0)

phi_res = st.sidebar.number_input("Reservoir Porosity", min_value=0.0001, value=0.25, format="%.4f")
k_res = st.sidebar.number_input("Reservoir Permeability (mD)", min_value=0.0001, value=500.0)

# ---------------------------------------------------------
# Data Upload & Prediction
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Capillary Pressure CSV", type=["csv"])

if uploaded_file is not None:
    try:
        try:
            curve = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            curve = pd.read_csv(uploaded_file, encoding="latin1")

        if not all(col in curve.columns for col in ["Sw", "Core_Pc"]):
            st.error("CSV must contain columns: Sw and Core_Pc")
            st.stop()

        curve = curve[["Sw", "Core_Pc"]].replace([np.inf, -np.inf], np.nan).dropna()
        curve = curve.sort_values("Sw").drop_duplicates(subset="Sw").reset_index(drop=True)

        if len(curve) < 2:
            st.error("At least two data points are required.")
            st.stop()

        st.subheader("Uploaded Data")
        st.dataframe(curve)

        if st.button("Predict Upscaled Curve"):
            with st.spinner("Predicting..."):
                # Standardize to 50 interpolation points matching the training pipeline
                common_sw = np.linspace(curve["Sw"].min(), curve["Sw"].max(), 50)

                interpolation = interp1d(curve["Sw"], curve["Core_Pc"], kind="linear", bounds_error=False, fill_value="extrapolate")
                core_pc_interp = interpolation(common_sw)
                core_pc_norm = core_pc_interp / core_pc_interp[0]

                # Construct Inference DataFrame
                feature_dict = {
                    "Core_Porosity": phi_core,
                    "Core_Permeability_mD": k_core,
                    "Reservoir_Porosity": phi_res,
                    "Reservoir_Permeability_mD": k_res,
                    "Lithology": lithology
                }
                
                for i in range(50):
                    feature_dict[f"Pc_norm_{i}"] = core_pc_norm[i]
                
                features_df = pd.DataFrame([feature_dict])

                # Model execution
                log_scale = model.predict(features_df)[0]
                scale = np.exp(log_scale)

                curve["Upscaled_Pc"] = curve["Core_Pc"] * scale

                st.metric("Predicted Scaling Factor", f"{scale:.4f}")

                # --- PLOT UPDATES HERE ---
                # Reduced figsize and use_container_width=False to prevent horizontal stretching
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(curve["Sw"], curve["Core_Pc"], linewidth=2, label="Core Curve")
                ax.plot(curve["Sw"], curve["Upscaled_Pc"], linewidth=2, label="Predicted Upscaled Curve")
                ax.set_xlabel("Water Saturation")
                ax.set_ylabel("Capillary Pressure")
                ax.set_title("Capillary Pressure Upscaling")
                ax.grid(True)
                ax.legend()
                
                st.pyplot(fig, use_container_width=False)
                # -------------------------

                st.subheader("Predicted Data")
                st.dataframe(curve)

                csv = curve.to_csv(index=False).encode("utf-8")
                st.download_button("Download Upscaled Curve", csv, "upscaled_curve.csv", "text/csv")

    except Exception as e:
        st.error(f"Error while processing file:\n\n{e}")