# y3.py  (updated: visualization & crop suggestion restored from chit.py behavior)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime
import plotly.express as px

# ------------------ Page config & CSS (unchanged) ------------------
st.set_page_config(page_title="Smart Crop Yield Predictor", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.postimg.cc/ZRt15qw6/2be3971e-28de-4f8c-b69b-d8e4fc4dc99c.png");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
}

/* General text color */
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: white !important;
    text-shadow: 1px 1px 2px black;
}

/* Input and select field box styling */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="slider"] > div {
    background-color: rgba(0, 50, 0, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Input text color */
input, textarea {
    color: white !important;
}

/* Dropdown menu background and text color */
div[role="listbox"] {
    background-color: rgba(0, 50, 0, 0.95) !important;
}
div[role="listbox"] div {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background-color: rgba(0, 128, 0, 0.7);
    color: white !important;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5em 1.5em;
}

/* Quote box */
.quote-box {
    background-color: rgba(0,128,0,0.3);
    border: 2px solid white;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-style: italic;
    color: white;
}

/* Container */
.container-box {
    background-color: rgba(0,0,0,0.18);
    padding: 18px;
    border-radius: 10px;
}

/* Hover button style */
.hover-box {
    background-color: rgba(0, 100, 0, 0.5);
    border-radius: 15px;
    padding: 20px 40px;
    color: white;
    text-align: center;
    font-weight: bold;
    font-size: 25px;
    transition: all 0.4s ease;
    cursor: pointer;
}
.hover-box:hover {
    background-color: rgba(0, 150, 0, 0.8);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------ Simple user management (unchanged) ------------------
USER_FILE = "users.csv"
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)

def load_users():
    return pd.read_csv(USER_FILE).fillna("")

def save_user(u, p):
    df = load_users()
    if u in df["username"].values:
        return False, "Username already exists"
    df = pd.concat([df, pd.DataFrame([{"username": u, "password": p}])], ignore_index=True)
    df.to_csv(USER_FILE, index=False)
    return True, "Signup successful!"

def check_user(u, p):
    df = load_users()
    if u in df["username"].values:
        if str(df.loc[df["username"] == u, "password"].values[0]) == p:
            return True, "Login successful!"
        else:
            return False, "Invalid password"
    return False, "User not found"

# ------------------ Load model artifacts (unchanged) ------------------
def safe_load(fname):
    try:
        return joblib.load(fname)
    except Exception:
        return None

model = safe_load("trained_model.pkl") or safe_load("trained_yield_model.pkl") or safe_load("trained_model.pkl")
encoders = safe_load("label_encoders.pkl")
scaler = safe_load("scaler.pkl")

# ------------------ Helper: build model input (unchanged) ------------------
def make_input(crop, year, season, state, area, prod, rain, fert, pest):
    # convert using encoders if available
    if encoders:
        try:
            crop_val = encoders['Crop'].transform([crop])[0]
            season_val = encoders['Season'].transform([season])[0]
            state_val = encoders['State'].transform([state])[0]
        except Exception:
            crop_val, season_val, state_val = crop, season, state
    else:
        crop_val, season_val, state_val = crop, season, state

    df = pd.DataFrame({
        "Crop":[crop_val],
        "Crop_Year":[year],
        "Season":[season_val],
        "State":[state_val],
        "Area":[area],
        "Production":[prod],
        "Annual_Rainfall":[rain],
        "Fertilizer":[fert],
        "Pesticide":[pest]
    })
    # scale numeric if scaler available
    if scaler is not None:
        try:
            numeric_cols = ["Area","Production","Annual_Rainfall","Fertilizer","Pesticide"]
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        except Exception:
            pass
    return df

# ------------------ Dataset loader used by visualization & suggestion ------------------
def load_any_dataset():
    # Try files commonly used in your project (ordered)
    candidates = ["crop_yield.csv", "Cleaned_Crop_Yield.csv", "crop_yield_data.csv", "agri_yield_data.csv", "crop_yield.csv"]
    for f in candidates:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                # normalize column names
                df.columns = [c.strip() for c in df.columns]
                return df, f
            except Exception:
                continue
    return None, None

# ------------------ PREDICT SECTION (unchanged) ------------------
def predict_section():
    st.header("🌾 Smart Yield Predictor")
    if encoders:
        try:
            crop_opt = list(encoders["Crop"].classes_)
            season_opt = list(encoders["Season"].classes_)
            state_opt = list(encoders["State"].classes_)
        except Exception:
            crop_opt = season_opt = state_opt = []
    else:
        crop_opt = season_opt = state_opt = []

    if crop_opt:
        crop = st.selectbox("Crop", crop_opt)
    else:
        crop = st.text_input("Crop")

    year = st.number_input("Crop Year", 1900, 2100, 2023)
    if season_opt:
        season = st.selectbox("Season", season_opt)
    else:
        season = st.text_input("Season")
    if state_opt:
        state = st.selectbox("State", state_opt)
    else:
        state = st.text_input("State")

    area = st.number_input("Area (ha)", 0.0)
    prod = st.number_input("Production (tons)", 0.0)
    rain = st.number_input("Annual Rainfall (mm)", 0.0)
    fert = st.number_input("Fertilizer (kg/ha)", 0.0)
    pest = st.number_input("Pesticide (kg/ha)", 0.0)

    if st.button("Predict Yield"):
        if model is None:
            st.error("Model not found!")
        else:
            df_in = make_input(crop, year, season, state, area, prod, rain, fert, pest)
            try:
                pred = model.predict(df_in)[0]
                st.markdown(f"<div style='background-color:rgba(255,255,255,0.2);padding:15px;border-radius:10px;text-align:center;color:white;font-weight:bold;font-size:22px;'>🌾 Predicted Crop Yield: {pred:.2f} tons/hectare</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

# ------------------ REPORT (unchanged placeholder, you can keep CSV/PDF logic) ------------------
def report_section():
    st.header("📄 Download Report")
    # If a previous prediction was made, you can adapt this to output that result
    sample = pd.DataFrame([{
        "Crop":"Arecanut","Season":"Autumn","State":"Andhra Pradesh","Crop Year":2023,
        "Area":1,"Production":1,"Annual Rainfall":500
    }])
    st.dataframe(sample)
    csv = sample.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "yield_report.csv", "text/csv")

# ------------------ VISUALIZATION (REPLACED - now like chit.py) ------------------
def viz_section():
    st.header("📊 Visualization (Yield Insights)")

    df, fname = load_any_dataset()
    if df is None:
        st.info("No dataset found. Place 'crop_yield.csv' or 'Cleaned_Crop_Yield.csv' in the app folder to enable visualizations.")
        return

    st.success(f"Loaded data from: {fname}")

    # Ensure required columns exist
    expected_cols = set(['Crop','Crop_Year','State','Yield'])
    missing = expected_cols - set(df.columns)
    if missing:
        st.warning(f"Dataset missing columns required for full visualization: {', '.join(missing)}. Partial visuals may be available.")

    # Sidebar-like filter UI inside section (keeps inline, not sidebar)
    st.markdown("#### Filters")
    states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    crops = sorted(df['Crop'].dropna().unique().tolist()) if 'Crop' in df.columns else []
    years = sorted(df['Crop_Year'].dropna().astype(int).unique().tolist()) if 'Crop_Year' in df.columns else []

    colf1, colf2, colf3 = st.columns([1,1,2])
    with colf1:
        sel_state = st.selectbox("State", ["All"] + states, index=0)
    with colf2:
        sel_crop = st.selectbox("Crop", ["All"] + crops, index=0)
    with colf3:
        if years:
            sel_year = st.select_slider("Year range", options=years, value=(min(years), max(years)))
        else:
            sel_year = (None, None)

    vis_df = df.copy()
    if sel_state != "All":
        vis_df = vis_df[vis_df['State'] == sel_state]
    if sel_crop != "All":
        vis_df = vis_df[vis_df['Crop'] == sel_crop]
    if sel_year and sel_year[0] is not None:
        try:
            vis_df = vis_df[(vis_df['Crop_Year'].astype(int) >= int(sel_year[0])) & (vis_df['Crop_Year'].astype(int) <= int(sel_year[1]))]
        except Exception:
            pass

    st.markdown("##### Data preview")
    st.dataframe(vis_df.head(200), width='stretch')

    # Chart 1: Average yield per crop
    if 'Yield' in vis_df.columns:
        st.markdown("##### Average Yield per Crop")
        bar_df = vis_df.groupby('Crop', dropna=False)['Yield'].mean().reset_index().sort_values('Yield', ascending=False)
        if not bar_df.empty:
            fig = px.bar(bar_df, x='Crop', y='Yield', title='Average Yield by Crop', color='Yield', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to show average yield per crop.")

    # Chart 2: Yield trend by year for selected crop
    if sel_crop != "All" and 'Yield' in vis_df.columns and 'Crop_Year' in vis_df.columns:
        try:
            trend = vis_df.groupby(vis_df['Crop_Year'].astype(int))['Yield'].mean().reset_index().sort_values('Crop_Year')
            if not trend.empty:
                st.markdown(f"##### Yield trend for {sel_crop}")
                fig2 = px.line(trend, x='Crop_Year', y='Yield', title=f'Yield trend for {sel_crop}')
                st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass

# ------------------ CROP SUGGESTION HELPER (REPLACED: same logic as chit.py sidebar but as 4th section) ------------------
def crop_suggestion_section():
    st.header("🌿 Crop Suggestion Helper (data-driven + fallback rules)")
    df, fname = load_any_dataset()
    # Inputs (same UI feel as chit.py sidebar but placed here)
    if df is None:
        st.info("No dataset found. Crop suggestions will use rule-based fallback.")
        selected_state = None
        selected_season = None
    else:
        # Try to present dropdowns for State/Season if columns exist
        if 'State' in df.columns:
            selected_state = st.selectbox("Select State", options=["All"] + sorted(df['State'].dropna().unique().tolist()))
        else:
            selected_state = "All"
        if 'Season' in df.columns:
            selected_season = st.selectbox("Select Season", options=["All"] + sorted(df['Season'].dropna().unique().tolist()))
        else:
            selected_season = "All"

    if st.button("🔍 Suggest Best Crops"):
        # If dataset present and has required columns, provide top crops by average yield for selected filters
        if df is not None and {'Crop','Yield'}.issubset(set(df.columns)):
            filt = df.copy()
            if selected_state and selected_state != "All" and 'State' in filt.columns:
                filt = filt[filt['State'] == selected_state]
            if selected_season and selected_season != "All" and 'Season' in filt.columns:
                filt = filt[filt['Season'] == selected_season]
            try:
                top = filt.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(5)
                if not top.empty:
                    st.success("🌱 Top recommended crops based on historical average yield:")
                    for i, (cname, val) in enumerate(top.items(), start=1):
                        st.markdown(f"{i}. **{cname}** — Avg Yield: {val:.2f}")
                else:
                    st.warning("No matching records for the selected filters. Falling back to rule-based suggestions.")
                    raise ValueError("No records")
            except Exception:
                # fallback to rule-based
                st.info("Fallback rule-based suggestions:")
                # Simple rule-based suggestions (same rules you used)
                rain = st.number_input("If fallback: Enter Annual Rainfall (mm):", min_value=0.0, value=500.0)
                fert = st.number_input("If fallback: Enter Fertilizer used (kg/ha):", min_value=0.0, value=100.0)
                pest = st.number_input("If fallback: Enter Pesticide used (kg/ha):", min_value=0.0, value=10.0)
                if rain < 300:
                    suggestion = "Millets — Best for low rainfall areas!"
                elif 300 <= rain < 700 and fert < 200:
                    suggestion = "Wheat — Balanced for medium rainfall and low fertilizer."
                elif fert >= 200 and pest < 50:
                    suggestion = "Rice — Ideal under high fertilizer and medium rainfall."
                else:
                    suggestion = "Sugarcane — Thrives in high rainfall and high fertilizer input."
                st.success(f"✅ Suggested Crop: **{suggestion}**")
        else:
            # Dataset not present or missing required columns -> fallback rule-based
            st.info("Dataset not found or missing 'Crop'/'Yield' columns. Using rule-based suggestions.")
            rain = st.number_input("Enter Annual Rainfall (mm):", min_value=0.0, value=500.0)
            fert = st.number_input("Enter Fertilizer used (kg/ha):", min_value=0.0, value=100.0)
            pest = st.number_input("Enter Pesticide used (kg/ha):", min_value=0.0, value=10.0)
            if rain < 300:
                suggestion = "Millets — Best for low rainfall areas!"
            elif 300 <= rain < 700 and fert < 200:
                suggestion = "Wheat — Balanced for medium rainfall and low fertilizer."
            elif fert >= 200 and pest < 50:
                suggestion = "Rice — Ideal under high fertilizer and medium rainfall."
            else:
                suggestion = "Sugarcane — Thrives in high rainfall and high fertilizer input."
            st.success(f"✅ Suggested Crop: **{suggestion}**")

# ------------------ Main app layout & routing (unchanged) ------------------
def show_login():
    st.title("🌾 Welcome to Smart Crop Yield 🌞")
    st.markdown('<div class="quote-box">"The seed you plant today decides your harvest tomorrow."</div>', unsafe_allow_html=True)

    choice = st.radio("Select", ["Login", "Sign Up"])
    if choice == "Login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            ok, msg = check_user(u, p)
            if ok:
                st.session_state["user"] = u
                st.session_state["page"] = "predict"
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    else:
        newu = st.text_input("Create Username")
        newp = st.text_input("Create Password", type="password")
        if st.button("Sign Up"):
            ok, msg = save_user(newu, newp)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

def main_app():
    st.markdown(f"<h3 style='text-align:center;'>Welcome, {st.session_state.get('user','Guest')} 👋</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🌾 Predict Yield"):
            st.session_state['page'] = "predict"
    with col2:
        if st.button("📄 Download Report"):
            st.session_state['page'] = "report"
    with col3:
        if st.button("📊 Visualization"):
            st.session_state['page'] = "viz"
    with col4:
        if st.button("🌿 Crop Suggestion Helper"):
            st.session_state['page'] = "helper"

    st.markdown("---")

    page = st.session_state.get('page', 'predict')
    if page == "predict":
        predict_section()
    elif page == "report":
        report_section()
    elif page == "viz":
        viz_section()
    elif page == "helper":
        crop_suggestion_section()
    else:
        predict_section()

    if st.button("Sign Out"):
        st.session_state.clear()
        st.rerun()

def main():
    if "user" not in st.session_state:
        show_login()
    else:
        main_app()

if __name__ == "__main__":
    main()
