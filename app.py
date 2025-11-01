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

                # 🟢 Save prediction and inputs for report section
                st.session_state["last_input"] = {
                    "Crop": crop,
                    "Crop Year": year,
                    "Season": season,
                    "State": state,
                    "Area": area,
                    "Production": prod,
                    "Annual Rainfall": rain,
                    "Fertilizer": fert,
                    "Pesticide": pest
                }
                st.session_state["last_prediction"] = float(pred)
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(0, 128, 0, 0.4);
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        font-weight: bold;
                        font-size: 24px;
                        box-shadow: 0px 0px 10px rgba(0,255,0,0.3);
                        border: 1px solid rgba(255,255,255,0.2);
                        ">
                        🌾 Predicted Crop Yield: {pred:.2f} tons/hectare
                    </div>
                    """,
                    unsafe_allow_html=True
)


                
            except Exception as e:
                st.error(str(e))


# ------------------ REPORT (unchanged placeholder, you can keep CSV/PDF logic) ------------------
# ------------------ REPORT (UPDATED: now uses last prediction & allows CSV + PDF) ------------------
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def report_section():
    st.header("📄 Download Report")

    # Check if prediction and inputs exist
    if "last_input" not in st.session_state or "last_prediction" not in st.session_state:
        st.warning("Please make a prediction first in the 'Predict Yield' section.")
        return

    input_data = st.session_state["last_input"]
    predicted_yield = st.session_state["last_prediction"]

    # Display result summary
    st.markdown(
        f"""
        <div style='background-color:rgba(255,255,255,0.2);padding:15px;border-radius:10px;text-align:center;color:white;font-weight:bold;font-size:20px;'>
        🌾 <b>Predicted Crop Yield:</b> {predicted_yield:.2f} tons/hectare
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prepare report data
    report_df = pd.DataFrame([input_data])
    report_df["Predicted_Yield"] = predicted_yield

    # CSV Download
    csv_data = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Report (CSV)",
        data=csv_data,
        file_name="yield_report.csv",
        mime="text/csv"
    )

    # PDF Download
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "Smart Crop Yield Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 710, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y_position = 670
    for key, value in input_data.items():
        c.drawString(50, y_position, f"{key}: {value}")
        y_position -= 20

    c.drawString(50, y_position - 10, f"Predicted Yield: {predicted_yield:.2f} tons/hectare")
    c.showPage()
    c.save()

    pdf_buffer.seek(0)
    st.download_button(
        label="📘 Download Report (PDF)",
        data=pdf_buffer,
        file_name="yield_report.pdf",
        mime="application/pdf"
    )


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
# ------------------ CROP SUGGESTION HELPER (improved: no fallback sliders or messages) ------------------
def crop_suggestion_section():
    st.header("🌿 Crop Suggestion Helper")

    df, fname = load_any_dataset()
    if df is None:
        st.warning("Dataset not found. Please upload a valid crop yield dataset.")
        return

    # Select filters
    states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else []
    seasons = sorted(df['Season'].dropna().unique().tolist()) if 'Season' in df.columns else []

    col1, col2 = st.columns(2)
    with col1:
        selected_state = st.selectbox("Select State", options=["All"] + states)
    with col2:
        selected_season = st.selectbox("Select Season", options=["All"] + seasons)

    if st.button("🔍 Suggest Best Crops"):
        filt = df.copy()
        if selected_state != "All" and 'State' in filt.columns:
            filt = filt[filt['State'] == selected_state]
        if selected_season != "All" and 'Season' in filt.columns:
            filt = filt[filt['Season'] == selected_season]

        # If no matching records, show global top crops
        if filt.empty:
            st.info("No crops found for your result, showing top-performing crops across all regions instead.")
            filt = df.copy()

        # Calculate top crops
        if 'Crop' in filt.columns and 'Yield' in filt.columns:
            top_crops = (
                filt.groupby('Crop')['Yield']
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )

            st.success("🌱 Recommended Crops:")
            for i, row in top_crops.iterrows():
                st.markdown(f"**{i+1}. {row['Crop']}** — Avg Yield: {row['Yield']:.2f}")
        else:
            st.error("Dataset does not contain required columns 'Crop' and 'Yield'.")


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
