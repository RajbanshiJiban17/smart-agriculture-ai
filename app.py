"""
╔══════════════════════════════════════════════════════════╗
║    SMART AGRICULTURE ADVISOR — STREAMLIT DASHBOARD      ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import requests
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Agriculture Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Agri Theme
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
  --green-1: #00ff87;
  --green-2: #00c853;
  --green-3: #1b5e20;
  --soil:    #3e2723;
  --dark-bg: #0a0f0a;
  --card-bg: #101810;
  --card-border: #1e3a1e;
  --text-main: #e8f5e9;
  --text-muted: #81c784;
  --accent: #ffeb3b;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: var(--dark-bg) !important;
    color: var(--text-main) !important;
}
.stApp { background: var(--dark-bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a0d 0%, #0a0f0a 100%) !important;
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* Cards */
.agri-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.agri-card:hover { border-color: var(--green-1); }

/* Metric cards */
.metric-box {
    background: linear-gradient(135deg, #0d2010 0%, #0a1a0a 100%);
    border: 1px solid #1e4a1e;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label { font-size: 11px; color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800; color: var(--green-1); }
.metric-sub   { font-size: 11px; color: var(--text-muted); }

/* Header */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff87, #00bcd4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.page-sub { color: #81c784; font-size: 14px; letter-spacing: 1px; }

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--green-1);
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 8px;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
}

/* Chat bubbles */
.chat-user {
    background: #0d2010;
    border: 1px solid #1e4a1e;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0 8px 20%;
    color: #e8f5e9;
    font-size: 14px;
}
.chat-bot {
    background: #0a1a0a;
    border: 1px solid #00ff87;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 20% 8px 0;
    color: #e8f5e9;
    font-size: 14px;
}
.chat-avatar {
    font-size: 20px;
    margin-right: 8px;
}

/* Prediction result */
.pred-result {
    background: linear-gradient(135deg, #003300, #001a00);
    border: 2px solid var(--green-1);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.pred-crop {
    font-family: 'Syne', sans-serif;
    font-size: 38px;
    font-weight: 800;
    color: var(--green-1);
    text-transform: uppercase;
    letter-spacing: 3px;
}
.pred-conf { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

/* Streamlit overrides */
.stSlider > div > div { background: var(--card-border) !important; }
[data-testid="stMetric"] { background: var(--card-bg) !important; border-radius: 10px; padding: 12px; }
.stSelectbox > div > div { background: var(--card-bg) !important; color: var(--text-main) !important; }
div[data-testid="stForm"] { background: var(--card-bg) !important; border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; }
.stButton > button {
    background: linear-gradient(135deg, #00c853, #00ff87) !important;
    color: #001a00 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
.stTextInput > div > div > input {
    background: var(--card-bg) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════
CROP_EMOJI = {
    "rice":"🌾","maize":"🌽","chickpea":"🫘","kidneybeans":"🫘","pigeonpeas":"🌿",
    "mothbeans":"🫘","mungbean":"🫘","blackgram":"🫘","lentil":"🫘","pomegranate":"🍎",
    "banana":"🍌","mango":"🥭","grapes":"🍇","watermelon":"🍉","muskmelon":"🍈",
    "apple":"🍎","orange":"🍊","papaya":"🥭","coconut":"🥥","cotton":"🌸",
    "jute":"🌿","coffee":"☕"
}
FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]


BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "data", "Crop.csv")

df = pd.read_csv(file_path)

@st.cache_resource
def load_models():
   # df = load_data()
    le = LabelEncoder()
    df["enc"] = le.fit_transform(df["label"])
    X = df[FEATURES].values
    y = df["enc"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="mlogloss",
                        random_state=42, n_jobs=-1)

    rf.fit(X_tr, y_tr);   rf_acc  = rf.score(X_te, y_te)
    svm.fit(X_tr_sc, y_tr)  # dummy shortcut
    svm.fit(X_tr_sc, y_tr); svm_acc = svm.score(X_te_sc, y_te)
    xgb.fit(X_tr, y_tr);  xgb_acc = xgb.score(X_te, y_te)

    return {
        "rf": rf, "svm": svm, "xgb": xgb,
        "scaler": scaler, "le": le,
        "accs": {"Random Forest": rf_acc, "SVM": svm_acc, "XGBoost": xgb_acc},
        "X_te": X_te, "X_te_sc": X_te_sc, "y_te": y_te
    }

#df = load_data()
models_data = load_models()
le     = models_data["le"]
scaler = models_data["scaler"]
CROPS  = le.classes_

# ═══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:48px;'>🌾</div>
        <div style='font-family:Syne,sans-serif; font-size:18px; font-weight:800; 
                    color:#00ff87; letter-spacing:1px;'>AGRI ADVISOR</div>
        <div style='font-size:11px; color:#81c784; letter-spacing:2px;'>SMART FARMING AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigate", [
        "🏠  Dashboard",
        "📊  EDA Explorer",
        "🤖  Model Insights",
        "🔮  Crop Predictor",
        "💬  AI Chatbot"
    ], label_visibility="collapsed")
    st.divider()

    # Quick stats
    st.markdown("<div style='font-size:11px; color:#81c784; letter-spacing:2px;'>DATASET QUICK STATS</div>",
                unsafe_allow_html=True)
    best_acc = max(models_data["accs"].values())
    st.markdown(f"""
    <div style='font-size:13px; line-height:2;'>
    🌱 <b>2,200</b> samples<br>
    🏷️ <b>22</b> crop types<br>
    📐 <b>7</b> input features<br>
    🎯 Best accuracy: <span style='color:#00ff87;'><b>{best_acc*100:.1f}%</b></span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("""
    <div class='page-title'>Smart Agriculture<br>Advisor</div>
    <div class='page-sub'>MACHINE LEARNING POWERED CROP INTELLIGENCE SYSTEM</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("2,200", "Training Samples", "Balanced Dataset"),
        ("22", "Crop Classes", "Fruits, Grains & More"),
        ("7", "Input Features", "Soil + Climate"),
        (f"{max(models_data['accs'].values())*100:.1f}%", "Best Accuracy", "Random Forest")
    ]
    for col, (val, label, sub) in zip([col1,col2,col3,col4], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model accuracy cards
    st.markdown("<div class='section-head'>⚡ Model Performance Snapshot</div>", unsafe_allow_html=True)
    mcols = st.columns(3)
    model_info = [
        ("🌳", "Random Forest", models_data["accs"]["Random Forest"],
         "200 trees • Ensemble • Feature importance", "#00ff87"),
        ("🔷", "SVM (RBF)", models_data["accs"]["SVM"],
         "Kernel: RBF • C=10 • Probability calibrated", "#00bcd4"),
        ("⚡", "XGBoost", models_data["accs"]["XGBoost"],
         "200 rounds • Depth 6 • Gradient boosting", "#ffeb3b"),
    ]
    for col, (icon, name, acc, desc, color) in zip(mcols, model_info):
        with col:
            st.markdown(f"""
            <div class='agri-card'>
                <div style='font-size:28px;'>{icon}</div>
                <div style='font-family:Syne,sans-serif; font-size:17px; font-weight:700;
                            color:{color}; margin-top:6px;'>{name}</div>
                <div style='font-size:36px; font-family:Syne,sans-serif; font-weight:800;
                            color:#fff;'>{acc*100:.2f}%</div>
                <div style='font-size:11px; color:#81c784; margin-top:4px;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Crop grid
    st.markdown("<br><div class='section-head'>🌿 22 Supported Crops</div>", unsafe_allow_html=True)
    cols = st.columns(11)
    for i, crop in enumerate(sorted(CROPS)):
        with cols[i % 11]:
            emoji = CROP_EMOJI.get(crop, "🌱")
            st.markdown(f"""
            <div style='text-align:center; background:#101810; border:1px solid #1e3a1e;
                        border-radius:8px; padding:10px 4px; margin-bottom:8px;'>
                <div style='font-size:22px;'>{emoji}</div>
                <div style='font-size:10px; color:#81c784; text-transform:capitalize;
                            font-family:DM Mono,monospace;'>{crop}</div>
            </div>""", unsafe_allow_html=True)

    # Feature overview
    st.markdown("<br><div class='section-head'>📐 Feature Overview</div>", unsafe_allow_html=True)
    feat_info = [
        ("N", "Nitrogen (mg/kg)", df["N"].mean(), df["N"].min(), df["N"].max()),
        ("P", "Phosphorus (mg/kg)", df["P"].mean(), df["P"].min(), df["P"].max()),
        ("K", "Potassium (mg/kg)", df["K"].mean(), df["K"].min(), df["K"].max()),
        ("temperature", "Temperature (°C)", df["temperature"].mean(), df["temperature"].min(), df["temperature"].max()),
        ("humidity", "Humidity (%)", df["humidity"].mean(), df["humidity"].min(), df["humidity"].max()),
        ("ph", "Soil pH", df["ph"].mean(), df["ph"].min(), df["ph"].max()),
        ("rainfall", "Rainfall (mm)", df["rainfall"].mean(), df["rainfall"].min(), df["rainfall"].max()),
    ]
    fc1, fc2 = st.columns(2)
    for idx, (feat, label, mean, mn, mx) in enumerate(feat_info):
        col = fc1 if idx % 2 == 0 else fc2
        with col:
            st.markdown(f"""
            <div style='background:#101810; border:1px solid #1e3a1e; border-radius:8px;
                        padding:12px 16px; margin-bottom:8px; display:flex;
                        justify-content:space-between; align-items:center;'>
                <div>
                    <span style='font-family:Syne,sans-serif; font-weight:700;
                                 color:#00ff87;'>{feat}</span>
                    <span style='color:#81c784; font-size:12px; margin-left:8px;'>{label}</span>
                </div>
                <div style='font-size:12px; color:#ccc;'>
                    μ <b style='color:#fff;'>{mean:.1f}</b>
                    &nbsp;|&nbsp; [{mn:.1f} – {mx:.1f}]
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ═══════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown("<div class='page-title'>EDA Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>EXPLORATORY DATA ANALYSIS — INTERACTIVE CHARTS</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Distributions", "🔥 Correlations", "🌾 Crop Profiles", "📈 Feature Scatter"])

    with tab1:
        feat_sel = st.selectbox("Select Feature", FEATURES,
            format_func=lambda x: {"N":"Nitrogen","P":"Phosphorus","K":"Potassium",
                "temperature":"Temperature","humidity":"Humidity","ph":"pH","rainfall":"Rainfall"}[x])
        fig = px.histogram(df, x=feat_sel, color="label",
                           marginal="violin", nbins=50,
                           color_discrete_sequence=px.colors.qualitative.Vivid,
                           template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0a0f0a", plot_bgcolor="#101810",
                          legend=dict(font=dict(size=9)), height=480)
        st.plotly_chart(fig, use_container_width=True)

        # Boxplot
        fig2 = px.box(df, x="label", y=feat_sel, color="label",
                      color_discrete_sequence=px.colors.qualitative.Vivid,
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0a0f0a", plot_bgcolor="#101810",
                           showlegend=False, height=380,
                           xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        corr = df[FEATURES].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=FEATURES, y=FEATURES,
            colorscale="RdYlGn", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Corr: %{z:.3f}<extra></extra>"
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0f0a",
                          plot_bgcolor="#101810", height=480,
                          title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

        # Parallel coords
        df_sample = df.sample(400, random_state=1)
        df_sample["crop_id"] = le.transform(df_sample["label"])
        fig2 = px.parallel_coordinates(df_sample, dimensions=FEATURES,
                                        color="crop_id",
                                        color_continuous_scale=px.colors.sequential.Viridis,
                                        template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0a0f0a", height=400,
                           title="Parallel Coordinates — All Features")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        profile = df.groupby("label")[FEATURES].mean().reset_index()
        crops_sel = st.multiselect("Select crops to compare", sorted(CROPS),
                                    default=["rice","maize","wheat"] if "wheat" in CROPS else list(sorted(CROPS))[:5])
        if crops_sel:
            sub = profile[profile["label"].isin(crops_sel)]
            feat_radar = st.multiselect("Features for radar", FEATURES, default=["N","P","K","temperature","humidity"])
            if feat_radar:
                fig = go.Figure()
                for _, row in sub.iterrows():
                    norm_vals = []
                    for f in feat_radar:
                        rng = df[f].max() - df[f].min()
                        norm_vals.append((row[f] - df[f].min()) / rng if rng > 0 else 0)
                    norm_vals.append(norm_vals[0])
                    labels = feat_radar + [feat_radar[0]]
                    fig.add_trace(go.Scatterpolar(
                        r=norm_vals, theta=labels,
                        fill="toself", name=row["label"],
                        line=dict(width=2)
                    ))
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0f0a",
                                   polar=dict(bgcolor="#101810"), height=500,
                                   title="Normalized Crop Soil & Climate Profile")
                st.plotly_chart(fig, use_container_width=True)

            # NPK grouped bar
            npk_data = []
            for _, row in sub.iterrows():
                for f in ["N","P","K"]:
                    npk_data.append({"Crop": row["label"], "Nutrient": f, "Value": row[f]})
            fig2 = px.bar(pd.DataFrame(npk_data), x="Crop", y="Value", color="Nutrient",
                          barmode="group", color_discrete_map={"N":"#e74c3c","P":"#3498db","K":"#2ecc71"},
                          template="plotly_dark")
            fig2.update_layout(paper_bgcolor="#0a0f0a", plot_bgcolor="#101810", height=350,
                                title="NPK Requirements by Crop")
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            x_feat = st.selectbox("X-axis", FEATURES, index=3, key="sx")
        with c2:
            y_feat = st.selectbox("Y-axis", FEATURES, index=4, key="sy")

        sample_400 = df.sample(600, random_state=7)
        fig = px.scatter(sample_400, x=x_feat, y=y_feat, color="label",
                         hover_data=FEATURES,
                         color_discrete_sequence=px.colors.qualitative.Vivid,
                         template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0a0f0a", plot_bgcolor="#101810", height=520,
                           title=f"{x_feat} vs {y_feat} — 600 sample scatter")
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("<div class='page-title'>Model Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>3-MODEL BREAKDOWN — RANDOM FOREST · SVM · XGBOOST</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Accuracy comparison
    st.markdown("<div class='section-head'>🏆 Accuracy Comparison</div>", unsafe_allow_html=True)
    accs = models_data["accs"]
    fig = go.Figure()
    colors = ["#00ff87","#00bcd4","#ffeb3b"]
    for (name, acc), color in zip(accs.items(), colors):
        fig.add_trace(go.Bar(name=name, x=[name], y=[acc*100],
                             marker_color=color, text=[f"{acc*100:.2f}%"],
                             textposition="outside", width=0.4))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0f0a",
                      plot_bgcolor="#101810", height=320, showlegend=False,
                      yaxis=dict(range=[95, 101]), yaxis_title="Test Accuracy (%)",
                      bargroupgap=0.3)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("<div class='section-head'>🔍 Feature Importance</div>", unsafe_allow_html=True)
    rf_imp  = models_data["rf"].feature_importances_
    xgb_imp = models_data["xgb"].feature_importances_

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Random Forest", x=FEATURES, y=rf_imp,
                          marker_color="#00ff87", opacity=0.85))
    fig2.add_trace(go.Bar(name="XGBoost",       x=FEATURES, y=xgb_imp,
                          marker_color="#ffeb3b", opacity=0.85))
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0a0f0a",
                       plot_bgcolor="#101810", height=360,
                       barmode="group", yaxis_title="Importance Score",
                       title="Feature Importance: RF vs XGBoost")
    st.plotly_chart(fig2, use_container_width=True)

    # Code breakdown section
    st.markdown("<div class='section-head'>📋 Code Architecture Breakdown</div>", unsafe_allow_html=True)

    sections = [
        ("🔢 Data Loading & Preprocessing", "#0d2010",
         """df = pd.read_csv("Crop.csv")
le = LabelEncoder()
df["crop_encoded"] = le.fit_transform(df["label"])

X = df[["N","P","K","temperature","humidity","ph","rainfall"]].values
y = df["crop_encoded"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)"""),
        ("🌳 Random Forest", "#0d1a0d",
         """rf = RandomForestClassifier(
    n_estimators=200,    # 200 decision trees
    max_depth=None,      # Trees grow until pure
    min_samples_split=2,
    random_state=42,
    n_jobs=-1            # Use all CPU cores
)
rf.fit(X_train, y_train)
# Test accuracy: 99.55%"""),
        ("🔷 Support Vector Machine", "#001a2d",
         """svm = SVC(
    kernel="rbf",        # Radial Basis Function kernel
    C=10,                # Regularization parameter
    gamma="scale",       # Kernel coefficient
    probability=True,    # Enable probability output
    random_state=42
)
svm.fit(X_train_sc, y_train)  # Needs scaled input!
# Test accuracy: 98.86%"""),
        ("⚡ XGBoost", "#1a1400",
         """xgb = XGBClassifier(
    n_estimators=200,    # 200 boosting rounds
    max_depth=6,         # Tree depth per round
    learning_rate=0.1,   # Shrinkage factor
    subsample=0.8,       # Row sampling
    colsample_bytree=0.8,# Column sampling
    eval_metric="mlogloss",
    random_state=42
)
xgb.fit(X_train, y_train)
# Test accuracy: 99.32%"""),
        ("💾 Save & Load with Joblib", "#0d0d1a",
         """# Save models
joblib.dump(rf,     "model_random_forest.joblib")
joblib.dump(svm,    "model_svm_rbf.joblib")
joblib.dump(xgb,    "model_xgboost.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le,     "label_encoder.joblib")

# Load and predict
model  = joblib.load("model_random_forest.joblib")
scaler = joblib.load("scaler.joblib")
le     = joblib.load("label_encoder.joblib")

sample = np.array([[90,42,43,20.9,82,6.5,202.9]])
pred   = le.inverse_transform(model.predict(sample))
print(pred)  # → ['rice']"""),
    ]
    for title, bg, code in sections:
        with st.expander(title, expanded=False):
            st.code(code, language="python")

# ═══════════════════════════════════════════════════════════
# PAGE 4 — CROP PREDICTOR
# ═══════════════════════════════════════════════════════════
elif "Predictor" in page:
    st.markdown("<div class='page-title'>Crop Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>INPUT SOIL & CLIMATE CONDITIONS → GET CROP RECOMMENDATION</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='section-head'>🌡️ Soil & Climate Inputs</div>", unsafe_allow_html=True)

        with st.container():
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                N = st.slider("🧪 Nitrogen (N)", 0, 140, 80, help="Nitrogen content in soil (mg/kg)")
                K = st.slider("🟡 Potassium (K)", 5, 210, 40, help="Potassium content in soil (mg/kg)")
                humidity = st.slider("💧 Humidity (%)", 10.0, 100.0, 70.0, step=0.5)
                rainfall = st.slider("🌧️ Rainfall (mm)", 20.0, 300.0, 150.0, step=1.0)
            with r1c2:
                P = st.slider("🔵 Phosphorus (P)", 5, 145, 50, help="Phosphorus content in soil (mg/kg)")
                temperature = st.slider("🌡️ Temperature (°C)", 8.0, 44.0, 25.0, step=0.1)
                ph = st.slider("⚗️ Soil pH", 3.5, 10.0, 6.5, step=0.05)

        model_choice = st.selectbox("🤖 Choose Model", ["Ensemble (Vote)", "Random Forest", "SVM (RBF)", "XGBoost"])
        predict_btn = st.button("🔮 Predict Crop", use_container_width=True)

    with col_right:
        st.markdown("<div class='section-head'>📊 Your Input Profile</div>", unsafe_allow_html=True)
        input_arr = [N, P, K, temperature, humidity, ph, rainfall]
        profile_df = df.groupby("label")[FEATURES].mean()

        fig_radar = go.Figure()
        # Normalize user input
        norm_user = []
        for i, f in enumerate(FEATURES):
            rng = df[f].max() - df[f].min()
            norm_user.append((input_arr[i] - df[f].min()) / rng if rng > 0 else 0)
        norm_user_plot = norm_user + [norm_user[0]]
        labels_plot = FEATURES + [FEATURES[0]]
        fig_radar.add_trace(go.Scatterpolar(r=norm_user_plot, theta=labels_plot,
                                             fill="toself", name="Your Input",
                                             line=dict(color="#00ff87", width=2)))
        fig_radar.update_layout(template="plotly_dark", paper_bgcolor="#0a0f0a",
                                  polar=dict(bgcolor="#101810"), height=320,
                                  margin=dict(t=30, b=10))
        st.plotly_chart(fig_radar, use_container_width=True)

        if predict_btn:
            sample = np.array(input_arr).reshape(1, -1)
            sample_sc = scaler.transform(sample)

            rf_pred  = CROPS[models_data["rf"].predict(sample)[0]]
            svm_pred = CROPS[models_data["svm"].predict(sample_sc)[0]]
            xgb_pred = CROPS[models_data["xgb"].predict(sample)[0]]

            if model_choice == "Ensemble (Vote)":
                votes = [rf_pred, svm_pred, xgb_pred]
                final = max(set(votes), key=votes.count)
            elif model_choice == "Random Forest":
                final = rf_pred
            elif model_choice == "SVM (RBF)":
                final = svm_pred
            else:
                final = xgb_pred

            emoji = CROP_EMOJI.get(final, "🌱")
            st.markdown(f"""
            <div class='pred-result'>
                <div style='font-size:56px;'>{emoji}</div>
                <div class='pred-crop'>{final}</div>
                <div class='pred-conf'>RECOMMENDED CROP</div>
                <div style='margin-top:16px; font-size:12px; color:#81c784;'>
                    RF: {rf_pred} &nbsp;|&nbsp; SVM: {svm_pred} &nbsp;|&nbsp; XGB: {xgb_pred}
                </div>
            </div>""", unsafe_allow_html=True)

            # Show similar crops from dataset
            st.markdown("<br><div style='font-size:11px; color:#81c784;'>NEAREST CROP PROFILE FROM TRAINING DATA</div>",
                        unsafe_allow_html=True)
            crop_means = profile_df.loc[final]
            compare_df = pd.DataFrame({
                "Feature": FEATURES,
                "Your Input": input_arr,
                "Crop Average": crop_means.values
            })
            compare_df["Δ Difference"] = (compare_df["Your Input"] - compare_df["Crop Average"]).round(2)
            st.dataframe(compare_df.set_index("Feature"), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 5 — AI CHATBOT
# ═══════════════════════════════════════════════════════════
elif "Chatbot" in page:
    st.markdown("<div class='page-title'>Agri AI Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>POWERED BY CLAUDE — ASK ANYTHING ABOUT CROPS, SOIL & FARMING</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # API key input
    with st.sidebar:
        st.markdown("---")
        st.markdown("<div style='font-size:11px; color:#81c784; letter-spacing:2px;'>CLAUDE API KEY</div>",
                    unsafe_allow_html=True)
        api_key_input = st.text_input("", type="password", placeholder="sk-ant-...",
                                       value=st.session_state.api_key, label_visibility="collapsed")
        if api_key_input:
            st.session_state.api_key = api_key_input
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    SYSTEM_PROMPT = """You are an expert Smart Agriculture Advisor AI. You specialize in:
- Crop recommendation based on soil (N, P, K, pH) and climate (temperature, humidity, rainfall)
- Soil health analysis and improvement strategies
- Crop disease identification and treatment
- Irrigation and water management
- Fertilizer recommendations
- Seasonal farming advice
- The 22 crops in this dataset: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, 
  mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, 
  apple, orange, papaya, coconut, cotton, jute, coffee.

When answering:
- Be practical, specific, and helpful
- Use bullet points for lists
- Mention soil/climate values when relevant
- Keep responses concise but informative
- Use emojis sparingly for readability"""

    # Suggested questions
    st.markdown("<div class='section-head'>💡 Suggested Questions</div>", unsafe_allow_html=True)
    suggestions = [
        "What crops grow best in high humidity and low nitrogen soil?",
        "How do I improve soil pH for apple cultivation?",
        "Compare rice vs maize soil requirements",
        "What is the ideal rainfall for growing coffee?",
        "How can I identify nitrogen deficiency in crops?",
    ]
    s_cols = st.columns(3)
    for i, sugg in enumerate(suggestions[:3]):
        with s_cols[i]:
            if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": sugg})

    st.markdown("<div class='section-head'>💬 Conversation</div>", unsafe_allow_html=True)

    # Display chat
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style='text-align:center; padding:40px; color:#81c784;'>
                <div style='font-size:48px;'>🌱</div>
                <div style='font-size:16px; margin-top:12px;'>
                    Hello! I'm your Smart Agriculture AI.<br>
                    Ask me anything about crops, soil, or farming!
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class='chat-user'>
                        <span class='chat-avatar'>👤</span>{msg['content']}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-bot'>
                        <span class='chat-avatar'>🌾</span>{msg['content']}
                    </div>""", unsafe_allow_html=True)

    # Check if we need to generate a response
    if (st.session_state.messages and
        st.session_state.messages[-1]["role"] == "user"):
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your Claude API key in the sidebar to enable the chatbot.")
        else:
            with st.spinner("🌾 Thinking..."):
                try:
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": st.session_state.api_key,
                            "anthropic-version": "2023-06-01"
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 1024,
                            "system": SYSTEM_PROMPT,
                            "messages": [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ]
                        },
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        reply = data["content"][0]["text"]
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                    else:
                        st.error(f"API Error {response.status_code}: {response.text[:200]}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # Chat input
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        fc1, fc2 = st.columns([5, 1])
        with fc1:
            user_input = st.text_input("", placeholder="Ask about crops, soil health, climate conditions...",
                                        label_visibility="collapsed")
        with fc2:
            send = st.form_submit_button("Send 🌾", use_container_width=True)

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()