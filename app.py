
import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==========================================================
# NLTK
# ==========================================================

nltk.download("stopwords", quiet=True)

# ==========================================================
# PLOTLY THEME
# ==========================================================

pio.templates.default = "plotly_dark"

# ==========================================================
# COLOR PALETTE (PRESERVED & ENHANCED)
# ==========================================================

POSITIVE_COLOR = "#22c55e"
NEGATIVE_COLOR = "#ef4444"
PRIMARY_COLOR = "#6366f1"
SECONDARY_COLOR = "#8b5cf6"

BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
TEXT_COLOR = "#ffffff"

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# MODERN & STYLISH UI CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Global Font Override & App Base Background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #0f172a !important;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

/* Header Visibility Tweaks */
h2, h3, h4, h5, h6, label, p, span {
    color: #ffffff !important;
}

.stApp h2 {
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    margin-top: 1.5rem !important;
}

/* Premium Glassmorphic Hero Banner */
.hero {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 50%, rgba(236, 72, 153, 0.15) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    padding: 45px 35px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
}

.hero h1 {
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 12px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.05em;
}

.hero p {
    font-size: 1.2rem;
    color: #94a3b8 !important;
    font-weight: 500;
}

/* Modernized Feature List Cards */
.feature-item {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-weight: 500;
    transition: transform 0.2s ease;
}
.feature-item:hover {
    transform: translateX(4px);
    border-color: rgba(99, 102, 241, 0.3);
}

/* About Model Card Sidebar / Right Side Block */
.glass {
    background: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.glass h3 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
}
.glass p {
    color: #94a3b8 !important;
    line-height: 1.6;
}

/* Styled Inputs & Textareas */
[data-testid="stTextarea"] textarea {
    background-color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    padding: 16px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stTextarea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
}
[data-testid="stWidgetLabel"] p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}

/* Neon Gradient Animated Submission Button */
.stButton > button {
    width: 100%;
    height: 56px;
    border: none !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.6) !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
}

/* High Contrast Premium Sentiment Outcome Badges */
.result-positive {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    padding: 24px;
    border-radius: 20px;
    text-align: center;
    color: white !important;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.02em;
    box-shadow: 0 15px 30px -10px rgba(16, 185, 129, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.result-negative {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    padding: 24px;
    border-radius: 20px;
    text-align: center;
    color: white !important;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.02em;
    box-shadow: 0 15px 30px -10px rgba(239, 68, 68, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* KPI Metric Grid Styling Updates */
[data-testid="stMetric"] {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 20px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    letter-spacing: -0.03em !important;
}

/* Beautiful Modern Containers for Sample Data */
div.stAlert {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 18px !important;
    background-color: #1e293b !important;
}
div.stAlert p {
    font-weight: 500 !important;
}

/* Footer Element */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 60px;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🤖 AI Dashboard")

    st.markdown("---")

    st.markdown("### Features")
    
    st.markdown("""
    <div class="feature-item">✅ NLP Processing</div>
    <div class="feature-item">✅ Sentiment Analysis</div>
    <div class="feature-item">✅ Confidence Score</div>
    <div class="feature-item">✅ Interactive Charts</div>
    <div class="feature-item">✅ Prediction History</div>
    <div class="feature-item">✅ Machine Learning</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Version 4.0")

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("cv.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# ==========================================================
# NLP PREPROCESSING
# ==========================================================

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):

    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower()
    review = review.split()

    review = [
        stemmer.stem(word)
        for word in review
        if word not in stop_words
    ]

    return " ".join(review)

# ==========================================================
# SESSION HISTORY
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# HERO
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>🤖 AI Sentiment Analyzer</h1>
    <p>Analyze Amazon Alexa Reviews using Machine Learning & NLP Pipelines</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# INPUT SECTION
# ==========================================================

left, right = st.columns([3,1])

with left:

    review = st.text_area(
        "📝 Enter Customer Review",
        height=220,
        placeholder="Example: Alexa is fantastic, the sound quality is excellent..."
    )

    analyze = st.button("🚀 Analyze Sentiment")

with right:

    st.markdown("""
    <div class="glass">
        <h3>📊 About Model</h3>
        <hr style="border-color: rgba(255,255,255,0.1); margin-bottom: 15px;">
        <p>
        This optimized machine learning model predicts whether an incoming customer voice transcript or review exhibits a Positive or Negative sentiment baseline.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# PREDICTION
# ==========================================================

if analyze:

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    processed = preprocess_text(review)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    probs = model.predict_proba(vector)[0]

    negative_score = probs[0] * 100
    positive_score = probs[1] * 100

    confidence = max(
        positive_score,
        negative_score
    )

    st.session_state.history.append({
        "Positive": positive_score,
        "Negative": negative_score
    })

    st.markdown("---")
    st.subheader("📈 Analysis Result")

    if prediction == 0:

        st.markdown("""
        <div class="result-positive">
            😊 POSITIVE SENTIMENT MATCHED
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    else:

        st.markdown("""
        <div class="result-negative">
            😞 NEGATIVE SENTIMENT MATCHED
        </div>
        """, unsafe_allow_html=True)

    # ======================================================
    # METRICS
    # ======================================================

    st.markdown("## 📊 Key Metrics")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("😊 Positive", f"{positive_score:.2f}%")
    k2.metric("😞 Negative", f"{negative_score:.2f}%")
    k3.metric("🎯 Confidence", f"{confidence:.2f}%")
    k4.metric("📝 Words", len(review.split()))
    k5.metric("🔠 Characters", len(review))

    # ======================================================
    # CHART DATA
    # ======================================================

    chart_df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative"],
        "Score": [positive_score, negative_score]
    })

    st.markdown("---")
    st.subheader("📊 Analytics Dashboard")

    # ======================================================
    # GAUGE CHART
    # ======================================================

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={
                "suffix": "%",
                "font": {
                    "size": 42,
                    "color": TEXT_COLOR,
                    "family": "Plus Jakarta Sans"
                }
            },
            title={
                "text": "<b>Model Confidence</b>",
                "font": {"size": 16, "color": "#94a3b8", "family": "Plus Jakarta Sans"}
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {
                    "color": PRIMARY_COLOR,
                    "thickness": 0.35
                },
                "bgcolor": "rgba(255,255,255,0.03)",
                "steps": [
                    {"range": [0, 50], "color": "#ef4444"},
                    {"range": [50, 75], "color": "#f59e0b"},
                    {"range": [75, 100], "color": "#10b981"}
                ]
            }
        )
    )

    gauge_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_COLOR, "family": "Plus Jakarta Sans"},
        margin=dict(t=40, b=10, l=30, r=30),
        height=300
    )

    # ======================================================
    # DONUT CHART
    # ======================================================

    donut_fig = px.pie(
        chart_df,
        names="Sentiment",
        values="Score",
        hole=0.72,
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={
            "Positive": POSITIVE_COLOR,
            "Negative": NEGATIVE_COLOR
        }
    )

    donut_fig.update_traces(
        textinfo="percent+label",
        hoverinfo="label+percent",
        marker=dict(
            line=dict(
                color=BG_COLOR,
                width=3
            )
        )
    )

    donut_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Plus Jakarta Sans"),
        margin=dict(t=50, b=10, l=10, r=10),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )

    # ======================================================
    # BAR CHART
    # ======================================================

    bar_fig = px.bar(
        chart_df,
        x="Sentiment",
        y="Score",
        text="Score",
        title="Sentiment Comparison Breakdown",
        color="Sentiment",
        color_discrete_map={
            "Positive": POSITIVE_COLOR,
            "Negative": NEGATIVE_COLOR
        }
    )

    bar_fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_width=0,
        width=0.4
    )

    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=TEXT_COLOR, family="Plus Jakarta Sans"),
        showlegend=False,
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, title="Percentage Score"),
        transition={
            "duration": 800
        }
    )

    # ======================================================
    # DISPLAY CHARTS INSIDE CARDS
    # ======================================================

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.plotly_chart(bar_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ======================================================
    # HISTORY CHART
    # ======================================================

    if len(st.session_state.history) > 1:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        line_fig = px.line(
            history_df,
            y=["Positive", "Negative"],
            markers=True,
            title="Session Historical Tracking"
        )

        line_fig.data[0].line.color = POSITIVE_COLOR
        line_fig.data[1].line.color = NEGATIVE_COLOR

        line_fig.update_traces(
            line=dict(width=4),
            marker=dict(size=10, line=dict(width=2, color=CARD_COLOR))
        )

        line_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color=TEXT_COLOR, family="Plus Jakarta Sans"),
            hovermode="x unified",
            height=350,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis=dict(showgrid=False, title="Prediction Instance Runs"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Confidence Values"),
            legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.plotly_chart(line_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================================================
    # PROCESSED TEXT
    # ======================================================
    st.markdown('<br>', unsafe_allow_html=True)
    with st.expander("🔍 Processed Tokens Output"):
        st.code(processed)

# ==========================================================
# SAMPLE REVIEWS
# ==========================================================

st.markdown("---")
st.subheader("💡 Sample Reviews")

s1, s2 = st.columns(2)

with s1:
    st.success(
        "I absolutely love this Alexa device. Amazing sound quality."
    )

with s2:
    st.error(
        "Worst product ever. Completely disappointed."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit • Plotly Engine • Advanced NLP Pipeline • Scikit-Learn Ecosystem
</div>
""", unsafe_allow_html=True)
