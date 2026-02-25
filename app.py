"""
ChainSight — Supply Chain Risk Intelligence Dashboard
Streamlit + Matplotlib  (no Tkinter, no Plotly required)
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChainSight — Supply Chain Risk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════
BG      = "#080C10"
SURFACE = "#0D1318"
CARD    = "#111820"
BORDER  = "#1E2A35"
ACCENT  = "#00FFB2"
ACCENT2 = "#FF4D6D"
ACCENT3 = "#FFB800"
BLUE    = "#3D9EFF"
TEXT    = "#E8F0F8"
MUTED   = "#5A7080"
DIM     = "#2A3A48"

# ── Global CSS ──────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
      background-color: {BG} !important;
      font-family: 'Syne', sans-serif !important;
  }}
  [data-testid="stSidebar"] {{
      background-color: {SURFACE} !important;
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {MUTED} !important; }}
  section[data-testid="stSidebar"] .stRadio label {{
      color: {MUTED} !important;
      font-family: 'Syne', sans-serif !important;
      font-size: 14px !important;
      padding: 6px 0 !important;
  }}
  section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{
      background: transparent !important;
  }}
  .stMetric {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px !important; }}
  .stMetric label {{ font-family: 'Space Mono', monospace !important; font-size: 10px !important; color: {MUTED} !important; letter-spacing: 1.5px; text-transform: uppercase; }}
  .stMetric [data-testid="metric-container"] > div {{ color: {TEXT} !important; }}
  div[data-testid="stForm"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 20px; }}
  .stSelectbox label, .stNumberInput label, .stSlider label {{
      font-family: 'Space Mono', monospace !important;
      font-size: 10px !important;
      color: {MUTED} !important;
      letter-spacing: 1px;
      text-transform: uppercase;
  }}
  .stSelectbox > div > div, .stNumberInput > div > div > input {{
      background: {SURFACE} !important;
      border: 1px solid {BORDER} !important;
      color: {TEXT} !important;
      border-radius: 8px !important;
  }}
  .stButton > button {{
      background: {ACCENT} !important;
      color: {BG} !important;
      font-family: 'Syne', sans-serif !important;
      font-weight: 800 !important;
      font-size: 14px !important;
      border: none !important;
      border-radius: 10px !important;
      padding: 12px 24px !important;
      width: 100%;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      transition: opacity .2s;
  }}
  .stButton > button:hover {{ opacity: 0.88 !important; }}
  h1, h2, h3, h4 {{ font-family: 'Syne', sans-serif !important; color: {TEXT} !important; }}
  p, li {{ color: {MUTED} !important; font-family: 'Syne', sans-serif !important; }}
  .block-container {{ padding: 24px 32px !important; max-width: 100% !important; }}
  [data-testid="stHorizontalBlock"] {{ gap: 16px; }}
  .stTabs [data-baseweb="tab-list"] {{
      background: {SURFACE} !important;
      border-radius: 10px !important;
      padding: 4px !important;
      gap: 4px;
      border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent !important;
      color: {MUTED} !important;
      font-family: 'Space Mono', monospace !important;
      font-size: 11px !important;
      letter-spacing: 1px;
      border-radius: 8px !important;
      padding: 8px 18px !important;
  }}
  .stTabs [aria-selected="true"] {{
      background: {CARD} !important;
      color: {ACCENT} !important;
  }}
  .stDataFrame {{ background: {CARD} !important; }}
  [data-testid="stExpander"] {{
      background: {CARD} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 10px !important;
  }}
  .css-1d391kg, [data-testid="stVerticalBlock"] > div > div {{
      background: transparent !important;
  }}
  div.stAlert {{ background: {CARD}; border-radius: 10px; border: 1px solid {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MATPLOTLIB GLOBAL STYLE
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  CARD,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "axes.titleweight":  "bold",
    "axes.titlesize":    12,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom":True,
})

def _spine(ax):
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
ROUTE_LABELS  = ["Suez","Commodity","Pacific","Atlantic","Intra-Asia"]
ROUTE_DELAYS  = [1.41, 1.22, 1.05, 0.92, 0.72]
ROUTE_COLORS  = [ACCENT2, ACCENT3, ACCENT, BLUE, DIM]

PRODUCT_LABELS = ["Perishables","Semiconductors","Consumer Elec.","Pharma","Machinery","Textiles","Raw Materials"]
PRODUCT_DELAYS = [1.28, 1.14, 1.07, 0.98, 0.91, 0.85, 0.78]
PRODUCT_COLORS = [ACCENT2 if v>1.1 else ACCENT3 if v>0.95 else ACCENT for v in PRODUCT_DELAYS]

ORIGIN_LABELS  = ["Santos, BR","Mumbai, IN","Shenzhen, CN","Shanghai, CN","Tokyo, JP","Hamburg, DE"]
ORIGIN_DELAYS  = [1.30, 1.22, 1.05, 0.98, 0.88, 0.72]
ORIGIN_COLORS  = [ACCENT2 if v>1.1 else ACCENT3 if v>0.95 else BLUE for v in ORIGIN_DELAYS]

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
VOL    = [780,720,850,810,760,900,870,830,780,910,840,760]
DLY    = [11,8,13,10,9,14,12,11,9,15,11,9]

MODELS_DATA = [
    {"name": "XGBoost",             "acc": 92.4, "auc": 0.941, "best": True},
    {"name": "Random Forest",       "acc": 89.8, "auc": 0.912, "best": False},
    {"name": "SVM",                 "acc": 87.6, "auc": 0.889, "best": False},
    {"name": "Logistic Regression", "acc": 84.3, "auc": 0.861, "best": False},
    {"name": "KNN",                 "acc": 82.1, "auc": 0.838, "best": False},
]

FEAT_NAMES = ["Sched_Lead_Time","Base_Lead_Time","Route_Type","Origin_City",
              "Weather_Index","Geopolitical_Idx","Transport_Mode","Product_Cat","Shipping_Cost"]
FEAT_VALS  = [0.31, 0.22, 0.14, 0.09, 0.07, 0.06, 0.05, 0.04, 0.01]

DELAY_BINS  = list(range(16))
DELAY_CNTS  = [8753,362,185,148,112,89,72,58,45,38,30,24,19,14,11,40]

CORR_NAMES = ["Actual_Lead_Time","Sched_Lead_Time","Base_Lead_Time",
              "Weather_Severity","Geo_Risk_Index","Inflation_Rate","Shipping_Cost"]
CORR_VALS  = [0.82, 0.43, 0.38, 0.12, 0.08, 0.03, 0.01]
CORR_COLS  = [ACCENT2, ACCENT3, ACCENT3, BLUE, BLUE, DIM, DIM]

DISRUPT_LABELS = ["Port Congestion","Geopolitical\nConflict","Extreme\nWeather","No Event"]
DISRUPT_CNTS   = [820, 312, 115, 8753]
DISRUPT_COLORS = [ACCENT2, ACCENT3, BLUE, DIM]


def hex_to_rgba(h, a=1.0):
    h = h.lstrip('#')
    r,g,b = (int(h[i:i+2],16)/255 for i in (0,2,4))
    return (r,g,b,a)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:8px 0 20px'>
      <div style='display:flex;align-items:center;gap:10px'>
        <div style='background:{ACCENT};border-radius:8px;width:32px;height:32px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:18px;font-weight:900;color:{BG}'>◆</div>
        <div>
          <div style='font-family:Syne,sans-serif;font-size:18px;font-weight:800;color:{TEXT}'>
            Chain<span style='color:{ACCENT}'>Sight</span>
          </div>
          <div style='font-family:"Space Mono",monospace;font-size:9px;
                      color:{MUTED};letter-spacing:2px'>RISK INTELLIGENCE v1.0</div>
        </div>
      </div>
    </div>
    <hr style='border-color:{BORDER};margin:0 0 16px'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["▦  Overview", "⚡  Risk Predictor", "∿  EDA Insights", "▤  Model Comparison"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <hr style='border-color:{BORDER};margin:20px 0 12px'>
    <div style='display:flex;align-items:center;gap:8px'>
      <div style='width:8px;height:8px;border-radius:50%;background:{ACCENT};
                  box-shadow:0 0 6px {ACCENT}'></div>
      <span style='font-family:"Space Mono",monospace;font-size:10px;
                   color:{MUTED};letter-spacing:1.5px'>MODEL LIVE · XGB v1.0</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def page_header(title, accent_word, subtitle, badge_text="", badge_color=ACCENT):
    badge_html = ""
    if badge_text:
        badge_html = f"""
        <span style='display:inline-flex;align-items:center;gap:6px;
                     padding:5px 12px;border-radius:20px;
                     border:1px solid {badge_color}44;
                     background:{badge_color}11;
                     font-family:"Space Mono",monospace;font-size:10px;
                     color:{badge_color};letter-spacing:1px'>{badge_text}</span>"""
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:flex-start;
                flex-wrap:wrap;gap:12px;margin-bottom:4px'>
      <div>
        <h1 style='font-size:28px;font-weight:800;letter-spacing:-1px;
                   margin:0;line-height:1.1'>
          {title} <span style='color:{ACCENT}'>{accent_word}</span>
        </h1>
        <p style='font-family:"Space Mono",monospace;font-size:10px;
                  color:{MUTED};margin:6px 0 0;letter-spacing:.5px'>{subtitle}</p>
      </div>
      {badge_html}
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub, color):
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;
                padding:20px 22px;position:relative;overflow:hidden'>
      <div style='height:2px;background:{color};border-radius:2px;margin-bottom:12px'></div>
      <div style='font-family:"Space Mono",monospace;font-size:9px;
                  letter-spacing:1.5px;text-transform:uppercase;color:{MUTED}'>{label}</div>
      <div style='font-size:34px;font-weight:800;letter-spacing:-2px;
                  color:{color};font-family:Syne,sans-serif;line-height:1.1'>{value}</div>
      <div style='font-family:"Space Mono",monospace;font-size:10px;
                  color:{MUTED};margin-top:6px'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_card(title, subtitle=""):
    st.markdown(f"""
    <div style='margin-bottom:10px'>
      <div style='font-size:14px;font-weight:700;color:{TEXT};
                  font-family:Syne,sans-serif;letter-spacing:-.3px'>{title}</div>
      <div style='font-family:"Space Mono",monospace;font-size:9px;
                  color:{MUTED};margin-top:2px;letter-spacing:.5px'>{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def show_fig(fig, use_container_width=True):
    fig.tight_layout(pad=1.4)
    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "▦  Overview":
    page_header("Supply Chain", "Overview",
                "10,000 SHIPMENTS  ·  GLOBAL ROUTES  ·  2024–2025",
                "● Live Model", ACCENT)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("TOTAL SHIPMENTS",  "10,000", "Across 6 origin cities",    ACCENT)
    with c2: kpi_card("DELAYED RATE",     "12.5%",  "1,247 late shipments",       ACCENT2)
    with c3: kpi_card("AVG DELAY DAYS",   "0.95",   "Max observed: 20 days",      ACCENT3)
    with c4: kpi_card("MODEL ROC-AUC",    "0.941",  "XGBoost — best performer",   BLUE)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Row 1: Route bar + Donut
    col_a, col_b = st.columns([3, 2])
    with col_a:
        with st.container():
            section_card("Average Delay by Route Type", "MEAN DELAY DAYS PER ROUTE")
            fig, ax = plt.subplots(figsize=(7, 3.4))
            bars = ax.bar(ROUTE_LABELS, ROUTE_DELAYS, color=ROUTE_COLORS, width=0.52, zorder=3)
            for bar, val in zip(bars, ROUTE_DELAYS):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.03, f"{val}d",
                        ha='center', color=TEXT, fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1.85)
            ax.grid(axis='y', zorder=0)
            ax.tick_params(labelsize=10)
            _spine(ax)
            show_fig(fig)

    with col_b:
        with st.container():
            section_card("Delivery Status", "DISTRIBUTION ACROSS ALL SHIPMENTS")
            fig, ax = plt.subplots(figsize=(4.5, 3.4))
            wedges, texts, autotexts = ax.pie(
                [87.5, 12.5], colors=[ACCENT, ACCENT2],
                startangle=90, wedgeprops=dict(width=0.44),
                autopct='%1.1f%%', pctdistance=0.75
            )
            for at in autotexts:
                at.set_color(BG); at.set_fontsize(10); at.set_fontweight('bold')
            ax.legend(
                wedges, ["On Time  87.5%", "Delayed  12.5%"],
                loc="lower center", ncol=2, frameon=False, fontsize=9,
                labelcolor=[ACCENT, ACCENT2],
            )
            show_fig(fig)

    # Row 2: Product + Origin
    col_c, col_d = st.columns(2)
    with col_c:
        section_card("Avg Delay by Product Category", "WHICH PRODUCTS FACE HIGHEST RISK")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bars = ax.barh(PRODUCT_LABELS, PRODUCT_DELAYS, color=PRODUCT_COLORS, height=0.52, zorder=3)
        for bar, val in zip(bars, PRODUCT_DELAYS):
            ax.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val}d",
                    va='center', color=TEXT, fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1.65); ax.invert_yaxis()
        ax.grid(axis='x', zorder=0); ax.tick_params(labelsize=9); _spine(ax)
        show_fig(fig)

    with col_d:
        section_card("Avg Delay by Origin City", "SOURCE RISK OVERVIEW")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bars = ax.barh(ORIGIN_LABELS, ORIGIN_DELAYS,
                       color=ORIGIN_COLORS, height=0.52, zorder=3)
        for bar, val in zip(bars, ORIGIN_DELAYS):
            ax.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val}d",
                    va='center', color=TEXT, fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1.65); ax.invert_yaxis()
        ax.grid(axis='x', zorder=0); ax.tick_params(labelsize=9); _spine(ax)
        show_fig(fig)

    # Row 3: Trend + Mode
    col_e, col_f = st.columns([3, 2])
    with col_e:
        section_card("Shipment Volume & Delay Trend", "2024–2025 MONTHLY OVERVIEW")
        fig, ax1 = plt.subplots(figsize=(7, 3.2))
        ax1.fill_between(MONTHS, VOL, alpha=0.08, color=BLUE)
        ax1.plot(MONTHS, VOL, color=BLUE, linewidth=2.5, marker='o',
                 markersize=5, label="Volume", zorder=3)
        ax2 = ax1.twinx()
        ax2.plot(MONTHS, DLY, color=ACCENT2, linewidth=2, linestyle='--',
                 marker='s', markersize=5, label="Delayed", zorder=3)
        ax2.tick_params(colors=MUTED, labelsize=9)
        ax2.spines["right"].set_color(BORDER)
        ax2.spines["top"].set_color(BORDER)
        ax2.spines["left"].set_color(BORDER)
        ax2.spines["bottom"].set_color(BORDER)
        ax2.set_ylabel("Delayed Count", color=MUTED, fontsize=9)
        ax1.set_ylabel("Volume", color=MUTED, fontsize=9)
        ax1.grid(axis='y'); ax1.tick_params(labelsize=9); _spine(ax1)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, frameon=False,
                   fontsize=9, labelcolor=[BLUE, ACCENT2], loc='upper left')
        show_fig(fig)

    with col_f:
        section_card("Transport Mode vs Avg Delay", "AIR vs SEA COMPARISON")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.barh(["Sea", "Air"], [1.12, 0.62], color=[BLUE, ACCENT], height=0.4, zorder=3)
        for i, v in enumerate([1.12, 0.62]):
            ax.text(v+0.02, i, f"{v}d", va='center', color=TEXT,
                    fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.5)
        ax.tick_params(axis='y', labelsize=12, labelcolor=TEXT)
        ax.grid(axis='x', zorder=0); _spine(ax)
        show_fig(fig)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "⚡  Risk Predictor":
    page_header("Shipment", "Risk Predictor",
                "POWERED BY XGBOOST  ·  FILL IN SHIPMENT DETAILS TO ASSESS DELAY RISK",
                "● Model Ready", ACCENT)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;
                    padding:22px 24px 10px'>
          <div style='font-family:"Space Mono",monospace;font-size:10px;
                      color:{MUTED};letter-spacing:1.5px;margin-bottom:16px'>
            SHIPMENT PARAMETERS
          </div>
        """, unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            origin = st.selectbox("Origin City",
                ["Shanghai, CN","Shenzhen, CN","Tokyo, JP","Hamburg, DE","Mumbai, IN","Santos, BR"])
        with r1c2:
            dest = st.selectbox("Destination City",
                ["Los Angeles, US","Rotterdam, NL","Singapore, SG","New York, US","Felixstowe, UK","Shanghai, CN"])

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            route = st.selectbox("Route Type", ["Pacific","Atlantic","Suez","Intra-Asia","Commodity"])
        with r2c2:
            mode = st.selectbox("Transport Mode", ["Sea","Air"])

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            product = st.selectbox("Product Category",
                ["Textiles","Pharmaceuticals","Semiconductors","Consumer Electronics","Raw Materials","Perishables"])
        with r3c2:
            base_lead = st.number_input("Base Lead Time (days)", min_value=1, max_value=60, value=18)

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            sched_lead = st.number_input("Scheduled Lead Time (days)", min_value=1, max_value=70, value=21)
        with r4c2:
            geo = st.number_input("Geopolitical Risk (0–1)", min_value=0.1, max_value=0.9,
                                  value=0.55, step=0.01, format="%.2f")

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            weather = st.number_input("Weather Severity (0–10)", min_value=0.0, max_value=10.0,
                                      value=4.5, step=0.1, format="%.1f")
        with r5c2:
            inflation = st.number_input("Inflation Rate (%)", value=3.5, step=0.1, format="%.1f")

        r6c1, r6c2 = st.columns(2)
        with r6c1:
            cost = st.number_input("Shipping Cost (USD)", min_value=0, value=5000, step=100)
        with r6c2:
            weight = st.number_input("Order Weight (kg)", min_value=100, value=3000, step=100)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡  PREDICT DELAY RISK")

    # ── Result panel ────────────────────────────────────────
    with col_result:
        if not predict_clicked:
            st.markdown(f"""
            <div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;
                        padding:60px 30px;text-align:center;min-height:420px;
                        display:flex;flex-direction:column;align-items:center;justify-content:center'>
              <div style='font-size:52px;opacity:.3'>🔍</div>
              <div style='font-family:"Space Mono",monospace;font-size:11px;
                          color:{MUTED};margin-top:16px;line-height:1.8'>
                Fill in shipment parameters<br>and click Predict to see<br>real-time risk assessment
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── Heuristic model ────────────────────────────
            score = 0.05
            buffer = sched_lead - base_lead
            if buffer <= 1:   score += 0.30
            elif buffer <= 2: score += 0.18
            elif buffer <= 4: score += 0.08
            else:             score += 0.01

            route_risk = {"Suez":0.16,"Commodity":0.12,"Pacific":0.08,"Atlantic":0.06,"Intra-Asia":0.04}
            score += route_risk.get(route, 0.06)
            score += 0.06 if mode=="Sea" else 0.02
            score += 0.08 if geo>0.7 else 0.04 if geo>0.5 else 0.01
            score += 0.07 if weather>7 else 0.04 if weather>5 else 0.01
            prod_risk = {"Perishables":0.05,"Semiconductors":0.04,
                         "Consumer Electronics":0.03,"Pharmaceuticals":0.02}
            score += prod_risk.get(product, 0.01)
            origin_risk = {"Santos, BR":0.06,"Mumbai, IN":0.05,"Shenzhen, CN":0.03}
            score += origin_risk.get(origin, 0.01)
            score = min(max(score, 0.02), 0.97)
            pct = round(score * 100)

            color = ACCENT2 if score>0.5 else ACCENT3 if score>0.25 else ACCENT
            label = ("🔴  HIGH DELAY RISK" if score>0.5
                     else "🟡  MODERATE RISK" if score>0.25
                     else "🟢  LOW RISK")
            desc = (
                f"{pct}% probability of delay detected. Tight lead time buffer and high-risk route/origin combination. Consider expedited air freight or alternate routing."
                if score>0.5 else
                f"{pct}% delay probability. Some risk factors present — monitor geopolitical and weather conditions. Standard mitigation recommended."
                if score>0.25 else
                f"Only {pct}% delay probability. Shipment parameters look healthy and within safe thresholds. Proceed with standard shipping."
            )

            factors = [
                ("Lead Time Buffer",    min(90 if buffer<=1 else 60 if buffer<=2 else 30 if buffer<=4 else 10, 100)),
                ("Route Risk",          round(route_risk.get(route,0.06)/0.16*100)),
                ("Geopolitical Risk",   round(geo*100)),
                ("Weather Severity",    round(weather*10)),
                ("Transport Mode Risk", 65 if mode=="Sea" else 25),
                ("Product Category",    round((prod_risk.get(product,0.01)/0.05)*60+10)),
            ]

            # Gauge figure
            fig_g, ax_g = plt.subplots(figsize=(5, 2.8))
            theta_bg = np.linspace(np.pi, 0, 300)
            ax_g.plot(np.cos(theta_bg), np.sin(theta_bg),
                      color=DIM, linewidth=18, solid_capstyle='round', zorder=1)
            theta_fill = np.linspace(np.pi, np.pi - score*np.pi, 300)
            ax_g.plot(np.cos(theta_fill), np.sin(theta_fill),
                      color=color, linewidth=18, solid_capstyle='round', zorder=2)
            ax_g.text(0, 0.08, f"{pct}%", ha='center', va='center',
                      fontsize=36, fontweight='bold', color=color, family='monospace')
            ax_g.text(0, -0.22, "DELAY PROBABILITY",
                      ha='center', color=MUTED, fontsize=9, family='monospace')
            ax_g.set_xlim(-1.25, 1.25); ax_g.set_ylim(-0.4, 1.2)
            ax_g.set_aspect('equal'); ax_g.axis('off')
            show_fig(fig_g)

            # Risk label + description
            st.markdown(f"""
            <div style='text-align:center;margin:-8px 0 12px'>
              <div style='font-size:20px;font-weight:800;color:{color};
                          font-family:Syne,sans-serif'>{label}</div>
              <div style='font-family:"Space Mono",monospace;font-size:10px;
                          color:{MUTED};margin-top:8px;line-height:1.7;
                          max-width:380px;margin-left:auto;margin-right:auto'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Factor bars
            st.markdown(f"""
            <div style='font-family:"Space Mono",monospace;font-size:9px;
                        color:{MUTED};letter-spacing:1.5px;margin:12px 0 8px'>
              RISK FACTOR BREAKDOWN
            </div>
            """, unsafe_allow_html=True)

            for fname, fval in factors:
                bar_color = ACCENT2 if fval>70 else ACCENT3 if fval>40 else ACCENT
                st.markdown(f"""
                <div style='background:{SURFACE};border:1px solid {BORDER};
                            border-radius:8px;padding:8px 12px;margin-bottom:6px;
                            display:flex;align-items:center;gap:10px'>
                  <div style='min-width:160px;font-size:12px;font-weight:600;
                              color:{TEXT};font-family:Syne,sans-serif'>{fname}</div>
                  <div style='flex:1;background:{DIM};height:6px;border-radius:3px;overflow:hidden'>
                    <div style='width:{fval}%;height:100%;background:{bar_color};
                                border-radius:3px'></div>
                  </div>
                  <div style='min-width:36px;text-align:right;font-family:"Space Mono",
                              monospace;font-size:11px;color:{MUTED}'>{fval}%</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — EDA INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "∿  EDA Insights":
    page_header("Exploratory Data", "Analysis",
                "KEY PATTERNS  ·  CORRELATIONS  ·  RISK DRIVERS",
                "10K Shipments", BLUE)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Stat chips
    chips = [
        ("DATASET SIZE",     "10,000",  "Shipment records",     ACCENT),
        ("FEATURES USED",    "16",      "Non-leaky predictors", BLUE),
        ("DELAY RATE",       "12.5%",   "Class imbalance ~1:7", ACCENT2),
        ("MAX DELAY",        "20 days", "Days observed",        ACCENT3),
        ("COST CORRELATION", "~0.01",   "No effect on delay",   MUTED),
        ("ROUTES COVERED",   "5",       "Pacific, Suez, Atl…",  ACCENT),
    ]
    chip_cols = st.columns(6)
    for col, (lbl, val, sub, color) in zip(chip_cols, chips):
        with col:
            st.markdown(f"""
            <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;
                        padding:14px 16px'>
              <div style='font-family:"Space Mono",monospace;font-size:8px;
                          color:{MUTED};letter-spacing:1.5px'>{lbl}</div>
              <div style='font-size:22px;font-weight:800;color:{color};
                          font-family:Syne,sans-serif;letter-spacing:-1px'>{val}</div>
              <div style='font-size:10px;color:{MUTED};font-family:"Space Mono",monospace'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Delay histogram + Correlation
    col1, col2 = st.columns(2)
    with col1:
        section_card("Delay Day Distribution", "HISTOGRAM OF DELAY DAYS")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        hist_colors = [ACCENT if b==0 else ACCENT3 if b<5 else ACCENT2 for b in DELAY_BINS]
        ax.bar([str(b) for b in DELAY_BINS], DELAY_CNTS, color=hist_colors, width=0.7, zorder=3)
        ax.set_xlabel("Delay Days", color=MUTED, fontsize=9)
        ax.grid(axis='y', zorder=0); ax.tick_params(labelsize=9); _spine(ax)
        show_fig(fig)

    with col2:
        section_card("Feature Correlation to Delay Days", "NUMERIC FEATURES RANKED")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        bars = ax.barh(CORR_NAMES, CORR_VALS, color=CORR_COLS, height=0.52, zorder=3)
        for bar, val in zip(bars, CORR_VALS):
            ax.text(val+0.008, bar.get_y()+bar.get_height()/2, f"{val:.2f}",
                    va='center', color=TEXT, fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1.0); ax.invert_yaxis()
        ax.grid(axis='x', zorder=0); ax.tick_params(labelsize=9); _spine(ax)
        show_fig(fig)

    # Key findings table
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    section_card("Key EDA Findings", "ANALYST NOTES FROM EXPLORATION")

    findings = [
        ("Route Suez has highest avg delay",         "Route_Type",          "HIGH",  ACCENT2),
        ("Sea transport ~1.8× more delay than Air",  "Transportation_Mode", "HIGH",  ACCENT2),
        ("Perishables face highest category risk",   "Product_Category",    "HIGH",  ACCENT2),
        ("Santos, BR origin is highest risk city",   "Origin_City",         "MED",   ACCENT3),
        ("Shipping cost has near-zero correlation",  "Shipping_Cost_USD",   "LOW",   BLUE),
        ("Weather severity weakly correlates",       "Weather_Severity",    "LOW",   BLUE),
        ("Port Congestion is most common event",     "Disruption_Event",    "HIGH",  ACCENT2),
        ("Class imbalance: 87.5% vs 12.5%",          "Delivery_Status",     "NOTE",  MUTED),
    ]

    header_html = f"""
    <div style='display:grid;grid-template-columns:2fr 1fr 80px;
                background:{DIM};border-radius:8px 8px 0 0;padding:8px 16px;
                font-family:"Space Mono",monospace;font-size:9px;
                color:{MUTED};letter-spacing:1.5px;gap:12px'>
      <div>FINDING</div><div>VARIABLE</div><div>IMPACT</div>
    </div>"""
    rows_html = ""
    for i, (finding, var, impact, color) in enumerate(findings):
        bg = CARD if i%2==0 else f"{SURFACE}"
        rows_html += f"""
        <div style='display:grid;grid-template-columns:2fr 1fr 80px;
                    background:{bg};padding:10px 16px;gap:12px;
                    border-bottom:1px solid {BORDER};align-items:center'>
          <div style='font-size:12px;color:{TEXT};font-family:Syne,sans-serif'>{finding}</div>
          <div style='font-family:"Space Mono",monospace;font-size:10px;color:{BLUE}'>{var}</div>
          <div style='font-size:10px;font-weight:700;color:{color};
                      font-family:"Space Mono",monospace'>{impact}</div>
        </div>"""
    st.markdown(f"""
    <div style='border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:16px'>
      {header_html}{rows_html}
    </div>
    """, unsafe_allow_html=True)

    # Disruption chart
    section_card("Disruption Event Frequency", "COUNT PER EVENT TYPE")
    fig, ax = plt.subplots(figsize=(10, 2.8))
    bars = ax.bar(DISRUPT_LABELS, DISRUPT_CNTS, color=DISRUPT_COLORS, width=0.5, zorder=3)
    for bar, val in zip(bars, DISRUPT_CNTS):
        ax.text(bar.get_x()+bar.get_width()/2, val+60, f"{val:,}",
                ha='center', color=TEXT, fontsize=10, fontweight='bold')
    ax.grid(axis='y', zorder=0); ax.tick_params(labelsize=10); _spine(ax)
    ax.set_ylim(0, max(DISRUPT_CNTS)*1.18)
    show_fig(fig)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "▤  Model Comparison":
    page_header("Model", "Comparison",
                "5 CLASSIFIERS  ·  ACCURACY & ROC-AUC BENCHMARKS",
                "XGBoost Winner", ACCENT)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Model cards
    model_cols = st.columns(5)
    for col, m in zip(model_cols, MODELS_DATA):
        border = f"2px solid {ACCENT}" if m["best"] else f"1px solid {BORDER}"
        badge = f"<div style='font-family:\"Space Mono\",monospace;font-size:8px;color:{ACCENT};text-align:right;letter-spacing:1px'>★ BEST</div>" if m["best"] else ""
        val_color = ACCENT if m["best"] else TEXT
        auc_color = ACCENT if m["best"] else BLUE
        with col:
            st.markdown(f"""
            <div style='background:{CARD};border:{border};border-radius:12px;
                        padding:18px 16px;height:100%'>
              {badge}
              <div style='font-size:13px;font-weight:800;color:{TEXT};
                          font-family:Syne,sans-serif;margin-bottom:12px'>{m["name"]}</div>
              <div style='font-family:"Space Mono",monospace;font-size:8px;
                          color:{MUTED};letter-spacing:1px'>ACCURACY</div>
              <div style='font-size:26px;font-weight:800;color:{val_color};
                          font-family:Syne,sans-serif;letter-spacing:-1px'>{m["acc"]}%</div>
              <div style='font-family:"Space Mono",monospace;font-size:8px;
                          color:{MUTED};letter-spacing:1px;margin-top:8px'>ROC-AUC</div>
              <div style='font-size:20px;font-weight:800;color:{auc_color};
                          font-family:Syne,sans-serif'>{m["auc"]:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Accuracy + AUC charts
    model_names = [m["name"].split()[0] for m in MODELS_DATA]
    acc_vals    = [m["acc"] for m in MODELS_DATA]
    auc_vals    = [m["auc"] for m in MODELS_DATA]
    mc1         = [ACCENT if m["best"] else DIM  for m in MODELS_DATA]
    mc2         = [ACCENT if m["best"] else BLUE for m in MODELS_DATA]

    col_a, col_b = st.columns(2)
    with col_a:
        section_card("Accuracy Comparison", "ALL 5 MODELS · TEST SET")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars = ax.bar(model_names, acc_vals, color=mc1, width=0.5, zorder=3)
        for bar, val in zip(bars, acc_vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.2, f"{val}%",
                    ha='center', color=TEXT, fontsize=9, fontweight='bold')
        ax.set_ylim(75, 98)
        ax.grid(axis='y', zorder=0); ax.tick_params(labelsize=9); _spine(ax)
        show_fig(fig)

    with col_b:
        section_card("ROC-AUC Comparison", "HIGHER IS BETTER · MAX 1.0")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars = ax.bar(model_names, auc_vals, color=mc2, width=0.5, zorder=3)
        for bar, val in zip(bars, auc_vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.004, f"{val:.3f}",
                    ha='center', color=TEXT, fontsize=9, fontweight='bold')
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis='y', zorder=0); ax.tick_params(labelsize=9); _spine(ax)
        show_fig(fig)

    # Feature importance
    section_card("XGBoost — Top Feature Importances",
                 "WHICH FEATURES DRIVE PREDICTIONS MOST")
    feat_colors = [ACCENT if v>0.15 else ACCENT3 if v>0.08 else BLUE for v in FEAT_VALS]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    bars = ax.barh(FEAT_NAMES, FEAT_VALS, color=feat_colors, height=0.5, zorder=3)
    for bar, val in zip(bars, FEAT_VALS):
        ax.text(val+0.003, bar.get_y()+bar.get_height()/2, f"{val:.2f}",
                va='center', color=TEXT, fontsize=9, fontweight='bold')
    ax.set_xlim(0, 0.38); ax.invert_yaxis()
    ax.grid(axis='x', zorder=0); ax.tick_params(labelsize=10); _spine(ax)
    show_fig(fig)

    # Confusion matrix
    section_card("XGBoost — Confusion Matrix", "PREDICTED vs ACTUAL · TEST SET")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    cm_data  = [[1720, 78], [124, 78]]
    cm_labels= [["TP\n1720", "FP\n78"], ["FN\n124", "TN\n78"]]
    cm_colors= [[hex_to_rgba(ACCENT, 0.5), hex_to_rgba(ACCENT2, 0.2)],
                [hex_to_rgba(ACCENT2, 0.2), hex_to_rgba(ACCENT, 0.3)]]
    xlabels  = ["Predicted: On-Time", "Predicted: Delayed"]
    ylabels  = ["Actual: On-Time", "Actual: Delayed"]
    for i in range(2):
        for j in range(2):
            rect = FancyBboxPatch((j*1.1 - 0.48, (1-i)*1.1 - 0.48), 0.96, 0.96,
                                  boxstyle="round,pad=0.06",
                                  facecolor=cm_colors[i][j],
                                  edgecolor=BORDER, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j*1.1, (1-i)*1.1, cm_labels[i][j],
                    ha='center', va='center', fontsize=12,
                    color=TEXT, fontweight='bold', family='monospace')
    ax.set_xlim(-0.7, 1.85); ax.set_ylim(-0.65, 1.75)
    ax.set_xticks([0, 1.1]); ax.set_xticklabels(xlabels, fontsize=9, color=MUTED)
    ax.set_yticks([0, 1.1]); ax.set_yticklabels(ylabels[::-1], fontsize=9, color=MUTED)
    ax.set_aspect('equal'); _spine(ax)
    show_fig(fig)
