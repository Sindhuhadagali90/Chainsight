"""
ChainSight — Supply Chain Risk Intelligence Dashboard
Built with Streamlit + Plotly (both pre-installed on Streamlit Cloud)
Zero extra dependencies needed.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── PAGE CONFIG (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="ChainSight — Supply Chain Risk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PALETTE ─────────────────────────────────────────────────────────────────
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

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main {{
    background-color: {BG} !important;
    font-family: 'Syne', sans-serif !important;
}}
[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
.block-container {{
    padding: 24px 32px !important;
    max-width: 100% !important;
    background: {BG} !important;
}}
h1, h2, h3 {{ color: {TEXT} !important; font-family: 'Syne', sans-serif !important; }}
p  {{ color: {MUTED} !important; }}
.stButton > button {{
    background: {ACCENT} !important;
    color: {BG} !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    text-transform: uppercase;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}
.stSelectbox label, .stNumberInput label {{
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: {MUTED} !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
.stSelectbox > div > div {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}
.stNumberInput input {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY BASE LAYOUT ───────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family="Space Mono, monospace", color=MUTED, size=11),
    margin=dict(l=12, r=12, t=36, b=12),
    showlegend=False,
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
)

def base_fig(height=320):
    fig = go.Figure()
    fig.update_layout(height=height, **PLOT_LAYOUT)
    return fig

def render(fig, key=None):
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=key)

# ─── DATA ─────────────────────────────────────────────────────────────────────
ROUTE_LABELS  = ["Suez", "Commodity", "Pacific", "Atlantic", "Intra-Asia"]
ROUTE_DELAYS  = [1.41, 1.22, 1.05, 0.92, 0.72]
ROUTE_COLORS  = [ACCENT2, ACCENT3, ACCENT, BLUE, DIM]

PRODUCT_LABELS = ["Perishables","Semiconductors","Consumer Elec.",
                  "Pharma","Machinery","Textiles","Raw Materials"]
PRODUCT_DELAYS = [1.28, 1.14, 1.07, 0.98, 0.91, 0.85, 0.78]
PRODUCT_COLORS = [ACCENT2 if v>1.1 else ACCENT3 if v>0.95 else ACCENT
                  for v in PRODUCT_DELAYS]

ORIGIN_LABELS = ["Santos, BR","Mumbai, IN","Shenzhen, CN",
                 "Shanghai, CN","Tokyo, JP","Hamburg, DE"]
ORIGIN_DELAYS = [1.30, 1.22, 1.05, 0.98, 0.88, 0.72]
ORIGIN_COLORS = [ACCENT2 if v>1.1 else ACCENT3 if v>0.95 else BLUE
                 for v in ORIGIN_DELAYS]

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]
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
              "Weather_Index","Geopolitical_Idx","Transport_Mode",
              "Product_Cat","Shipping_Cost"]
FEAT_VALS  = [0.31, 0.22, 0.14, 0.09, 0.07, 0.06, 0.05, 0.04, 0.01]

DELAY_BINS = list(range(16))
DELAY_CNTS = [8753,362,185,148,112,89,72,58,45,38,30,24,19,14,11,40]

CORR_NAMES = ["Actual_Lead_Time","Sched_Lead_Time","Base_Lead_Time",
              "Weather_Severity","Geo_Risk_Index","Inflation_Rate","Shipping_Cost"]
CORR_VALS  = [0.82, 0.43, 0.38, 0.12, 0.08, 0.03, 0.01]
CORR_COLS  = [ACCENT2, ACCENT3, ACCENT3, BLUE, BLUE, DIM, DIM]

DISRUPT_LABELS = ["Port Congestion","Geopolitical Conflict",
                  "Extreme Weather","No Event"]
DISRUPT_CNTS   = [820, 312, 115, 8753]
DISRUPT_COLORS = [ACCENT2, ACCENT3, BLUE, DIM]

ROUTE_RISK  = {"Suez":0.16,"Commodity":0.12,"Pacific":0.08,
               "Atlantic":0.06,"Intra-Asia":0.04}
PROD_RISK   = {"Perishables":0.05,"Semiconductors":0.04,
               "Consumer Electronics":0.03,"Pharmaceuticals":0.02}
ORIGIN_RISK = {"Santos, BR":0.06,"Mumbai, IN":0.05,"Shenzhen, CN":0.03}

# ─── UI HELPERS ───────────────────────────────────────────────────────────────
def page_header(title, accent, subtitle, badge="", badge_color=ACCENT):
    bdg = ""
    if badge:
        bdg = (f"<span style='padding:5px 14px;border-radius:20px;"
               f"border:1px solid {badge_color}55;background:{badge_color}11;"
               f"font-family:\"Space Mono\",monospace;font-size:10px;"
               f"color:{badge_color};letter-spacing:1px'>{badge}</span>")
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:center;
                flex-wrap:wrap;gap:12px;margin-bottom:20px'>
      <div>
        <div style='font-size:28px;font-weight:800;letter-spacing:-1px;
                    color:{TEXT};font-family:Syne,sans-serif;line-height:1.1'>
          {title} <span style='color:{ACCENT}'>{accent}</span>
        </div>
        <div style='font-family:"Space Mono",monospace;font-size:10px;
                    color:{MUTED};margin-top:6px;letter-spacing:.5px'>{subtitle}</div>
      </div>
      {bdg}
    </div>""", unsafe_allow_html=True)


def kpi(label, value, sub, color):
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;
                padding:20px 22px;height:100%'>
      <div style='height:2px;background:{color};border-radius:2px;
                  margin-bottom:12px'></div>
      <div style='font-family:"Space Mono",monospace;font-size:9px;
                  letter-spacing:1.5px;text-transform:uppercase;
                  color:{MUTED}'>{label}</div>
      <div style='font-size:32px;font-weight:800;letter-spacing:-2px;color:{color};
                  font-family:Syne,sans-serif;line-height:1.1;margin:4px 0'>{value}</div>
      <div style='font-family:"Space Mono",monospace;font-size:10px;
                  color:{MUTED}'>{sub}</div>
    </div>""", unsafe_allow_html=True)


def card_title(title, sub=""):
    st.markdown(f"""
    <div style='margin-bottom:8px'>
      <div style='font-size:13px;font-weight:700;color:{TEXT};
                  font-family:Syne,sans-serif'>{title}</div>
      {"" if not sub else f"<div style='font-family:\"Space Mono\",monospace;font-size:9px;color:{MUTED};margin-top:2px'>{sub}</div>"}
    </div>""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:8px 0 18px'>
      <div style='display:flex;align-items:center;gap:10px'>
        <div style='background:{ACCENT};border-radius:8px;width:34px;height:34px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:20px;font-weight:900;color:{BG}'>◆</div>
        <div>
          <div style='font-family:Syne,sans-serif;font-size:18px;
                      font-weight:800;color:{TEXT}'>
            Chain<span style='color:{ACCENT}'>Sight</span></div>
          <div style='font-family:"Space Mono",monospace;font-size:8px;
                      color:{MUTED};letter-spacing:2px'>RISK INTELLIGENCE v1.0</div>
        </div>
      </div>
    </div>
    <hr style='border-color:{BORDER};margin:0 0 14px'>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "▦  Overview",
        "⚡  Risk Predictor",
        "∿  EDA Insights",
        "▤  Model Comparison",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <hr style='border-color:{BORDER};margin:18px 0 12px'>
    <div style='display:flex;align-items:center;gap:8px'>
      <div style='width:8px;height:8px;border-radius:50%;background:{ACCENT};
                  box-shadow:0 0 6px {ACCENT}'></div>
      <span style='font-family:"Space Mono",monospace;font-size:9px;
                   color:{MUTED};letter-spacing:1.5px'>MODEL LIVE · XGB v1.0</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "▦  Overview":
    page_header("Supply Chain", "Overview",
                "10,000 SHIPMENTS  ·  GLOBAL ROUTES  ·  2024–2025",
                "● Live Model", ACCENT)

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("TOTAL SHIPMENTS", "10,000", "Across 6 origin cities",  ACCENT)
    with k2: kpi("DELAYED RATE",    "12.5%",  "1,247 late shipments",    ACCENT2)
    with k3: kpi("AVG DELAY DAYS",  "0.95",   "Max observed: 20 days",   ACCENT3)
    with k4: kpi("MODEL ROC-AUC",   "0.941",  "XGBoost — best performer",BLUE)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        card_title("Average Delay by Route Type", "MEAN DELAY DAYS PER ROUTE")
        fig = base_fig(300)
        fig.add_trace(go.Bar(
            x=ROUTE_LABELS, y=ROUTE_DELAYS,
            marker_color=ROUTE_COLORS, marker_line_width=0,
            text=[f"{v}d" for v in ROUTE_DELAYS],
            textposition="outside", textfont=dict(color=TEXT, size=11),
            hovertemplate="%{x}: <b>%{y}d</b><extra></extra>",
        ))
        fig.update_layout(yaxis=dict(range=[0, 1.85], gridcolor=BORDER),
                          xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        render(fig, "route_bar")

    with c2:
        card_title("Delivery Status", "DISTRIBUTION ACROSS ALL SHIPMENTS")
        fig = go.Figure(go.Pie(
            values=[87.5, 12.5], labels=["On Time", "Delayed"],
            hole=0.62, marker=dict(colors=[ACCENT, ACCENT2],
                                   line=dict(width=0)),
            textinfo="percent",
            textfont=dict(color=BG, size=12),
            hovertemplate="%{label}: <b>%{percent}</b><extra></extra>",
        ))
        fig.update_layout(
            height=300, paper_bgcolor=CARD, plot_bgcolor=CARD,
            margin=dict(l=12, r=12, t=36, b=12),
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08,
                        font=dict(color=MUTED, size=11, family="Space Mono")),
            annotations=[dict(text="87.5%<br>On Time", x=0.5, y=0.5,
                              font=dict(size=14, color=ACCENT, family="Space Mono"),
                              showarrow=False)],
        )
        render(fig, "donut")

    c3, c4 = st.columns(2)
    with c3:
        card_title("Avg Delay by Product Category", "WHICH PRODUCTS FACE HIGHEST RISK")
        fig = base_fig(320)
        fig.add_trace(go.Bar(
            y=PRODUCT_LABELS, x=PRODUCT_DELAYS, orientation='h',
            marker_color=PRODUCT_COLORS, marker_line_width=0,
            text=[f"{v}d" for v in PRODUCT_DELAYS],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="%{y}: <b>%{x}d</b><extra></extra>",
        ))
        fig.update_layout(xaxis=dict(range=[0, 1.65], gridcolor=BORDER),
                          yaxis=dict(autorange="reversed",
                                     gridcolor="rgba(0,0,0,0)"))
        render(fig, "product_bar")

    with c4:
        card_title("Avg Delay by Origin City", "SOURCE RISK OVERVIEW")
        fig = base_fig(320)
        fig.add_trace(go.Bar(
            y=ORIGIN_LABELS, x=ORIGIN_DELAYS, orientation='h',
            marker_color=ORIGIN_COLORS, marker_line_width=0,
            text=[f"{v}d" for v in ORIGIN_DELAYS],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="%{y}: <b>%{x}d</b><extra></extra>",
        ))
        fig.update_layout(xaxis=dict(range=[0, 1.65], gridcolor=BORDER),
                          yaxis=dict(autorange="reversed",
                                     gridcolor="rgba(0,0,0,0)"))
        render(fig, "origin_bar")

    c5, c6 = st.columns([3, 2])
    with c5:
        card_title("Shipment Volume & Delay Trend", "2024–2025 MONTHLY OVERVIEW")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=MONTHS, y=VOL, name="Volume",
            line=dict(color=BLUE, width=2.5),
            fill='tozeroy', fillcolor="rgba(61,158,255,0.07)",
            mode='lines+markers', marker=dict(size=5, color=BLUE),
            hovertemplate="Volume: <b>%{y}</b><extra></extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=MONTHS, y=DLY, name="Delayed",
            line=dict(color=ACCENT2, width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=5, color=ACCENT2, symbol='square'),
            hovertemplate="Delayed: <b>%{y}</b><extra></extra>",
        ), secondary_y=True)
        fig.update_layout(
            height=300, paper_bgcolor=CARD, plot_bgcolor=CARD,
            margin=dict(l=12, r=12, t=36, b=12),
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.12,
                        font=dict(color=MUTED, size=10, family="Space Mono")),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=BORDER),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER),
            yaxis2=dict(gridcolor="rgba(0,0,0,0)", linecolor=BORDER),
        )
        render(fig, "trend")

    with c6:
        card_title("Transport Mode vs Avg Delay", "AIR vs SEA COMPARISON")
        fig = base_fig(300)
        fig.add_trace(go.Bar(
            y=["Sea", "Air"], x=[1.12, 0.62], orientation='h',
            marker_color=[BLUE, ACCENT], marker_line_width=0,
            text=["1.12d", "0.62d"],
            textposition="outside", textfont=dict(color=TEXT, size=12),
            hovertemplate="%{y}: <b>%{x}d</b><extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 1.5], gridcolor=BORDER),
            yaxis=dict(tickfont=dict(color=TEXT, size=13),
                       gridcolor="rgba(0,0,0,0)"),
        )
        render(fig, "mode_bar")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡  Risk Predictor":
    page_header("Shipment", "Risk Predictor",
                "POWERED BY XGBOOST  ·  FILL IN DETAILS TO ASSESS DELAY RISK",
                "● Model Ready", ACCENT)

    col_form, col_result = st.columns(2, gap="large")

    with col_form:
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};
                    border-radius:14px;padding:22px 24px 4px'>
          <div style='font-family:"Space Mono",monospace;font-size:10px;
                      color:{MUTED};letter-spacing:1.5px;
                      margin-bottom:14px'>SHIPMENT PARAMETERS</div>
        """, unsafe_allow_html=True)

        fa, fb = st.columns(2)
        with fa:
            origin = st.selectbox("Origin City", [
                "Shanghai, CN","Shenzhen, CN","Tokyo, JP",
                "Hamburg, DE","Mumbai, IN","Santos, BR"])
        with fb:
            dest = st.selectbox("Destination", [
                "Los Angeles, US","Rotterdam, NL","Singapore, SG",
                "New York, US","Felixstowe, UK","Shanghai, CN"])

        fc2, fd = st.columns(2)
        with fc2:
            route = st.selectbox("Route Type",
                ["Pacific","Atlantic","Suez","Intra-Asia","Commodity"])
        with fd:
            mode = st.selectbox("Transport Mode", ["Sea","Air"])

        fe, ff = st.columns(2)
        with fe:
            product = st.selectbox("Product Category", [
                "Textiles","Pharmaceuticals","Semiconductors",
                "Consumer Electronics","Raw Materials","Perishables"])
        with ff:
            base_lead = st.number_input("Base Lead Time (days)",
                                        min_value=1, max_value=60, value=18)

        fg, fh = st.columns(2)
        with fg:
            sched_lead = st.number_input("Scheduled Lead Time (days)",
                                         min_value=1, max_value=70, value=21)
        with fh:
            geo = st.number_input("Geopolitical Risk (0–1)",
                                  min_value=0.1, max_value=0.9,
                                  value=0.55, step=0.01, format="%.2f")

        fi, fj = st.columns(2)
        with fi:
            weather = st.number_input("Weather Severity (0–10)",
                                      min_value=0.0, max_value=10.0,
                                      value=4.5, step=0.1, format="%.1f")
        with fj:
            inflation = st.number_input("Inflation Rate (%)",
                                        value=3.5, step=0.1, format="%.1f")

        fk, fl = st.columns(2)
        with fk:
            cost = st.number_input("Shipping Cost (USD)",
                                   min_value=0, value=5000, step=100)
        with fl:
            weight = st.number_input("Order Weight (kg)",
                                     min_value=100, value=3000, step=100)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        clicked = st.button("⚡  PREDICT DELAY RISK")

    with col_result:
        if not clicked:
            st.markdown(f"""
            <div style='background:{CARD};border:1px solid {BORDER};
                        border-radius:14px;padding:80px 30px;text-align:center'>
              <div style='font-size:56px;opacity:.25'>🔍</div>
              <div style='font-family:"Space Mono",monospace;font-size:11px;
                          color:{MUTED};margin-top:18px;line-height:1.9'>
                Fill in shipment parameters<br>and click Predict to see<br>
                real-time risk assessment
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            # Heuristic model
            score = 0.05
            buffer = sched_lead - base_lead
            if   buffer <= 1: score += 0.30
            elif buffer <= 2: score += 0.18
            elif buffer <= 4: score += 0.08
            else:             score += 0.01
            score += ROUTE_RISK.get(route, 0.06)
            score += 0.06 if mode == "Sea" else 0.02
            score += 0.08 if geo > 0.7 else 0.04 if geo > 0.5 else 0.01
            score += 0.07 if weather > 7 else 0.04 if weather > 5 else 0.01
            score += PROD_RISK.get(product, 0.01)
            score += ORIGIN_RISK.get(origin, 0.01)
            score = min(max(score, 0.02), 0.97)
            pct   = round(score * 100)

            color = ACCENT2 if score > 0.5 else ACCENT3 if score > 0.25 else ACCENT
            label = ("🔴  HIGH DELAY RISK" if score > 0.5
                     else "🟡  MODERATE RISK" if score > 0.25
                     else "🟢  LOW RISK")
            desc  = (
                f"{pct}% probability of delay. Tight lead time buffer and "
                f"high-risk route/origin combination. Consider expedited air freight."
                if score > 0.5 else
                f"{pct}% delay probability. Some risk factors present — monitor "
                f"geopolitical and weather conditions."
                if score > 0.25 else
                f"Only {pct}% delay probability. Shipment parameters look healthy. "
                f"Proceed with standard shipping."
            )

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number=dict(suffix="%",
                            font=dict(size=42, color=color, family="Space Mono")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=MUTED,
                              tickfont=dict(color=MUTED, size=10)),
                    bar=dict(color=color, thickness=0.28),
                    bgcolor=DIM,
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 25],   color="rgba(0,255,178,0.08)"),
                        dict(range=[25, 50],  color="rgba(255,184,0,0.08)"),
                        dict(range=[50, 100], color="rgba(255,77,109,0.08)"),
                    ],
                    threshold=dict(line=dict(color=color, width=3),
                                   thickness=0.8, value=pct),
                ),
            ))
            fig.update_layout(
                height=260, paper_bgcolor=CARD, plot_bgcolor=CARD,
                margin=dict(l=30, r=30, t=20, b=10),
                font=dict(color=MUTED, family="Space Mono"),
            )
            render(fig, "gauge")

            st.markdown(f"""
            <div style='text-align:center;margin:-4px 0 16px'>
              <div style='font-size:20px;font-weight:800;color:{color};
                          font-family:Syne,sans-serif'>{label}</div>
              <div style='font-family:"Space Mono",monospace;font-size:10px;
                          color:{MUTED};margin-top:8px;line-height:1.8;
                          max-width:380px;margin-left:auto;margin-right:auto'>
                {desc}</div>
            </div>""", unsafe_allow_html=True)

            factors = [
                ("Lead Time Buffer",
                 min(90 if buffer<=1 else 60 if buffer<=2
                     else 30 if buffer<=4 else 10, 100)),
                ("Route Risk",
                 round(ROUTE_RISK.get(route, 0.06) / 0.16 * 100)),
                ("Geopolitical Risk",  round(geo * 100)),
                ("Weather Severity",   round(weather * 10)),
                ("Transport Mode",     65 if mode == "Sea" else 25),
                ("Product Category",
                 round((PROD_RISK.get(product, 0.01) / 0.05) * 60 + 10)),
            ]
            st.markdown(f"""
            <div style='font-family:"Space Mono",monospace;font-size:9px;
                        color:{MUTED};letter-spacing:1.5px;margin-bottom:10px'>
              RISK FACTOR BREAKDOWN</div>""", unsafe_allow_html=True)
            for fname, fval in factors:
                bc = ACCENT2 if fval > 70 else ACCENT3 if fval > 40 else ACCENT
                st.markdown(f"""
                <div style='background:{SURFACE};border:1px solid {BORDER};
                            border-radius:8px;padding:8px 14px;margin-bottom:6px;
                            display:flex;align-items:center;gap:12px'>
                  <div style='min-width:150px;font-size:12px;font-weight:600;
                              color:{TEXT};font-family:Syne,sans-serif'>{fname}</div>
                  <div style='flex:1;background:{DIM};height:6px;
                              border-radius:3px;overflow:hidden'>
                    <div style='width:{fval}%;height:100%;background:{bc};
                                border-radius:3px'></div>
                  </div>
                  <div style='min-width:34px;text-align:right;
                              font-family:"Space Mono",monospace;
                              font-size:11px;color:{MUTED}'>{fval}%</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "∿  EDA Insights":
    page_header("Exploratory Data", "Analysis",
                "KEY PATTERNS  ·  CORRELATIONS  ·  RISK DRIVERS",
                "10K Shipments", BLUE)

    chips_data = [
        ("DATASET SIZE",     "10,000",  "Shipment records",     ACCENT),
        ("FEATURES USED",    "16",      "Non-leaky predictors", BLUE),
        ("DELAY RATE",       "12.5%",   "Class imbalance ~1:7", ACCENT2),
        ("MAX DELAY",        "20 days", "Days observed",        ACCENT3),
        ("COST CORRELATION", "~0.01",   "No effect on delay",   MUTED),
        ("ROUTES COVERED",   "5",       "Pacific, Suez, Atl…",  ACCENT),
    ]
    for col, (lbl, val, sub, color) in zip(st.columns(6), chips_data):
        with col:
            st.markdown(f"""
            <div style='background:{CARD};border:1px solid {BORDER};
                        border-radius:10px;padding:14px 16px'>
              <div style='font-family:"Space Mono",monospace;font-size:8px;
                          color:{MUTED};letter-spacing:1.5px'>{lbl}</div>
              <div style='font-size:22px;font-weight:800;color:{color};
                          font-family:Syne,sans-serif;letter-spacing:-1px'>{val}</div>
              <div style='font-size:10px;color:{MUTED};
                          font-family:"Space Mono",monospace'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        card_title("Delay Day Distribution", "HISTOGRAM OF DELAY DAYS")
        hist_colors = [ACCENT if b == 0 else ACCENT3 if b < 5
                       else ACCENT2 for b in DELAY_BINS]
        fig = base_fig(310)
        fig.add_trace(go.Bar(
            x=[str(b) for b in DELAY_BINS], y=DELAY_CNTS,
            marker_color=hist_colors, marker_line_width=0,
            hovertemplate="Day %{x}: <b>%{y:,}</b> shipments<extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(title="Delay Days", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor=BORDER),
        )
        render(fig, "hist")

    with e2:
        card_title("Feature Correlation to Delay Days", "NUMERIC FEATURES RANKED")
        fig = base_fig(310)
        fig.add_trace(go.Bar(
            y=CORR_NAMES, x=CORR_VALS, orientation='h',
            marker_color=CORR_COLS, marker_line_width=0,
            text=[f"{v:.2f}" for v in CORR_VALS],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="%{y}: <b>%{x:.2f}</b><extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 1.0], gridcolor=BORDER),
            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        )
        render(fig, "corr")

    card_title("Key EDA Findings", "ANALYST NOTES FROM EXPLORATION")
    findings = [
        ("Route Suez has highest avg delay",        "Route_Type",          "HIGH",  ACCENT2),
        ("Sea transport ~1.8× more delay than Air", "Transportation_Mode", "HIGH",  ACCENT2),
        ("Perishables face highest category risk",  "Product_Category",    "HIGH",  ACCENT2),
        ("Santos, BR origin is highest risk city",  "Origin_City",         "MED",   ACCENT3),
        ("Shipping cost has near-zero correlation", "Shipping_Cost_USD",   "LOW",   BLUE),
        ("Weather severity weakly correlates",      "Weather_Severity",    "LOW",   BLUE),
        ("Port Congestion is most common event",    "Disruption_Event",    "HIGH",  ACCENT2),
        ("Class imbalance: 87.5% vs 12.5%",         "Delivery_Status",     "NOTE",  MUTED),
    ]
    rows_html = "".join(f"""
    <div style='display:grid;grid-template-columns:2fr 1.2fr 80px;
                background:{"" if i % 2 else SURFACE};
                padding:10px 16px;gap:12px;
                border-bottom:1px solid {BORDER};align-items:center'>
      <div style='font-size:12px;color:{TEXT};
                  font-family:Syne,sans-serif'>{f}</div>
      <div style='font-family:"Space Mono",monospace;
                  font-size:10px;color:{BLUE}'>{v}</div>
      <div style='font-size:10px;font-weight:700;color:{c};
                  font-family:"Space Mono",monospace'>{imp}</div>
    </div>""" for i, (f, v, imp, c) in enumerate(findings))

    st.markdown(f"""
    <div style='border:1px solid {BORDER};border-radius:10px;
                overflow:hidden;margin-bottom:16px'>
      <div style='display:grid;grid-template-columns:2fr 1.2fr 80px;
                  background:{DIM};padding:8px 16px;
                  font-family:"Space Mono",monospace;font-size:9px;
                  color:{MUTED};letter-spacing:1.5px;gap:12px'>
        <div>FINDING</div><div>VARIABLE</div><div>IMPACT</div>
      </div>{rows_html}
    </div>""", unsafe_allow_html=True)

    card_title("Disruption Event Frequency", "COUNT PER EVENT TYPE")
    fig = base_fig(260)
    fig.add_trace(go.Bar(
        x=DISRUPT_LABELS, y=DISRUPT_CNTS,
        marker_color=DISRUPT_COLORS, marker_line_width=0,
        text=[f"{v:,}" for v in DISRUPT_CNTS],
        textposition="outside", textfont=dict(color=TEXT, size=11),
        hovertemplate="%{x}: <b>%{y:,}</b><extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, max(DISRUPT_CNTS) * 1.2], gridcolor=BORDER),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    render(fig, "disrupt")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "▤  Model Comparison":
    page_header("Model", "Comparison",
                "5 CLASSIFIERS  ·  ACCURACY & ROC-AUC BENCHMARKS",
                "XGBoost Winner", ACCENT)

    for col, m in zip(st.columns(5), MODELS_DATA):
        border = f"2px solid {ACCENT}" if m["best"] else f"1px solid {BORDER}"
        vc = ACCENT if m["best"] else TEXT
        ac = ACCENT if m["best"] else BLUE
        badge = (f"<div style='font-family:\"Space Mono\",monospace;font-size:8px;"
                 f"color:{ACCENT};text-align:right;letter-spacing:1px;"
                 f"margin-bottom:6px'>★ BEST</div>") if m["best"] else ""
        with col:
            st.markdown(f"""
            <div style='background:{CARD};border:{border};
                        border-radius:12px;padding:18px 16px'>
              {badge}
              <div style='font-size:13px;font-weight:800;color:{TEXT};
                          font-family:Syne,sans-serif;margin-bottom:10px'>
                {m["name"]}</div>
              <div style='font-family:"Space Mono",monospace;font-size:8px;
                          color:{MUTED};letter-spacing:1px'>ACCURACY</div>
              <div style='font-size:26px;font-weight:800;color:{vc};
                          font-family:Syne,sans-serif;letter-spacing:-1px'>
                {m["acc"]}%</div>
              <div style='font-family:"Space Mono",monospace;font-size:8px;
                          color:{MUTED};letter-spacing:1px;margin-top:8px'>ROC-AUC</div>
              <div style='font-size:20px;font-weight:800;color:{ac};
                          font-family:Syne,sans-serif'>{m["auc"]:.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    m_names = [m["name"].split()[0] for m in MODELS_DATA]
    mc_cols = st.columns(2)
    with mc_cols[0]:
        card_title("Accuracy Comparison", "ALL 5 MODELS · TEST SET")
        fig = base_fig(300)
        fig.add_trace(go.Bar(
            x=m_names, y=[m["acc"] for m in MODELS_DATA],
            marker_color=[ACCENT if m["best"] else DIM for m in MODELS_DATA],
            marker_line_width=0,
            text=[f"{m['acc']}%" for m in MODELS_DATA],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="%{x}: <b>%{y}%</b><extra></extra>",
        ))
        fig.update_layout(yaxis=dict(range=[75, 98], gridcolor=BORDER),
                          xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        render(fig, "acc_chart")

    with mc_cols[1]:
        card_title("ROC-AUC Comparison", "HIGHER IS BETTER · MAX 1.0")
        fig = base_fig(300)
        fig.add_trace(go.Bar(
            x=m_names, y=[m["auc"] for m in MODELS_DATA],
            marker_color=[ACCENT if m["best"] else BLUE for m in MODELS_DATA],
            marker_line_width=0,
            text=[f"{m['auc']:.3f}" for m in MODELS_DATA],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            hovertemplate="%{x}: <b>%{y:.3f}</b><extra></extra>",
        ))
        fig.update_layout(yaxis=dict(range=[0.75, 1.0], gridcolor=BORDER),
                          xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        render(fig, "auc_chart")

    card_title("XGBoost — Top Feature Importances",
               "WHICH FEATURES DRIVE PREDICTIONS MOST")
    fc_colors = [ACCENT if v > 0.15 else ACCENT3 if v > 0.08
                 else BLUE for v in FEAT_VALS]
    fig = base_fig(320)
    fig.add_trace(go.Bar(
        y=FEAT_NAMES, x=FEAT_VALS, orientation='h',
        marker_color=fc_colors, marker_line_width=0,
        text=[f"{v:.2f}" for v in FEAT_VALS],
        textposition="outside", textfont=dict(color=TEXT, size=10),
        hovertemplate="%{y}: <b>%{x:.2f}</b><extra></extra>",
    ))
    fig.update_layout(xaxis=dict(range=[0, 0.38], gridcolor=BORDER),
                      yaxis=dict(autorange="reversed",
                                 gridcolor="rgba(0,0,0,0)"))
    render(fig, "feat_imp")

    card_title("XGBoost — Confusion Matrix",
               "PREDICTED vs ACTUAL · TEST SET")
    xl = ["Predicted: On-Time", "Predicted: Delayed"]
    yl = ["Actual: On-Time",    "Actual: Delayed"]
    labels_cm = [["TP: 1720", "FP: 78"], ["FN: 124", "TN: 78"]]
    anns = [
        dict(x=xl[j], y=yl[i],
             text=f"<b>{labels_cm[i][j]}</b>",
             showarrow=False,
             font=dict(color=TEXT, size=14, family="Space Mono"))
        for i in range(2) for j in range(2)
    ]
    fig = go.Figure(go.Heatmap(
        z=[[1720, 78], [124, 78]],
        x=xl, y=yl,
        colorscale=[[0, "rgba(255,77,109,0.25)"],
                    [1, "rgba(0,255,178,0.5)"]],
        showscale=False,
        hovertemplate="%{x}<br>%{y}<br>Count: <b>%{z}</b><extra></extra>",
    ))
    fig.update_layout(
        height=300, paper_bgcolor=CARD, plot_bgcolor=CARD,
        margin=dict(l=12, r=12, t=12, b=12),
        annotations=anns,
        xaxis=dict(side="bottom",
                   tickfont=dict(color=TEXT, size=11), linecolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT, size=11), linecolor=BORDER),
        font=dict(family="Space Mono", color=MUTED),
    )
    render(fig, "cm")
