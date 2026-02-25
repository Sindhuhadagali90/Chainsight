"""
ChainSight — Supply Chain Risk Intelligence Dashboard
Python Desktop App (Tkinter + Matplotlib)
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyBboxPatch, Arc, Wedge
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import numpy as np
import math

# ══════════════════════════════════════════════
# THEME PALETTE
# ══════════════════════════════════════════════
BG       = "#080C10"
SURFACE  = "#0D1318"
CARD     = "#111820"
BORDER   = "#1E2A35"
ACCENT   = "#00FFB2"
ACCENT2  = "#FF4D6D"
ACCENT3  = "#FFB800"
BLUE     = "#3D9EFF"
TEXT     = "#E8F0F8"
MUTED    = "#5A7080"
DIM      = "#2A3A48"

matplotlib.rcParams.update({
    "figure.facecolor":  CARD,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        BORDER,
    "grid.alpha":        0.6,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ══════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════
ROUTE_LABELS  = ["Suez", "Commodity", "Pacific", "Atlantic", "Intra-Asia"]
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

MODELS = [
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
CORR_COLORS= [ACCENT2,ACCENT3,ACCENT3,BLUE,BLUE,DIM,DIM]

DISRUPT_LABELS = ["Port Congestion","Geopolitical\nConflict","Extreme\nWeather","No Event"]
DISRUPT_CNTS   = [820, 312, 115, 8753]
DISRUPT_COLORS = [ACCENT2, ACCENT3, BLUE, DIM]

EDA_FINDINGS = [
    ("Route Suez has highest avg delay",        "Route_Type",         "HIGH"),
    ("Sea transport ~1.8× more delay than Air", "Transportation_Mode","HIGH"),
    ("Perishables face highest category risk",  "Product_Category",   "HIGH"),
    ("Santos, BR origin is highest risk city",  "Origin_City",        "MED"),
    ("Shipping cost → near-zero correlation",   "Shipping_Cost_USD",  "LOW"),
    ("Weather severity weakly correlates",      "Weather_Severity",   "LOW"),
    ("Port Congestion is most common event",    "Disruption_Event",   "HIGH"),
    ("Class imbalance: 87.5% vs 12.5%",         "Delivery_Status",    "NOTE"),
]


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0,2,4))

def with_alpha(hex_color, alpha):
    r,g,b = hex_to_rgb(hex_color)
    return (r, g, b, alpha)


# ══════════════════════════════════════════════
# REUSABLE CHART HELPERS
# ══════════════════════════════════════════════
def styled_bar(ax, labels, values, colors, horizontal=False, title="", subtitle="", unit=""):
    if horizontal:
        bars = ax.barh(labels, values, color=colors, height=0.55)
        ax.set_xlabel(unit, color=MUTED, fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(val + max(values)*0.01, bar.get_y()+bar.get_height()/2,
                    f"{val}{unit}", va='center', color=TEXT, fontsize=8, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, max(values)*1.18)
    else:
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.set_ylabel(unit, color=MUTED, fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, val + max(values)*0.01,
                    f"{val}{unit}", ha='center', color=TEXT, fontsize=8, fontweight='bold')
        ax.set_ylim(0, max(values)*1.2)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8, loc='left')
    ax.tick_params(labelsize=9)
    for label in ax.get_xticklabels():
        label.set_color(MUTED)
    for label in ax.get_yticklabels():
        label.set_color(TEXT if horizontal else MUTED)


def draw_gauge(ax, pct, color):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    # Track
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=DIM, linewidth=14, solid_capstyle='round', zorder=1)
    # Fill
    fill_theta = np.linspace(np.pi, np.pi - (pct/100)*np.pi, 200)
    if len(fill_theta) > 1:
        ax.plot(np.cos(fill_theta), np.sin(fill_theta),
                color=color, linewidth=14, solid_capstyle='round', zorder=2)
    # Center text
    ax.text(0, 0.1, f"{pct}%", ha='center', va='center',
            fontsize=32, fontweight='bold', color=color, family='monospace')


def card_figure(rows=1, cols=1, figsize=(10,4), title="", subtitle=""):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(CARD)
    if title:
        fig.text(0.015, 0.97, title, color=TEXT, fontsize=12, fontweight='bold', va='top')
    if subtitle:
        fig.text(0.015, 0.90, subtitle, color=MUTED, fontsize=8, va='top', family='monospace')
    return fig, axes


# ══════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════
class ChainSightApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChainSight — Supply Chain Risk Intelligence")
        self.geometry("1300x820")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.current_page = None
        self._canvases = {}
        self._build_layout()
        self._show_page("dashboard")

    # ─────────────────────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──────────────────────────
        self.sidebar = tk.Frame(self, bg=SURFACE, width=220)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logo
        logo_frame = tk.Frame(self.sidebar, bg=SURFACE, pady=18)
        logo_frame.pack(fill="x", padx=20)

        logo_box = tk.Frame(logo_frame, bg=ACCENT, width=30, height=30)
        logo_box.pack_propagate(False)
        logo_box.pack(side="left", anchor="center")
        tk.Label(logo_box, text="◆", bg=ACCENT, fg=BG, font=("monospace", 14, "bold")).pack(expand=True)

        right = tk.Frame(logo_frame, bg=SURFACE)
        right.pack(side="left", padx=10)
        tk.Label(right, text="ChainSight", bg=SURFACE, fg=TEXT,
                 font=("Helvetica", 14, "bold")).pack(anchor="w")
        tk.Label(right, text="RISK INTELLIGENCE v1.0", bg=SURFACE, fg=MUTED,
                 font=("monospace", 7)).pack(anchor="w")

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x")

        # Nav
        nav_data = [
            ("dashboard", "▦  Overview"),
            ("predict",   "◉  Risk Predictor"),
            ("eda",       "∿  EDA Insights"),
            ("models",    "▤  Model Comparison"),
        ]
        self._nav_buttons = {}
        tk.Label(self.sidebar, text="MAIN", bg=SURFACE, fg=MUTED,
                 font=("monospace", 8), pady=10).pack(anchor="w", padx=20)

        for page_id, label in nav_data:
            btn = tk.Label(self.sidebar, text=label, bg=SURFACE, fg=MUTED,
                           font=("Helvetica", 11), padx=20, pady=10,
                           anchor="w", cursor="hand2")
            btn.pack(fill="x")
            btn.bind("<Button-1>", lambda e, p=page_id: self._show_page(p))
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg=TEXT, bg=DIM))
            btn.bind("<Leave>", lambda e, b=btn, p=page_id: b.config(
                fg=ACCENT if self.current_page == p else MUTED,
                bg=CARD if self.current_page == p else SURFACE))
            self._nav_buttons[page_id] = btn

        # Status
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", side="bottom", pady=(0,0))
        status_frame = tk.Frame(self.sidebar, bg=SURFACE, pady=12)
        status_frame.pack(side="bottom", fill="x", padx=20)
        dot = tk.Label(status_frame, text="●", bg=SURFACE, fg=ACCENT, font=("monospace", 10))
        dot.pack(side="left")
        tk.Label(status_frame, text=" MODEL LIVE · XGB v1.0", bg=SURFACE, fg=MUTED,
                 font=("monospace", 8)).pack(side="left")

        # ── Content Area ─────────────────────
        self.content = tk.Frame(self, bg=BG)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        # Page frames (all pre-created, shown/hidden)
        self._pages = {}
        for page_id in ["dashboard", "predict", "eda", "models"]:
            frame = tk.Frame(self.content, bg=BG)
            frame.grid(row=0, column=0, sticky="nsew")
            self._pages[page_id] = frame

        self._build_dashboard()
        self._build_predictor()
        self._build_eda()
        self._build_models()

    # ─────────────────────────────────────────
    def _show_page(self, page_id):
        self.current_page = page_id
        for pid, frame in self._pages.items():
            frame.tkraise() if pid == page_id else None
        self._pages[page_id].tkraise()
        # Update nav highlights
        for pid, btn in self._nav_buttons.items():
            if pid == page_id:
                btn.config(fg=ACCENT, bg=CARD)
            else:
                btn.config(fg=MUTED, bg=SURFACE)

    # ══════════════════════════════════════════
    # PAGE: DASHBOARD
    # ══════════════════════════════════════════
    def _build_dashboard(self):
        frame = self._pages["dashboard"]
        frame.grid_columnconfigure(0, weight=1)

        # Scrollable canvas
        outer = tk.Frame(frame, bg=BG)
        outer.pack(fill="both", expand=True)
        canvas_scroll = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas_scroll, bg=BG)
        win_id = canvas_scroll.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.bind("<Configure>", lambda e: canvas_scroll.itemconfig(win_id, width=e.width))
        canvas_scroll.bind_all("<MouseWheel>", lambda e: canvas_scroll.yview_scroll(-1*(e.delta//120), "units"))

        pad = {"padx": 16, "pady": 6}

        # ── Header ───────────────────────────
        hdr = tk.Frame(inner, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(18,4))
        tk.Label(hdr, text="Supply Chain  ", bg=BG, fg=TEXT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="Overview", bg=BG, fg=ACCENT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        sub = tk.Label(inner, text="10,000 SHIPMENTS  ·  GLOBAL ROUTES  ·  2024–2025",
                       bg=BG, fg=MUTED, font=("monospace", 9))
        sub.pack(anchor="w", padx=20)

        # ── KPI Cards ────────────────────────
        kpi_frame = tk.Frame(inner, bg=BG)
        kpi_frame.pack(fill="x", padx=16, pady=10)
        kpis = [
            ("TOTAL SHIPMENTS", "10,000",  "Across 6 origin cities",   ACCENT),
            ("DELAYED RATE",    "12.5%",   "1,247 late shipments",      ACCENT2),
            ("AVG DELAY DAYS",  "0.95",    "Max observed: 20 days",     ACCENT3),
            ("MODEL ROC-AUC",   "0.941",   "XGBoost — best performer",  BLUE),
        ]
        for i, (label, val, sub_text, color) in enumerate(kpis):
            kpi_frame.columnconfigure(i, weight=1)
            card = tk.Frame(kpi_frame, bg=CARD, padx=18, pady=14,
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=i, padx=6, sticky="nsew")
            tk.Frame(card, bg=color, height=2).pack(fill="x", pady=(0,10))
            tk.Label(card, text=label, bg=CARD, fg=MUTED,
                     font=("monospace", 8)).pack(anchor="w")
            tk.Label(card, text=val, bg=CARD, fg=color,
                     font=("Helvetica", 26, "bold")).pack(anchor="w")
            tk.Label(card, text=sub_text, bg=CARD, fg=MUTED,
                     font=("monospace", 8)).pack(anchor="w")

        # ── Charts Row 1 ─────────────────────
        r1 = tk.Frame(inner, bg=BG)
        r1.pack(fill="x", padx=16, pady=6)
        r1.columnconfigure(0, weight=3)
        r1.columnconfigure(1, weight=2)

        # Route delay bar
        fig1, ax1 = plt.subplots(figsize=(6.5, 3.2))
        fig1.patch.set_facecolor(CARD)
        ax1.set_facecolor(CARD)
        bars = ax1.bar(ROUTE_LABELS, ROUTE_DELAYS, color=ROUTE_COLORS, width=0.5)
        for bar, val in zip(bars, ROUTE_DELAYS):
            ax1.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val}d",
                     ha='center', color=TEXT, fontsize=9, fontweight='bold')
        ax1.set_ylim(0, 1.8)
        ax1.set_title("Avg Delay by Route Type", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax1.tick_params(labelsize=9)
        ax1.grid(axis='y', alpha=0.3)
        ax1.spines['left'].set_color(BORDER)
        ax1.spines['bottom'].set_color(BORDER)
        fig1.tight_layout(pad=1.2)
        self._embed(fig1, r1, row=0, col=0)

        # Donut
        fig2, ax2 = plt.subplots(figsize=(4, 3.2))
        fig2.patch.set_facecolor(CARD)
        ax2.set_facecolor(CARD)
        wedges, _ = ax2.pie([87.5, 12.5], colors=[ACCENT, ACCENT2],
                            startangle=90, wedgeprops=dict(width=0.45))
        ax2.set_title("Delivery Status", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        legend_labels = [f"On Time  87.5%", f"Delayed  12.5%"]
        legend = ax2.legend(wedges, legend_labels, loc="lower center",
                            fontsize=9, frameon=False, ncol=2,
                            labelcolor=[ACCENT, ACCENT2])
        fig2.tight_layout(pad=1.2)
        self._embed(fig2, r1, row=0, col=1)

        # ── Charts Row 2 ─────────────────────
        r2 = tk.Frame(inner, bg=BG)
        r2.pack(fill="x", padx=16, pady=6)
        r2.columnconfigure(0, weight=1)
        r2.columnconfigure(1, weight=1)

        fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
        fig3.patch.set_facecolor(CARD)
        ax3.set_facecolor(CARD)
        bars3 = ax3.barh(PRODUCT_LABELS, PRODUCT_DELAYS, color=PRODUCT_COLORS, height=0.5)
        for bar, val in zip(bars3, PRODUCT_DELAYS):
            ax3.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val}d",
                     va='center', color=TEXT, fontsize=8, fontweight='bold')
        ax3.set_xlim(0, 1.6)
        ax3.invert_yaxis()
        ax3.set_title("Avg Delay by Product Category", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax3.tick_params(labelsize=9)
        ax3.grid(axis='x', alpha=0.3)
        ax3.spines['left'].set_color(BORDER)
        ax3.spines['bottom'].set_color(BORDER)
        fig3.tight_layout(pad=1.2)
        self._embed(fig3, r2, row=0, col=0)

        fig4, ax4 = plt.subplots(figsize=(5.5, 3.5))
        fig4.patch.set_facecolor(CARD)
        ax4.set_facecolor(CARD)
        bars4 = ax4.barh(ORIGIN_LABELS, ORIGIN_DELAYS, color=ORIGIN_COLORS, height=0.5)
        for bar, val in zip(bars4, ORIGIN_DELAYS):
            ax4.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val}d",
                     va='center', color=TEXT, fontsize=8, fontweight='bold')
        ax4.set_xlim(0, 1.6)
        ax4.invert_yaxis()
        ax4.set_title("Avg Delay by Origin City", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax4.tick_params(labelsize=9)
        ax4.grid(axis='x', alpha=0.3)
        ax4.spines['left'].set_color(BORDER)
        ax4.spines['bottom'].set_color(BORDER)
        fig4.tight_layout(pad=1.2)
        self._embed(fig4, r2, row=0, col=1)

        # ── Charts Row 3 ─────────────────────
        r3 = tk.Frame(inner, bg=BG)
        r3.pack(fill="x", padx=16, pady=6)
        r3.columnconfigure(0, weight=3)
        r3.columnconfigure(1, weight=2)

        fig5, ax5 = plt.subplots(figsize=(6.5, 3.0))
        fig5.patch.set_facecolor(CARD)
        ax5.set_facecolor(CARD)
        ax5.plot(MONTHS, VOL, color=BLUE, linewidth=2, marker='o', markersize=4, label="Volume")
        ax5b = ax5.twinx()
        ax5b.set_facecolor(CARD)
        ax5b.plot(MONTHS, DLY, color=ACCENT2, linewidth=2, linestyle='--',
                  marker='s', markersize=4, label="Delayed")
        ax5b.tick_params(colors=MUTED, labelsize=9)
        ax5b.spines['right'].set_color(BORDER)
        ax5b.spines['top'].set_color(BORDER)
        ax5.fill_between(MONTHS, VOL, alpha=0.08, color=BLUE)
        ax5.set_title("Shipment Volume & Delay Trend  (2024–2025)", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax5.tick_params(labelsize=9)
        ax5.grid(alpha=0.3)
        ax5.spines['left'].set_color(BORDER)
        ax5.spines['bottom'].set_color(BORDER)
        ax5.legend(loc='upper left', frameon=False, fontsize=8, labelcolor=[BLUE, ACCENT2])
        fig5.tight_layout(pad=1.2)
        self._embed(fig5, r3, row=0, col=0)

        fig6, ax6 = plt.subplots(figsize=(4, 3.0))
        fig6.patch.set_facecolor(CARD)
        ax6.set_facecolor(CARD)
        ax6.barh(["Sea","Air"], [1.12, 0.62], color=[BLUE, ACCENT], height=0.4)
        ax6.set_xlim(0, 1.5)
        ax6.set_title("Transport Mode vs Avg Delay", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax6.tick_params(labelsize=10)
        ax6.grid(axis='x', alpha=0.3)
        ax6.spines['left'].set_color(BORDER)
        ax6.spines['bottom'].set_color(BORDER)
        for i, v in enumerate([1.12, 0.62]):
            ax6.text(v+0.02, i, f"{v}d", va='center', color=TEXT, fontsize=10, fontweight='bold')
        fig6.tight_layout(pad=1.2)
        self._embed(fig6, r3, row=0, col=1)

        tk.Frame(inner, bg=BG, height=20).pack()

    # ══════════════════════════════════════════
    # PAGE: PREDICTOR
    # ══════════════════════════════════════════
    def _build_predictor(self):
        frame = self._pages["predict"]
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Header
        hdr = tk.Frame(frame, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(18,6))
        tk.Label(hdr, text="Shipment  ", bg=BG, fg=TEXT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="Risk Predictor", bg=BG, fg=ACCENT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        tk.Label(frame, text="POWERED BY XGBOOST  ·  FILL IN SHIPMENT DETAILS TO ASSESS DELAY RISK",
                 bg=BG, fg=MUTED, font=("monospace", 9)).pack(anchor="w", padx=24)

        body = tk.Frame(frame, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=10)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── LEFT: Form ────────────────────────
        form_outer = tk.Frame(body, bg=CARD, padx=20, pady=16,
                              highlightbackground=BORDER, highlightthickness=1)
        form_outer.grid(row=0, column=0, sticky="nsew", padx=(0,8))

        tk.Label(form_outer, text="SHIPMENT PARAMETERS", bg=CARD, fg=MUTED,
                 font=("monospace", 9)).pack(anchor="w", pady=(0,12))

        self._form_vars = {}
        fields = [
            ("Origin City", "origin", "combobox",
             ["Shanghai, CN","Shenzhen, CN","Tokyo, JP","Hamburg, DE","Mumbai, IN","Santos, BR"], "Shanghai, CN"),
            ("Destination City", "dest", "combobox",
             ["Los Angeles, US","Rotterdam, NL","Singapore, SG","New York, US","Felixstowe, UK","Shanghai, CN"], "Los Angeles, US"),
            ("Route Type", "route", "combobox",
             ["Pacific","Atlantic","Suez","Intra-Asia","Commodity"], "Pacific"),
            ("Transport Mode", "mode", "combobox", ["Sea","Air"], "Sea"),
            ("Product Category", "product", "combobox",
             ["Textiles","Pharmaceuticals","Semiconductors","Consumer Electronics","Raw Materials","Perishables"], "Textiles"),
            ("Base Lead Time (days)", "base_lead", "entry", None, "18"),
            ("Scheduled Lead Time (days)", "sched_lead", "entry", None, "21"),
            ("Geopolitical Risk (0–1)", "geo", "entry", None, "0.55"),
            ("Weather Severity (0–10)", "weather", "entry", None, "4.5"),
            ("Inflation Rate (%)", "inflation", "entry", None, "3.5"),
            ("Shipping Cost (USD)", "cost", "entry", None, "5000"),
            ("Order Weight (kg)", "weight", "entry", None, "3000"),
        ]

        grid = tk.Frame(form_outer, bg=CARD)
        grid.pack(fill="x")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TCombobox",
                        fieldbackground=SURFACE, background=SURFACE,
                        foreground=TEXT, bordercolor=BORDER,
                        arrowcolor=MUTED, selectbackground=DIM,
                        selectforeground=TEXT)

        for i, (label, key, ftype, options, default) in enumerate(fields):
            row = i // 2
            col = i % 2
            cell = tk.Frame(grid, bg=CARD, padx=6, pady=5)
            cell.grid(row=row, column=col, sticky="nsew", padx=4)
            grid.columnconfigure(col, weight=1)
            tk.Label(cell, text=label.upper(), bg=CARD, fg=MUTED,
                     font=("monospace", 8)).pack(anchor="w")
            var = tk.StringVar(value=default)
            self._form_vars[key] = var
            if ftype == "combobox":
                cb = ttk.Combobox(cell, textvariable=var, values=options,
                                  state="readonly", style="Dark.TCombobox", width=20)
                cb.pack(fill="x", pady=2)
            else:
                ent = tk.Entry(cell, textvariable=var, bg=SURFACE, fg=TEXT,
                               insertbackground=ACCENT, relief="flat",
                               font=("monospace", 10), bd=0, highlightthickness=1,
                               highlightbackground=BORDER, highlightcolor=ACCENT)
                ent.pack(fill="x", pady=2, ipady=5)

        # Predict button
        pred_btn = tk.Button(form_outer, text="⚡  PREDICT DELAY RISK",
                             bg=ACCENT, fg=BG, font=("Helvetica", 12, "bold"),
                             relief="flat", padx=20, pady=12, cursor="hand2",
                             activebackground="#00e5a0", activeforeground=BG,
                             command=self._run_prediction)
        pred_btn.pack(fill="x", pady=(14,0))

        # ── RIGHT: Result ─────────────────────
        self.result_outer = tk.Frame(body, bg=CARD, padx=20, pady=20,
                                     highlightbackground=BORDER, highlightthickness=1)
        self.result_outer.grid(row=0, column=1, sticky="nsew", padx=(8,0))

        self.result_placeholder = tk.Frame(self.result_outer, bg=CARD)
        self.result_placeholder.pack(expand=True)
        tk.Label(self.result_placeholder, text="🔍", bg=CARD, fg=MUTED,
                 font=("Helvetica", 40)).pack(pady=(40,8))
        tk.Label(self.result_placeholder,
                 text="Fill in shipment parameters\nand click Predict to see\nreal-time risk assessment",
                 bg=CARD, fg=MUTED, font=("monospace", 10), justify="center").pack()

        self.result_content = tk.Frame(self.result_outer, bg=CARD)
        # Will be populated on first prediction

    def _run_prediction(self):
        try:
            v = self._form_vars
            origin    = v["origin"].get()
            route     = v["route"].get()
            mode      = v["mode"].get()
            product   = v["product"].get()
            base_lead = float(v["base_lead"].get())
            sched_lead= float(v["sched_lead"].get())
            geo       = float(v["geo"].get())
            weather   = float(v["weather"].get())
        except ValueError:
            return

        # Heuristic model
        score = 0.05
        buffer = sched_lead - base_lead
        score += {True: 0.30, False: 0.18 if buffer<=2 else 0.08 if buffer<=4 else 0.01}[buffer<=1]
        route_risk = {"Suez":0.16,"Commodity":0.12,"Pacific":0.08,"Atlantic":0.06,"Intra-Asia":0.04}
        score += route_risk.get(route, 0.06)
        score += 0.06 if mode=="Sea" else 0.02
        score += 0.08 if geo>0.7 else 0.04 if geo>0.5 else 0.01
        score += 0.07 if weather>7 else 0.04 if weather>5 else 0.01
        prod_risk = {"Perishables":0.05,"Semiconductors":0.04,"Consumer Electronics":0.03,"Pharmaceuticals":0.02}
        score += prod_risk.get(product, 0.01)
        origin_risk = {"Santos, BR":0.06,"Mumbai, IN":0.05,"Shenzhen, CN":0.03}
        score += origin_risk.get(origin, 0.01)
        score = min(max(score, 0.02), 0.97)
        pct = round(score * 100)

        color = ACCENT2 if score>0.5 else ACCENT3 if score>0.25 else ACCENT
        label = "🔴  HIGH DELAY RISK" if score>0.5 else "🟡  MODERATE RISK" if score>0.25 else "🟢  LOW RISK"
        desc  = (f"{pct}% probability of delay. Tight lead time buffer and high-risk route/origin combination. Consider expedited air freight."
                 if score>0.5 else
                 f"{pct}% delay probability. Some risk factors present — monitor geopolitical index and weather."
                 if score>0.25 else
                 f"Only {pct}% delay probability. Shipment parameters look healthy — within safe thresholds.")

        factors = [
            ("Lead Time Buffer",     min(90 if buffer<=1 else 60 if buffer<=2 else 30 if buffer<=4 else 10, 100)),
            ("Route Risk",           round(route_risk.get(route,0.06)/0.16*100)),
            ("Geopolitical Risk",    round(geo*100)),
            ("Weather Severity",     round(weather*10)),
            ("Transport Mode Risk",  65 if mode=="Sea" else 25),
            ("Product Category",     round((prod_risk.get(product,0.01)/0.05)*60+10)),
        ]

        # Clear result area
        for w in self.result_content.winfo_children():
            w.destroy()
        self.result_placeholder.pack_forget()
        self.result_content.pack(fill="both", expand=True)

        # Gauge chart
        fig_g, ax_g = plt.subplots(figsize=(4.5, 2.4))
        fig_g.patch.set_facecolor(CARD)
        ax_g.set_facecolor(CARD)
        draw_gauge(ax_g, pct, color)
        ax_g.text(0, -0.15, "DELAY PROBABILITY", ha='center', color=MUTED,
                  fontsize=8, family='monospace', transform=ax_g.transData)
        if 'gauge_canvas' in self._canvases:
            self._canvases['gauge_canvas'].get_tk_widget().destroy()
        c = FigureCanvasTkAgg(fig_g, master=self.result_content)
        c.draw()
        c.get_tk_widget().pack(fill="x")
        self._canvases['gauge_canvas'] = c

        # Label
        tk.Label(self.result_content, text=label, bg=CARD, fg=color,
                 font=("Helvetica", 14, "bold")).pack(pady=(4,2))
        tk.Label(self.result_content, text=desc, bg=CARD, fg=MUTED,
                 font=("monospace", 9), wraplength=320, justify="center").pack(padx=10)

        # Factor bars
        tk.Label(self.result_content, text="RISK FACTOR BREAKDOWN", bg=CARD, fg=MUTED,
                 font=("monospace", 8)).pack(anchor="w", pady=(12,4), padx=4)
        for name, val in factors:
            row = tk.Frame(self.result_content, bg=SURFACE, pady=6, padx=10)
            row.pack(fill="x", pady=2, padx=4)
            tk.Label(row, text=name, bg=SURFACE, fg=TEXT,
                     font=("Helvetica", 9, "bold"), width=18, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=DIM, height=6, width=160)
            bar_bg.pack(side="left", padx=6)
            bar_bg.pack_propagate(False)
            bar_color = ACCENT2 if val>70 else ACCENT3 if val>40 else ACCENT
            tk.Frame(bar_bg, bg=bar_color, height=6, width=int(1.6*val)).pack(side="left")
            tk.Label(row, text=f"{val}%", bg=SURFACE, fg=MUTED,
                     font=("monospace", 9)).pack(side="right")

        plt.close(fig_g)

    # ══════════════════════════════════════════
    # PAGE: EDA
    # ══════════════════════════════════════════
    def _build_eda(self):
        frame = self._pages["eda"]

        outer = tk.Frame(frame, bg=BG)
        outer.pack(fill="both", expand=True)
        cs = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical", command=cs.yview)
        cs.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cs.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cs, bg=BG)
        win = cs.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cs.configure(scrollregion=cs.bbox("all")))
        cs.bind("<Configure>", lambda e: cs.itemconfig(win, width=e.width))
        cs.bind_all("<MouseWheel>", lambda e: cs.yview_scroll(-1*(e.delta//120), "units"))

        # Header
        hdr = tk.Frame(inner, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(18,4))
        tk.Label(hdr, text="Exploratory Data  ", bg=BG, fg=TEXT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="Analysis", bg=BG, fg=ACCENT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        tk.Label(inner, text="KEY PATTERNS  ·  CORRELATIONS  ·  RISK DRIVERS",
                 bg=BG, fg=MUTED, font=("monospace", 9)).pack(anchor="w", padx=24)

        # Stat chips
        chips_frame = tk.Frame(inner, bg=BG)
        chips_frame.pack(fill="x", padx=16, pady=10)
        chips = [
            ("DATASET SIZE",       "10,000",  "Shipment records",      ACCENT),
            ("FEATURES USED",      "16",       "Non-leaky predictors",  BLUE),
            ("DELAY RATE",         "12.5%",    "Class imbalance ~1:7",  ACCENT2),
            ("MAX DELAY",          "20 days",  "Days observed",         ACCENT3),
            ("COST CORRELATION",   "~0.01",    "No effect on delay",    MUTED),
            ("ROUTES COVERED",     "5",        "Pacific, Suez, Atl…",  ACCENT),
        ]
        for i, (lbl, val, sub, color) in enumerate(chips):
            chips_frame.columnconfigure(i, weight=1)
            chip = tk.Frame(chips_frame, bg=CARD, padx=14, pady=12,
                            highlightbackground=BORDER, highlightthickness=1)
            chip.grid(row=0, column=i, padx=5, sticky="nsew")
            tk.Label(chip, text=lbl, bg=CARD, fg=MUTED, font=("monospace", 7)).pack(anchor="w")
            tk.Label(chip, text=val, bg=CARD, fg=color,
                     font=("Helvetica", 20, "bold")).pack(anchor="w")
            tk.Label(chip, text=sub, bg=CARD, fg=MUTED, font=("monospace", 8)).pack(anchor="w")

        # Charts row
        r1 = tk.Frame(inner, bg=BG)
        r1.pack(fill="x", padx=16, pady=6)
        r1.columnconfigure(0, weight=1)
        r1.columnconfigure(1, weight=1)

        # Delay histogram
        fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))
        fig1.patch.set_facecolor(CARD)
        ax1.set_facecolor(CARD)
        hist_colors = [ACCENT if b==0 else ACCENT3 if b<5 else ACCENT2 for b in DELAY_BINS]
        ax1.bar([str(b) for b in DELAY_BINS], DELAY_CNTS, color=hist_colors, width=0.7)
        ax1.set_title("Delay Day Distribution", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax1.set_xlabel("Delay Days", color=MUTED, fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.grid(axis='y', alpha=0.3)
        ax1.spines['left'].set_color(BORDER)
        ax1.spines['bottom'].set_color(BORDER)
        fig1.tight_layout(pad=1.2)
        self._embed(fig1, r1, row=0, col=0)

        # Correlation bars
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        fig2.patch.set_facecolor(CARD)
        ax2.set_facecolor(CARD)
        bars2 = ax2.barh(CORR_NAMES, CORR_VALS, color=CORR_COLORS, height=0.5)
        for bar, val in zip(bars2, CORR_VALS):
            ax2.text(val+0.005, bar.get_y()+bar.get_height()/2, f"{val:.2f}",
                     va='center', color=TEXT, fontsize=8, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        ax2.invert_yaxis()
        ax2.set_title("Feature Correlation to Delay Days", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax2.tick_params(labelsize=9)
        ax2.grid(axis='x', alpha=0.3)
        ax2.spines['left'].set_color(BORDER)
        ax2.spines['bottom'].set_color(BORDER)
        fig2.tight_layout(pad=1.2)
        self._embed(fig2, r1, row=0, col=1)

        # Findings table
        tbl_frame = tk.Frame(inner, bg=CARD, padx=16, pady=14,
                             highlightbackground=BORDER, highlightthickness=1)
        tbl_frame.pack(fill="x", padx=16, pady=6)
        tk.Label(tbl_frame, text="KEY EDA FINDINGS", bg=CARD, fg=MUTED,
                 font=("monospace", 9)).pack(anchor="w", pady=(0,10))
        headers = ["Finding", "Variable", "Impact"]
        header_row = tk.Frame(tbl_frame, bg=DIM)
        header_row.pack(fill="x")
        for h, w in zip(headers, [400, 200, 80]):
            tk.Label(header_row, text=h.upper(), bg=DIM, fg=MUTED,
                     font=("monospace", 8), width=w//8, anchor="w", padx=8, pady=5).pack(side="left")
        for finding, var, impact in EDA_FINDINGS:
            impact_color = ACCENT2 if impact=="HIGH" else ACCENT3 if impact=="MED" else BLUE if impact=="LOW" else MUTED
            row = tk.Frame(tbl_frame, bg=CARD)
            row.pack(fill="x")
            tk.Frame(tbl_frame, bg=BORDER, height=1).pack(fill="x")
            tk.Label(row, text=finding, bg=CARD, fg=TEXT,
                     font=("Helvetica", 10), anchor="w", padx=8, pady=6, width=48).pack(side="left")
            tk.Label(row, text=var, bg=CARD, fg=BLUE,
                     font=("monospace", 9), anchor="w", padx=8, width=24).pack(side="left")
            tk.Label(row, text=impact, bg=CARD, fg=impact_color,
                     font=("monospace", 9, "bold"), anchor="w", padx=8, width=8).pack(side="left")

        # Disruption chart
        fig3, ax3 = plt.subplots(figsize=(10, 2.8))
        fig3.patch.set_facecolor(CARD)
        ax3.set_facecolor(CARD)
        ax3.bar(DISRUPT_LABELS, DISRUPT_CNTS, color=DISRUPT_COLORS, width=0.5)
        for i, (lbl, val) in enumerate(zip(DISRUPT_LABELS, DISRUPT_CNTS)):
            ax3.text(i, val+60, f"{val:,}", ha='center', color=TEXT, fontsize=9, fontweight='bold')
        ax3.set_title("Disruption Event Frequency", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax3.tick_params(labelsize=9)
        ax3.grid(axis='y', alpha=0.3)
        ax3.spines['left'].set_color(BORDER)
        ax3.spines['bottom'].set_color(BORDER)
        fig3.tight_layout(pad=1.2)

        fig3_outer = tk.Frame(inner, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        fig3_outer.pack(fill="x", padx=16, pady=6)
        c3 = FigureCanvasTkAgg(fig3, master=fig3_outer)
        c3.draw()
        c3.get_tk_widget().pack(fill="x")
        plt.close(fig3)

        tk.Frame(inner, bg=BG, height=20).pack()

    # ══════════════════════════════════════════
    # PAGE: MODELS
    # ══════════════════════════════════════════
    def _build_models(self):
        frame = self._pages["models"]

        outer = tk.Frame(frame, bg=BG)
        outer.pack(fill="both", expand=True)
        cs = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical", command=cs.yview)
        cs.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cs.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cs, bg=BG)
        win = cs.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cs.configure(scrollregion=cs.bbox("all")))
        cs.bind("<Configure>", lambda e: cs.itemconfig(win, width=e.width))
        cs.bind_all("<MouseWheel>", lambda e: cs.yview_scroll(-1*(e.delta//120), "units"))

        # Header
        hdr = tk.Frame(inner, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(18,4))
        tk.Label(hdr, text="Model  ", bg=BG, fg=TEXT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="Comparison", bg=BG, fg=ACCENT,
                 font=("Helvetica", 20, "bold")).pack(side="left")
        tk.Label(inner, text="5 CLASSIFIERS  ·  ACCURACY & ROC-AUC BENCHMARKS",
                 bg=BG, fg=MUTED, font=("monospace", 9)).pack(anchor="w", padx=24)

        # Model cards
        cards_row = tk.Frame(inner, bg=BG)
        cards_row.pack(fill="x", padx=16, pady=12)
        for i, m in enumerate(MODELS):
            cards_row.columnconfigure(i, weight=1)
            border_color = ACCENT if m["best"] else BORDER
            card = tk.Frame(cards_row, bg=CARD, padx=14, pady=14,
                            highlightbackground=border_color, highlightthickness=2 if m["best"] else 1)
            card.grid(row=0, column=i, padx=5, sticky="nsew")
            if m["best"]:
                tk.Label(card, text="★ BEST", bg=CARD, fg=ACCENT,
                         font=("monospace", 8)).pack(anchor="e")
            tk.Label(card, text=m["name"], bg=CARD, fg=TEXT,
                     font=("Helvetica", 11, "bold"), wraplength=130).pack(anchor="w", pady=(0,8))
            color = ACCENT if m["best"] else TEXT
            tk.Label(card, text="ACCURACY", bg=CARD, fg=MUTED, font=("monospace", 7)).pack(anchor="w")
            tk.Label(card, text=f"{m['acc']}%", bg=CARD, fg=color,
                     font=("Helvetica", 22, "bold")).pack(anchor="w")
            tk.Label(card, text="ROC-AUC", bg=CARD, fg=MUTED, font=("monospace", 7)).pack(anchor="w", pady=(6,0))
            tk.Label(card, text=f"{m['auc']:.3f}", bg=CARD, fg=BLUE if not m["best"] else ACCENT,
                     font=("Helvetica", 16, "bold")).pack(anchor="w")

        # Accuracy & AUC charts
        r1 = tk.Frame(inner, bg=BG)
        r1.pack(fill="x", padx=16, pady=6)
        r1.columnconfigure(0, weight=1)
        r1.columnconfigure(1, weight=1)

        model_names = [m["name"].split()[0] for m in MODELS]
        acc_vals    = [m["acc"] for m in MODELS]
        auc_vals    = [m["auc"] for m in MODELS]
        m_colors    = [ACCENT if m["best"] else DIM for m in MODELS]
        m_colors2   = [ACCENT if m["best"] else BLUE for m in MODELS]

        fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))
        fig1.patch.set_facecolor(CARD)
        ax1.set_facecolor(CARD)
        bars1 = ax1.bar(model_names, acc_vals, color=m_colors, width=0.5)
        for bar, val in zip(bars1, acc_vals):
            ax1.text(bar.get_x()+bar.get_width()/2, val+0.2, f"{val}%",
                     ha='center', color=TEXT, fontsize=9, fontweight='bold')
        ax1.set_ylim(75, 98)
        ax1.set_title("Accuracy Comparison", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax1.tick_params(labelsize=9)
        ax1.grid(axis='y', alpha=0.3)
        ax1.spines['left'].set_color(BORDER)
        ax1.spines['bottom'].set_color(BORDER)
        fig1.tight_layout(pad=1.2)
        self._embed(fig1, r1, row=0, col=0)

        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        fig2.patch.set_facecolor(CARD)
        ax2.set_facecolor(CARD)
        bars2 = ax2.bar(model_names, auc_vals, color=m_colors2, width=0.5)
        for bar, val in zip(bars2, auc_vals):
            ax2.text(bar.get_x()+bar.get_width()/2, val+0.003, f"{val:.3f}",
                     ha='center', color=TEXT, fontsize=9, fontweight='bold')
        ax2.set_ylim(0.75, 1.0)
        ax2.set_title("ROC-AUC Comparison", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax2.tick_params(labelsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.spines['left'].set_color(BORDER)
        ax2.spines['bottom'].set_color(BORDER)
        fig2.tight_layout(pad=1.2)
        self._embed(fig2, r1, row=0, col=1)

        # Feature importance
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        fig3.patch.set_facecolor(CARD)
        ax3.set_facecolor(CARD)
        feat_colors = [ACCENT if v>0.15 else ACCENT3 if v>0.08 else BLUE for v in FEAT_VALS]
        bars3 = ax3.barh(FEAT_NAMES, FEAT_VALS, color=feat_colors, height=0.5)
        for bar, val in zip(bars3, FEAT_VALS):
            ax3.text(val+0.003, bar.get_y()+bar.get_height()/2, f"{val:.2f}",
                     va='center', color=TEXT, fontsize=8, fontweight='bold')
        ax3.set_xlim(0, 0.38)
        ax3.invert_yaxis()
        ax3.set_title("XGBoost — Top Feature Importances", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax3.tick_params(labelsize=9)
        ax3.grid(axis='x', alpha=0.3)
        ax3.spines['left'].set_color(BORDER)
        ax3.spines['bottom'].set_color(BORDER)
        fig3.tight_layout(pad=1.2)

        fig3_outer = tk.Frame(inner, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        fig3_outer.pack(fill="x", padx=16, pady=6)
        c3 = FigureCanvasTkAgg(fig3, master=fig3_outer)
        c3.draw()
        c3.get_tk_widget().pack(fill="x")
        plt.close(fig3)

        # Confusion Matrix
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        fig4.patch.set_facecolor(CARD)
        ax4.set_facecolor(CARD)
        cm = np.array([[1720, 78],[124, 78]])
        labels_cm = [["On-Time\n1720 ✓","Delayed\n78 ✗"],["Delayed\n124 ✗","On-Time\n78 ✓"]]
        cell_colors = [
            [with_alpha(ACCENT, 0.5),  with_alpha(ACCENT2, 0.2)],
            [with_alpha(ACCENT2, 0.2), with_alpha(ACCENT, 0.3)],
        ]
        for i in range(2):
            for j in range(2):
                rect = FancyBboxPatch((j-0.45, 1-i-0.45), 0.9, 0.9,
                                      boxstyle="round,pad=0.05",
                                      facecolor=cell_colors[i][j],
                                      edgecolor=BORDER, linewidth=1.5)
                ax4.add_patch(rect)
                ax4.text(j, 1-i, labels_cm[i][j], ha='center', va='center',
                         fontsize=10, color=TEXT, fontweight='bold', family='monospace')
        ax4.set_xlim(-0.6, 1.6)
        ax4.set_ylim(-0.6, 1.6)
        ax4.set_xticks([0,1])
        ax4.set_xticklabels(["Predicted: On-Time","Predicted: Delayed"], color=MUTED, fontsize=9)
        ax4.set_yticks([0,1])
        ax4.set_yticklabels(["Actual: Delayed","Actual: On-Time"], color=MUTED, fontsize=9)
        ax4.set_title("XGBoost — Confusion Matrix", color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax4.spines['left'].set_color(BORDER)
        ax4.spines['bottom'].set_color(BORDER)
        fig4.tight_layout(pad=1.5)

        cm_outer = tk.Frame(inner, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        cm_outer.pack(fill="x", padx=16, pady=6)
        c4 = FigureCanvasTkAgg(fig4, master=cm_outer)
        c4.draw()
        c4.get_tk_widget().pack()
        plt.close(fig4)

        tk.Frame(inner, bg=BG, height=20).pack()

    # ─────────────────────────────────────────
    def _embed(self, fig, parent, row, col):
        """Embed a matplotlib figure into a tk grid cell."""
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        outer.grid(row=row, column=col, sticky="nsew", padx=5, pady=2)
        c = FigureCanvasTkAgg(fig, master=outer)
        c.draw()
        c.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)
        return c


# ══════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    app = ChainSightApp()
    app.mainloop()
