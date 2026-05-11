# ================================================================
# Farm Productivity & Health Dashboard (R22 – XGBoost, M1, C2)
# ================================================================

import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import math
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit.components.v1 as components
from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plotly.subplots import make_subplots

from xgboost import XGBRegressor

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4



# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Farm Productivity & Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# SESSION STATE – THEME + PLOTLY TEMPLATE
# ================================================================
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark"

if "plotly_template" not in st.session_state:
    st.session_state.plotly_template = "plotly_dark"


# ================================================================
# SIDEBAR – THEME + DEV MODE + NAV
# ================================================================
st.sidebar.title("🐷 Farm Productivity & Health Dashboard")

theme_choice = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"],
    index=["Dark", "Light"].index(st.session_state.theme_choice),
    horizontal=True,
    key="theme_selector",
)
st.session_state.theme_choice = theme_choice

# ================================================================
# GLOBAL DEV MODE CHECKBOX (Single Source of Truth)
# ================================================================
st.sidebar.checkbox(
    "Developer Mode (show extra technical details)",
    key="dev_mode_enabled",
    value=False,
)

dev_mode = st.session_state.get("dev_mode_enabled", False)


page = st.sidebar.radio(
    "Go to page:",
    [
        "📌 Executive Summary",
        "📊 Descriptive Analytics",
        "📈 Predictive Analytics",
        "🧬 K-Means Segmentation",
        "🏭 Production Operations",
        "💰 Harvest & Revenue Simulator",
        "🧠 Explainable AI (SHAP)",
        "🤖 Ask the Farm AI",
        "📚 Data Dictionary",
        "🔗 Data Lineage",
        "🤖 AI/ML Journey",
        
    ],
)

# global CSS 

if st.session_state.theme_choice == "Dark":
    st.session_state.plotly_template = "plotly_dark"
    theme_css = """
    /* ADD THIS: Force labels and headers to be white in Dark Mode */
    .stWidgetLabel p, h2, .stMarkdown p {
        color: #FFFFFF !important;
    }
    
    /* Ensure input text is readable */
    input {
        color: #FFFFFF !important;
    }
    """
    st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)
    /* ============================================================
       GLOBAL DARK MODE BACKGROUND + HEADER / PADDING FIX
       ============================================================ */
    html, body, .stApp, [data-testid="stAppViewContainer"] 
                            padding-top: 0 !important;
                        }

    /* Remove the white strip at the very top */
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0rem !important;
        margin-top: 0 !important;
    }
    /* ----- REMOVE DEFAULT STREAMLIT TOP PADDING ----- */
    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Hide default Streamlit header background */
    header, [data-testid="stHeader"] {
        background-color: transparent !;
        height: 28px !important;        /* required for Deploy button */
        min-height: 28px !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        }

        /* ---- Make sure main page does not overlap toolbar ---- */
    [data-testid="stAppViewContainer"] {
        padding-top: 0px !important;   /* clear space below Deploy bar */
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #030712 !important;
        color: #e5e7eb !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
        padding-top: 32 !important;  aligns sidebar with main content
    }

    /* Base text colour so sliders / labels etc. are visible */
    .stApp, .stApp * {
        color: #e5e7eb !important;
    }

    /* ============================================================
       EXPANDERS – FORCE DARK IN ALL STATES (FIXES WHITE HEADER)
       ============================================================ */

    /* Outer expander container (Streamlit uses both div+details
       and details[data-testid="st-expander"] in different builds) */
    div[data-testid="stExpander"],
    details[data-testid="st-expander"] {
        background-color: #020617 !important;
        border-radius: 8px !important;
        border: 1px solid #1f2937 !important;
        overflow: hidden !important;
    }

    /* Header row (collapsed) */
    div[data-testid="stExpander"] > details > summary,
    details[data-testid="st-expander"] > summary {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        padding: 0.65rem 1rem !important;
        border-bottom: 1px solid #1f2937 !important;
        cursor: pointer;
    }

    /* Header row when OPEN – stop it turning light grey */
    div[data-testid="stExpander"] > details[open] > summary,
    details[data-testid="st-expander"][open] > summary {
        background-color: #020617 !important;
        color: #e5e7eb !important;
    }

    /* Remove focus / active highlight that brings back white */
    details[data-testid="st-expander"] > summary:focus,
    details[data-testid="st-expander"] > summary:focus-visible,
    div[data-testid="stExpander"] > details > summary:focus,
    div[data-testid="stExpander"] > details > summary:focus-visible {
        outline: none !important;
        box-shadow: none !important;
        background-color: #020617 !important;
        color: #e5e7eb !important;
    }

    /* Hover state (slightly lighter, but still dark) */
    div[data-testid="stExpander"] > details > summary:hover,
    details[data-testid="st-expander"] > summary:hover {
        background-color: #111827 !important;
        color: #e5e7eb !important;
    }

    /* Expander body/content */
    div[data-testid="stExpander"] > div,
    details[data-testid="st-expander"] > div {
        background-color: #030712 !important;
        color: #e5e7eb !important;
    }

    /* ============================================================
       SLIDER LABEL VISIBILITY (Daily feed intake, Age, etc.)
       ============================================================ */
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] span {
        color: #e5e7eb !important;
    }

    /* ============================================================
       METRIC + BUTTON STYLING
       ============================================================ */
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #cbd5e1 !important; }
    [data-testid="stMetricDelta"] { color: #22c55e !important; }

    .stButton > button {
        background-color: #111827 !important;
        color: #e5e7eb !important;
        border: 1px solid #6366f1 !important;
        border-radius: 999px !important;
        padding: 0.4rem 1rem !important;
        transition: 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #1f2937 !important;
        box-shadow: 0 0 18px rgba(99,102,241,0.5);
        transform: translateY(-1px);
    }
    /* ============================================================
   DARK MODE — FIX SELECTBOX / DROPDOWN BACKGROUND + TEXT
   ============================================================ */

/* Selectbox container */
div[data-baseweb="select"] {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
    border-radius: 6px !important;
}

/* The clickable select control */
div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
    border-color: #1f2937 !important;
}

/* The text inside the select box */
div[data-baseweb="select"] span {
    color: #e5e7eb !important;
}

/* Dropdown menu list */
ul[role="listbox"] {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
    border: 1px solid #1f2937 !important;
}

/* Dropdown options */
ul[role="listbox"] li {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
}

/* Hover highlight on options */
ul[role="listbox"] li:hover {
    background-color: #1f2937 !important;
    color: #ffffff !important;
}
    """

else:
    st.session_state.plotly_template = "plotly_white"
    theme_css = """
  
    /* Main app background (soft neutral light tone) */
    .stApp, html, body, [data-testid="stAppViewContainer"] {
        background-color: #f7fbff !important;
        color: #111827 !important;
        padding-top: 32px !important;   /* prevents Deploy clipping */
    }
    /* ----- REMOVE DEFAULT STREAMLIT TOP PADDING ----- */
    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Sidebar matches main background */
    section[data-testid="stSidebar"] {
        background-color: #f7fbff !important;
    }
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    /* HEADER FIX — do NOT hide header; just flatten it */
    header, [data-testid="stHeader"] {
        background-color: transparent !important;
        height: 28px !important;        /* needed for Deploy button */
        min-height: 28px !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #111827 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1d4ed8 !important;
        color: white !important;
        padding: 0.4rem 1rem !important;
        border-radius: 999px !important;
        border: none !important;
        transition: 0.3s !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 14px rgba(37,99,235,0.8);
        transform: scale(1.02);
    }

    /* Metrics */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: #111827 !important;
    }
    """


st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)
plotly_template = st.session_state.plotly_template

# ---- THEME-AWARE COLORS (BRIGHTER VERSION) ----
is_dark = st.session_state.plotly_template == "plotly_dark"

GAUGE_GREEN = "rgba(0, 200, 50, 0.9)"       # bright green
GAUGE_YELLOW = "rgba(255, 200, 0, 0.9)"     # bright yellow
GAUGE_RED = "rgba(255, 60, 60, 0.9)"         # bright red

CARD_BG = "#0f172a" if is_dark else "#ffffff"
CARD_TEXT = "#ffffff" if is_dark else "#0f172a"
CARD_SUBTEXT = "#cbd5e1" if is_dark else "#475569"
CARD_BORDER = "#1e293b" if is_dark else "#e2e8f0"



# ================================================================
# GLOBAL HELPER FUNCTIONS (STABLE + SAFE + PRODUCTION READY)
# ================================================================

def safe_run(title, func):
    """
    Prevent K-Means Developer Mode from crashing the entire page.
    Shows a warning instead of breaking the UI.
    """
    try:
        func()
    except Exception as e:
        st.warning(f"⚠️ {title} failed: {e}")


def clean_barn_code(x):
    """
    Normalize barn IDs to avoid mismatches:
    e.g., 'C 1', 'C1 ', ' c1', ' C01 ' → 'C1'
    """
    if pd.isna(x):
        return None
    x = str(x).upper().strip()
    x = re.sub(r"\s+", "", x)  # remove all whitespace
    x = re.sub(r"^0+", "", x)  # remove leading zeros
    return x


# ================================================================
# GLOBAL HELPER FUNCTIONS (MUST BE ABOVE ALL PAGE CODE)
# ================================================================

def safe_run(title, func):
    try:
        func()
    except Exception as e:
        st.warning(f"⚠️ {title} failed: {e}")

def clean_barn_code(x):
    if pd.isna(x):
        return None
    x = str(x).upper().strip()
    x = re.sub(r"\s+", "", x)
    return x

def show_k_diagnostics(X_scaled):
    """Developer Mode: Elbow + Silhouette Charts"""
    with st.expander("🧪 Developer Diagnostics (Elbow & Silhouette)", expanded=False):

        if X_scaled.shape[0] < 3:
            st.warning("Not enough data for diagnostics.")
            return

        Ks = range(2, min(8, X_scaled.shape[0]))

        # ------------- Elbow -------------
        inertias = []
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            inertias.append(km.inertia_)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(Ks), y=inertias, mode="lines+markers"))
        fig1.update_layout(
            template=st.session_state.plotly_template,
            height=300,
            xaxis_title="k",
            yaxis_title="Inertia"
        )
        st.subheader("Elbow Method")
        st.plotly_chart(fig1, width="stretch")

               
        # ----------------------------------------------------
        # 2️⃣ SILHOUETTE SCORE (only valid when k < n_samples)
        # ----------------------------------------------------
        sil_scores = []
        for k in Ks:
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
                sil = silhouette_score(X_scaled, km.labels_)
                sil_scores.append(sil)
            except Exception:
                sil_scores.append(None)

        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(x=list(Ks), y=sil_scores))
        fig_sil.update_layout(
            template=plotly_template,
            height=320,
            xaxis_title="k",
            yaxis_title="Silhouette Score (higher = better)",
        )

        st.subheader("Silhouette Method")
        st.plotly_chart(fig_sil, width="stretch")
        # --------------------------
        # Interpretation Note
        # --------------------------
        st.markdown("""
        ### 📝 Interpretation Summary
        
        **Elbow Method:**  
        - The inertia curve drops sharply from **K=2 → K=3**.  
        - After **K=3**, improvements become marginal.  
        ➜ **This indicates K=3 is the optimal value.**
        
        **Silhouette Score:**  
        - **K=2** shows the highest silhouette score (strong separation).  
        - However, K=2 produces overly broad clusters that are **not useful for farm management**.
        
        **Final Recommendation:**  
        ➜ **K=3 provides the best balance between model quality and actionable segmentation.**  
        It produces meaningful groups:
        
        • **Cluster 0 — High performers**  
        • **Cluster 1 — Average performers**  
        • **Cluster 2 — Underperforming / High-risk barns**
        
        This segmentation enables more targeted interventions across barns and growth stages.
        """)
# ================================================================
# DATA DICTIONARY & LINEAGE HELPERS
# ================================================================

def _load_excel_dictionary(path: str = "Data_Dictionary (Sharing).xlsx"):
    """
    Try to load your external data dictionary from Excel.
    Returns a DataFrame with at least: Field, Description (if possible),
    or None if file not found / can't be read.
    """
    excel_path = Path(path)
    if not excel_path.exists():
        return None

    try:
        raw = pd.read_excel(excel_path)
    except Exception:
        return None

    # Normalise column names
    cols = {c.lower(): c for c in raw.columns}

    # Try to detect "field" column
    field_col = None
    for key in ["field_name", "field", "column", "name", "field_rename"]:
        for c in raw.columns:
            if key in c.lower():
                field_col = c
                break
        if field_col:
            break
    if field_col is None:
        # fallback = first column
        field_col = raw.columns[0]

    # Try to detect "description" column
    desc_col = None
    for key in ["description", "field_description", "field_descr", "field_descrip", "desc"]:
        for c in raw.columns:
            if key in c.lower():
                desc_col = c
                break
        if desc_col:
            break

    # Try to detect "type" column (optional)
    type_col = None
    for key in ["type", "datatype", "data_type"]:
        for c in raw.columns:
            if key in c.lower():
                type_col = c
                break
        if type_col:
            break

    out = pd.DataFrame()
    out["Field"] = raw[field_col].astype(str).str.strip()
    if desc_col:
        out["Description"] = raw[desc_col].astype(str).str.strip()
    else:
        out["Description"] = ""
    if type_col:
        out["Datatype"] = raw[type_col].astype(str).str.strip()
    else:
        out["Datatype"] = ""
    return out


def _infer_category(col: str) -> str:
    """Roughly classify fields into logical groups for the dictionary."""
    name = col.upper()

    # Raw farm ops
    if any(k in name for k in ["FARM_ID", "BARN", "RAISE_DAY", "RAISE_WEEK",
                               "BEGIN_POP", "ANIMAL_IN", "DEATH", "CULL"]):
        return "Raw Farm Operations"

    # Growth
    if "ADG" in name or "WEIGHT" in name or "ESTIMATED_WEIGHT" in name:
        return "Growth & Performance"

    # Feed
    if "FEED" in name:
        return "Feed & Efficiency"

    # Climate / temp / heat
    if "TEMP" in name or "HEAT" in name:
        return "Climate & Heat Stress"

    # Health / respiratory / reason
    if "RESP" in name or "HEALTH" in name or "REASON" in name:
        return "Health & Risk"

    # Barn performance scores / unified metrics
    if any(k in name for k in ["UNIFIED_SCORE", "SAMPLE_STRENGTH", "AGE_COVERAGE",
                               "HEALTH_STABILITY", "SCORE"]):
        return "Barn Performance KPIs"

    # Segmentation
    if "CLUSTER" in name or "SEGMENT" in name:
        return "Segmentation & K-Means"

    # Encoded / model features
    if "ENC" in name or "TARGET_ENC" in name or name.startswith("FEAT_"):
        return "ML Feature Engineering"

    return "Other / General"


def _infer_table(col: str, df: pd.DataFrame, barn_perf: pd.DataFrame | None) -> str:
    in_df = col in df.columns
    in_barn = barn_perf is not None and col in barn_perf.columns

    if in_df and in_barn:
        return "Main Dataset (df) + Barn Performance (barn_perf)"
    if in_df:
        return "Main Dataset (df)"
    if in_barn:
        return "Barn Performance (barn_perf)"
    return "Other / Derived"


def _auto_description(col: str) -> str:
    """Fallback description if not supplied in Excel."""
    name = col.upper()
    if name == "ADG":
        return "Average Daily Gain (weight / days raised)"
    if "RESPIRATORY_PERCENT" in name:
        return "Percentage of animals showing respiratory symptoms"
    if "RESPIRATORY_PERCENT_FLAG" in name:
        return "Indicator flag when respiratory percentage exceeds threshold"
    if "HEAT_STRESS" in name:
        return "Heat stress indicator based on temperature difference"
    if "TEMP_DIFF" in name:
        return "Difference between max and min temperature within a period"
    if "UNIFIED_SCORE" in name:
        return "Composite barn performance score combining growth, health and sample strength"
    if "MEDIAN_ADG" in name:
        return "Median Average Daily Gain per barn"
    if "FEED_EFF" in name:
        return "Feed efficiency metric (e.g. weight gained per unit of feed)"
    if "BARN_CLEAN" in name:
        return "Normalised barn identifier used for clustering and reporting"
    if "CLUSTER_LABEL" in name:
        return "Human-readable label assigned to K-Means cluster"
    return "Description TBD (derived in dashboard)"


def build_enhanced_data_dictionary(
    df: pd.DataFrame,
    barn_perf: pd.DataFrame | None,
    dict_path: str = "Data_Dictionary (Sharing).xlsx",
) -> pd.DataFrame:
    """
    Merge:
      - External Excel data dictionary (if present)
      - Columns from df
      - Columns from barn_perf
    into one enhanced dictionary.
    """
    external = _load_excel_dictionary(dict_path)

    # Base: all unique fields from df + barn_perf
    cols = set(df.columns)
    if barn_perf is not None:
        cols.update(barn_perf.columns)
    all_fields = sorted(cols)

    # External lookup for descriptions
    external_desc = {}
    external_dtype = {}
    if external is not None:
        for _, row in external.iterrows():
            key = str(row["Field"]).strip().upper()
            external_desc[key] = row.get("Description", "")
            external_dtype[key] = row.get("Datatype", "")

    rows = []
    for col in all_fields:
        key = col.upper()

        desc = external_desc.get(key, "").strip()
        dtype = external_dtype.get(key, "").strip()

        # If no datatype from external, infer from df / barn_perf
        if not dtype:
            if col in df.columns:
                dtype = str(df[col].dtype)
            elif barn_perf is not None and col in barn_perf.columns:
                dtype = str(barn_perf[col].dtype)
            else:
                dtype = "-"

        # If no description at all, auto-generate a helpful one
        if not desc:
            desc = _auto_description(col)

        category = _infer_category(col)
        table = _infer_table(col, df, barn_perf if barn_perf is not None else pd.DataFrame())

        rows.append(
            {
                "Field": col,
                "Description": desc,
                "Datatype": dtype,
                "Category": category,
                "Table / Source": table,
            }
        )

    dd = pd.DataFrame(rows)
    return dd.sort_values(["Category", "Field"])

# ================================================================
# DATA DICTIONARY & LINEAGE HELPERS
# ================================================================


def _load_excel_dictionary(path: str = "Data_Dictionary (Sharing).xlsx"):
    """
    Try to load your external data dictionary from Excel.
    Returns a DataFrame with at least: Field, Description (if possible),
    or None if file not found / can't be read.
    """
    excel_path = Path(path)
    if not excel_path.exists():
        return None

    try:
        raw = pd.read_excel(excel_path)
    except Exception:
        return None

    # Normalise column names
    cols = {c.lower(): c for c in raw.columns}

    # Try to detect "field" column
    field_col = None
    for key in ["field_name", "field", "column", "name", "field_rename"]:
        for c in raw.columns:
            if key in c.lower():
                field_col = c
                break
        if field_col:
            break
    if field_col is None:
        # fallback = first column
        field_col = raw.columns[0]

    # Try to detect "description" column
    desc_col = None
    for key in ["description", "field_description", "field_descr", "field_descrip", "desc"]:
        for c in raw.columns:
            if key in c.lower():
                desc_col = c
                break
        if desc_col:
            break

    # Try to detect "type" column (optional)
    type_col = None
    for key in ["type", "datatype", "data_type"]:
        for c in raw.columns:
            if key in c.lower():
                type_col = c
                break
        if type_col:
            break

    out = pd.DataFrame()
    out["Field"] = raw[field_col].astype(str).str.strip()
    if desc_col:
        out["Description"] = raw[desc_col].astype(str).str.strip()
    else:
        out["Description"] = ""
    if type_col:
        out["Datatype"] = raw[type_col].astype(str).str.strip()
    else:
        out["Datatype"] = ""
    return out


def _infer_category(col: str) -> str:
    """Roughly classify fields into logical groups for the dictionary."""
    name = col.upper()

    # Raw farm ops
    if any(k in name for k in ["FARM_ID", "BARN", "RAISE_DAY", "RAISE_WEEK",
                               "BEGIN_POP", "ANIMAL_IN", "DEATH", "CULL"]):
        return "Raw Farm Operations"

    # Growth
    if "ADG" in name or "WEIGHT" in name or "ESTIMATED_WEIGHT" in name:
        return "Growth & Performance"

    # Feed
    if "FEED" in name:
        return "Feed & Efficiency"

    # Climate / temp / heat
    if "TEMP" in name or "HEAT" in name:
        return "Climate & Heat Stress"

    # Health / respiratory / reason
    if "RESP" in name or "HEALTH" in name or "REASON" in name:
        return "Health & Risk"

    # Barn performance scores / unified metrics
    if any(k in name for k in ["UNIFIED_SCORE", "SAMPLE_STRENGTH", "AGE_COVERAGE",
                               "HEALTH_STABILITY", "SCORE"]):
        return "Barn Performance KPIs"

    # Segmentation
    if "CLUSTER" in name or "SEGMENT" in name:
        return "Segmentation & K-Means"

    # Encoded / model features
    if "ENC" in name or "TARGET_ENC" in name or name.startswith("FEAT_"):
        return "ML Feature Engineering"

    return "Other / General"


def _infer_table(col: str, df: pd.DataFrame, barn_perf: pd.DataFrame | None) -> str:
    in_df = col in df.columns
    in_barn = barn_perf is not None and col in barn_perf.columns

    if in_df and in_barn:
        return "Main Dataset (df) + Barn Performance (barn_perf)"
    if in_df:
        return "Main Dataset (df)"
    if in_barn:
        return "Barn Performance (barn_perf)"
    return "Other / Derived"


def _auto_description(col: str) -> str:
    """Fallback description if not supplied in Excel."""
    name = col.upper()
    if name == "ADG":
        return "Average Daily Gain (weight / days raised)"
    if "RESPIRATORY_PERCENT" in name:
        return "Percentage of animals showing respiratory symptoms"
    if "RESPIRATORY_PERCENT_FLAG" in name:
        return "Indicator flag when respiratory percentage exceeds threshold"
    if "HEAT_STRESS" in name:
        return "Heat stress indicator based on temperature difference"
    if "TEMP_DIFF" in name:
        return "Difference between max and min temperature within a period"
    if "UNIFIED_SCORE" in name:
        return "Composite barn performance score combining growth, health and sample strength"
    if "MEDIAN_ADG" in name:
        return "Median Average Daily Gain per barn"
    if "FEED_EFF" in name:
        return "Feed efficiency metric (e.g. weight gained per unit of feed)"
    if "BARN_CLEAN" in name:
        return "Normalised barn identifier used for clustering and reporting"
    if "CLUSTER_LABEL" in name:
        return "Human-readable label assigned to K-Means cluster"
    return "Description TBD (derived in dashboard)"


def build_enhanced_data_dictionary(
    df: pd.DataFrame,
    barn_perf: pd.DataFrame | None,
    dict_path: str = "Data_Dictionary (Sharing).xlsx",
) -> pd.DataFrame:
    """
    Merge:
      - External Excel data dictionary (if present)
      - Columns from df
      - Columns from barn_perf
    into one enhanced dictionary.
    """
    external = _load_excel_dictionary(dict_path)

    # Base: all unique fields from df + barn_perf
    cols = set(df.columns)
    if barn_perf is not None:
        cols.update(barn_perf.columns)
    all_fields = sorted(cols)

    # External lookup for descriptions
    external_desc = {}
    external_dtype = {}
    if external is not None:
        for _, row in external.iterrows():
            key = str(row["Field"]).strip().upper()
            external_desc[key] = row.get("Description", "")
            external_dtype[key] = row.get("Datatype", "")

    rows = []
    for col in all_fields:
        key = col.upper()

        desc = external_desc.get(key, "").strip()
        dtype = external_dtype.get(key, "").strip()

        # If no datatype from external, infer from df / barn_perf
        if not dtype:
            if col in df.columns:
                dtype = str(df[col].dtype)
            elif barn_perf is not None and col in barn_perf.columns:
                dtype = str(barn_perf[col].dtype)
            else:
                dtype = "-"

        # If no description at all, auto-generate a helpful one
        if not desc:
            desc = _auto_description(col)

        category = _infer_category(col)
        table = _infer_table(col, df, barn_perf if barn_perf is not None else pd.DataFrame())

        rows.append(
            {
                "Field": col,
                "Description": desc,
                "Datatype": dtype,
                "Category": category,
                "Table / Source": table,
            }
        )

    dd = pd.DataFrame(rows)
    return dd.sort_values(["Category", "Field"])


# summary the XGBoost modeling journey
import pandas as pd

# ------------------------------------------------------------
# MODEL DEVELOPMENT JOURNEY TABLE (R16 → R22)
# ------------------------------------------------------------
def render_model_dev_journey_table():
    data = [
        {
            "Stage": "1. Baseline Start(R1)",
            "Model Trained": "Linear Regression",
            "Problem Encountered": "Very low accuracy",
            "Root Cause (Diagnosis)": "Weight–feed relationship is non-linear",
            "Fix Applied": "Move to tree-based models",
            "Outcome / R²": "R² ≈ 0.12",
        },
        {
            "Stage": "2. First Tree Models (R2)",
            "Model Trained": "Random Forest, XGBoost",
            "Problem Encountered": "Flat / linear sensitivity",
            "Root Cause (Diagnosis)": "Missing FEED×AGE interaction",
            "Fix Applied": "Added FEED×AGE; resampled data",
            "Outcome / R²": "R² ≈ 0.61–0.65",
        },
        {
            "Stage": "3. CRITICAL – Leakage Incident (R3)",
            "Model Trained": "RF, XGB",
            "Problem Encountered": "R² suddenly jumped to ~0.90 (too good to be true)",
            "Root Cause (Diagnosis)": (
                "Target leakage: ESTIMATED_WEIGHT + correlated fields leaked into training; "
                "split done after scaling/imputing"
            ),
            "Fix Applied": (
                "Rebuilt pipeline: cleaned feature list, removed weight-derived fields, "
                "ensured train-test split BEFORE scaling, dropped leaky features"
            ),
            "Outcome / R²": "R² corrected to realistic values",
        },
        {
            "Stage": "4. Cleaned Pipeline Rebuild (R3)",
            "Model Trained": "LR, RF, XGB",
            "Problem Encountered": "Linear Regression failing due to NaN",
            "Root Cause (Diagnosis)": "Missing values after merge",
            "Fix Applied": "Added Imputer() in sklearn pipeline",
            "Outcome / R²": "LR stable; RF/XGB consistent",
        },
        {
            "Stage": "5. Feature Engineering Upgrade (R4)",
            "Model Trained": "RF, XGB",
            "Problem Encountered": "Curve still too straight / underfitting",
            "Root Cause (Diagnosis)": "Training samples lacked young-age ranges",
            "Fix Applied": "Added FEED×AGE features and rebalanced dataset",
            "Outcome / R²": "R² improved to ~0.71",
        },
        {
            "Stage": "6. SHAP Integration Restore (R4 → R5)",
            "Model Trained": "XGB with SHAP",
            "Problem Encountered": "“Invalid load key” error when loading model",
            "Root Cause (Diagnosis)": "App expected .pkl file but models were trained in-memory",
            "Fix Applied": "Rebuilt SHAP using in-memory models only",
            "Outcome / R²": "SHAP restored and stable",
        },
        {
            "Stage": "7. Final Pipeline Align (R6)",
            "Model Trained": "XGBoost (R6 → R7), RF, LR",
            "Problem Encountered": "Discrepancy between notebook vs Streamlit R²",
            "Root Cause (Diagnosis)": "App was using old feature list / different pipeline version",
            "Fix Applied": "Migrated R20 → R21 with unified feature set",
            "Outcome / R²": "XGB R² ≈ 0.781; RF R² ≈ 0.728; LR R² ≈ 0.169",
        },
        {
            "Stage": "8. Production Optimization (R8)",
            "Model Trained": "XGBoost (Final, deployed)",
            "Problem Encountered": "Mid-weight range slightly under-predicted",
            "Root Cause (Diagnosis)": "Trees not deep enough + suboptimal learning rate",
            "Fix Applied": (
                "Hyperparameter tuning (Bayesian / grid): max_depth=7, "
                "learning_rate=0.05, n_estimators↑, tuned subsample/colsample; "
                "added FEED_EFF×TEMP_DIFF interaction"
            ),
            "Outcome / R²": "🚀 Final XGBoost R² ≈ **0.821**",
        },
    ]

    df_journey = pd.DataFrame(data)

    st.dataframe(
        df_journey,
        width="stretch",  # modern replacement for use_container_width
    )



# ================================================================
# TEXT / REASON UTILITIES
# ================================================================
RESP_KEYWORDS = [
    "hô hấp", "ho hap", "khó thở", "kho tho", "thở", "so mui", "sổ mũi",
    "ho", "viêm phổi", "viem phoi", "hen", "respiratory"
]

# ================================================================
# Management Target for EXECUTIVE SUMMARY KPI Card
# ================================================================
TARGETS = {
    "UNDERPERFORMING_BARNS": 20.0,   # %
    "ADG": 1.5,                     # kg/day
    "WEIGHT": 110.0,                  # kg
    "RESPIRATORY": 10.0,             # %
    "HEAT_STRESS": 10.0,             # %
    "TOP_UNIFIED":2.5               # target unified score
}

# ------------------------------------------------
# 🔧 Fix inconsistent barn names (prevents Unassigned clusters)
# ------------------------------------------------
def clean_barn_code(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip().upper()
    x = re.sub(r"\s+", "", x)   # remove spaces: "C 1" → "C1"
    x = re.sub(r"[^A-Z0-9]", "", x)  # remove invalid chars
    return x


# ------------------------------------------------
# 🔍 Categorize REASON text
# ------------------------------------------------
RESP_KEYWORDS = [
    "hô hấp", "ho hap", "khó thở", "kho tho", "thở", "so mui", "sổ mũi",
    "ho", "viêm phổi", "viem phoi", "hen", "respiratory"
]

def categorize_reason(text: str) -> str:
    if pd.isna(text):
        return "unknown"

    t = str(text).lower()

    if "hô hấp" in t or "ho hap" in t:
        return "respiratory_issue"
    if "thời tiết" in t or "thoi tiet" in t or "nắng" in t or "lạnh" in t:
        return "weather_change"
    if "cám" in t or "cam" in t or "ăn hết" in t:
        return "feed_issue"
    if "lmlm" in t or "prrs" in t:
        return "disease_outbreak"
    if "quy trình" in t or "quy trinh" in t:
        return "procedure_issue"

    return "other"


# ------------------------------------------------
# 🎚 Convert REASON category → severity score
# ------------------------------------------------
def reason_severity_from_category(cat: str) -> float:
    mapping = {
        "respiratory_issue": 0.20,
        "disease_outbreak": 0.25,
        "weather_change": 0.10,
        "feed_issue": 0.05,
    }
    return mapping.get(cat, 0.0)


# ------------------------------------------------
# 📊 Extract numeric respiratory % from text
# ------------------------------------------------
def extract_respiratory_percent(text):
    if pd.isna(text):
        return np.nan

    text = unicodedata.normalize("NFC", str(text)).lower()

    # Fix "2, 83" → "2.83"
    text = re.sub(r"(\d)\s*,\s*(\d)", r"\1.\2", text)

    # % values
    m1 = re.search(r"(\d+\.?\d*)\s*%", text)
    if m1:
        return float(m1.group(1))

    # Ranges like "10–15%"
    m2 = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*%", text)
    if m2:
        low, high = map(float, m2.groups())
        return (low + high) / 2

    return np.nan


# ------------------------------------------------
# 🫁 Detect presence of respiratory keywords
# ------------------------------------------------
def detect_respiratory_keywords(text):
    if pd.isna(text):
        return 0
    t = unicodedata.normalize("NFC", str(text)).lower()
    return int(any(k in t for k in RESP_KEYWORDS))


# ------------------------------------------------
# 🌟 Assign meaningful names to clusters
# ------------------------------------------------
def assign_cluster_names(df, cluster_col, sort_metric):
    """
    Rank clusters by performance and assign names:
    High Performance → Moderate → Low → etc.
    """
    ranked = (
        df.groupby(cluster_col)[sort_metric]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    name_list = [
        "High Performance",
        "Moderate Performance",
        "Low Performance",
        "Developing Group",
        "Underperforming",
        "Critical Risk",
    ]

    name_map = {
        ranked.loc[i, cluster_col]: name_list[i]
        for i in range(len(ranked))
    }

    df[cluster_col + "_Label"] = df[cluster_col].map(name_map)

    return df, name_map


# ================================================================
# DATA LOADING + CLEANING + FEATURE ENGINEERING
# ================================================================
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv("farm_data_prepared_v3.csv")
    except FileNotFoundError:
        df = pd.read_csv("farm_data_prepared_v2.csv")

    # ---- CLEAN BARN CODES (CRITICAL FOR CLUSTERING & SANKAY) ----
    df["BARN_CLEAN"] = df["BARN"].astype(str).apply(clean_barn_code)

    # Core numeric
    num_cols = [
        "ESTIMATED_WEIGHT",
        "FEED_INTAKE_ACTUAL",
        "RESPIRATORY_PERCENT",
        "RAISE_DAY",
        "MIN_INDOOR_TEMPERATURE",
        "MAX_INDOOR_TEMPERATURE",
        "MIN_OUTDOOR_TEMPERATURE",
        "MAX_OUTDOOR_TEMPERATURE",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix indoor temperature + HEAT_STRESS (pig-specific threshold)
    if "MAX_INDOOR_TEMPERATURE" in df.columns:
        df["MAX_INDOOR_TEMPERATURE"] = pd.to_numeric(
            df["MAX_INDOOR_TEMPERATURE"], errors="coerce"
        )
        indoor_median = df["MAX_INDOOR_TEMPERATURE"].median()
        df["MAX_INDOOR_TEMPERATURE"] = df["MAX_INDOOR_TEMPERATURE"].fillna(indoor_median)
    else:
        df["MAX_INDOOR_TEMPERATURE"] = 28

    df["HEAT_STRESS"] = (df["MAX_INDOOR_TEMPERATURE"] >= 30).astype(int)

    # FEED_X_AGE
    df["FEED_X_AGE"] = df["FEED_INTAKE_ACTUAL"] * df["RAISE_DAY"]

    # REASON_CATEGORY
    if "REASON" in df.columns:
        df["REASON_CATEGORY"] = df["REASON"].apply(categorize_reason)
    else:
        df["REASON_CATEGORY"] = "unknown"

    # TEMP_DIFF
    if (
        "MAX_OUTDOOR_TEMPERATURE" in df.columns
        and "MIN_OUTDOOR_TEMPERATURE" in df.columns
    ):
        df["TEMP_DIFF"] = df["MAX_OUTDOOR_TEMPERATURE"] - df["MIN_OUTDOOR_TEMPERATURE"]
    else:
        df["TEMP_DIFF"] = np.nan

    # REASON_SCORE
    df["REASON_SCORE"] = df["REASON_CATEGORY"].apply(reason_severity_from_category)

    # Respiratory %
    if "REASON" in df.columns:
        df["RESPIRATORY_PERCENT"] = df["REASON"].apply(extract_respiratory_percent)
    df["RESPIRATORY_PERCENT"] = df["RESPIRATORY_PERCENT"].fillna(0.0)

    df["RESPIRATORY_PERCENT_FLAG"] = (
        (df["RESPIRATORY_PERCENT"] > 0)
        | (df["REASON"].apply(detect_respiratory_keywords) == 1)
    ).astype(int)

    df["RESP_SEVERITY"] = (df["RESPIRATORY_PERCENT"] / 100).clip(0, 1)

    # ADG
    df["ADG"] = (
        df["ESTIMATED_WEIGHT"] / df["RAISE_DAY"].replace(0, np.nan)
    )
    df["ADG"] = pd.to_numeric(df["ADG"], errors="coerce")

    # BARN_TARGET_ENC
    barn_means = df.groupby("BARN_CLEAN")["ESTIMATED_WEIGHT"].mean()
    df["BARN_TARGET_ENC"] = df["BARN_CLEAN"].map(barn_means).fillna(
        df["ESTIMATED_WEIGHT"].mean()
    )

    return df


df = load_and_prepare_data()


# Defaults for live prediction
DEFAULT_TEMP_DIFF = float(df["TEMP_DIFF"].median())
DEFAULT_BARN_TE = float(df["BARN_TARGET_ENC"].median())
DEFAULT_REASON_SCORE = float(df["REASON_SCORE"].median())

# ================================================================
# BARN NORMALIZATION (must be applied BEFORE build_barn_performance)
# ================================================================
def normalize_barn(barn):
    if pd.isna(barn):
        return "UNKNOWN"
    barn = str(barn).upper().strip()
    barn = re.sub(r"\s+", "", barn)     # Remove ALL spaces: C1, C 1, C   1 → C1
    return barn

df = load_and_prepare_data()

# GLOBAL barn normalization
df["BARN_CLEAN"] = df["BARN"].apply(normalize_barn)



# ================================================================
# BARN PERFORMANCE (Unified Score for Exec + Ops)
# ================================================================
def build_barn_performance(df_in: pd.DataFrame) -> pd.DataFrame:
    required = ["BARN", "ESTIMATED_WEIGHT", "RAISE_DAY"]
    for col in required:
        if col not in df_in.columns:
            return pd.DataFrame()

    df_b = df_in.dropna(subset=required).copy()

    # Normalize barn code early
    df_b["BARN_CLEAN"] = df_b["BARN"].apply(normalize_barn)

    # -------------------------
    # Core performance metrics
    # -------------------------
    df_b["ADG"] = df_b["ESTIMATED_WEIGHT"] / df_b["RAISE_DAY"].replace(0, np.nan)
    df_b["FEED_EFF"] = df_b["ADG"] / df_b["FEED_INTAKE_ACTUAL"].replace(0, np.nan)

    df_b["RESP_FLAG"] = df_b["RESPIRATORY_PERCENT"].fillna(0) > 0
    df_b["HEAT_FLAG"] = df_b["HEAT_STRESS"] == 1

    # -------------------------
    # Aggregate at barn level
    # -------------------------
    barn = (
        df_b.groupby("BARN_CLEAN")
        .agg(
            AVG_WEIGHT=("ESTIMATED_WEIGHT", "mean"),
            MEDIAN_ADG=("ADG", "median"),
            FEED_EFF=("FEED_EFF", "mean"),
            RESP_RATE=("RESP_FLAG", "mean"),
            HEAT_RATE=("HEAT_FLAG", "mean"),
            WEIGHT_STD=("ESTIMATED_WEIGHT", "std"),
            N_RECORDS=("ESTIMATED_WEIGHT", "count"),
            N_AGES=("RAISE_DAY", "nunique"),
        )
        .reset_index()
    )

    # -------------------------
    # Growth Score
    # -------------------------
    adg_min, adg_max = barn["MEDIAN_ADG"].min(), barn["MEDIAN_ADG"].max()
    barn["GROWTH_SCORE"] = (
        (barn["MEDIAN_ADG"] - adg_min) / (adg_max - adg_min)
        if adg_max != adg_min else 0.5
    )

    # Health stability score (LOW resp_rate = GOOD)
    barn["HEALTH_STABILITY_SCORE"] = (1 - barn["RESP_RATE"]).clip(0, 1)

    # -------------------------
    # Sample Strength Score
    # -------------------------
    barn["AGE_COVERAGE_SCORE"] = barn["N_AGES"].clip(upper=6) / 6
    barn["RECORD_COVERAGE_SCORE"] = barn["N_RECORDS"].clip(upper=30) / 30
    barn["SAMPLE_STRENGTH_SCORE"] = (
        0.6 * barn["AGE_COVERAGE_SCORE"] + 0.4 * barn["RECORD_COVERAGE_SCORE"]
    )

    # ============================================================
    # 🔧 FIX ALL NaN VALUES BEFORE CALCULATING UNIFIED SCORE
    # ============================================================
    barn["GROWTH_SCORE"] = barn["GROWTH_SCORE"].fillna(0)
    barn["HEALTH_STABILITY_SCORE"] = barn["HEALTH_STABILITY_SCORE"].fillna(0)
    barn["SAMPLE_STRENGTH_SCORE"] = barn["SAMPLE_STRENGTH_SCORE"].fillna(0)

    # -------------------------
    # Health / Stability Score
    # -------------------------
    barn["HEALTH_RAW"] = (1 - barn["RESP_RATE"]).clip(0, 1)
    barn["HEALTH_STABILITY_SCORE"] = (
        0.5 * barn["HEALTH_RAW"] + 0.5 * barn["HEALTH_STABILITY_SCORE"]
    )

    # -------------------------
    # Unified Score
    # -------------------------
    barn["UNIFIED_SCORE"] = (
        0.50 * barn["GROWTH_SCORE"]
        + 0.30 * barn["SAMPLE_STRENGTH_SCORE"]
        + 0.20 * barn["HEALTH_STABILITY_SCORE"]
    )

    # -------------------------
    # Additional fields used elsewhere
    # -------------------------
    barn["RESP_RATE_PCT"] = barn["RESP_RATE"] * 100.0

    return barn

# Ensure df uses BARN_CLEAN for cluster mapping
df["BARN_CLEAN"] = df["BARN"].apply(normalize_barn)

barn_perf = build_barn_performance(df)

# Ensure barn_perf also has BARN_CLEAN (safety)
if "BARN_CLEAN" not in barn_perf.columns:
    barn_perf["BARN_CLEAN"] = barn_perf["BARN"].apply(normalize_barn)


# ================================================================
# GLOBAL BARN K-MEANS (Stable Labels)
# ================================================================
if not barn_perf.empty:

    features = ["MEDIAN_ADG", "FEED_EFF", "RESP_RATE", "WEIGHT_STD", "N_AGES", "N_RECORDS"]
    df_b = barn_perf.dropna(subset=features).copy()

    scaler = StandardScaler()
    Xb = scaler.fit_transform(df_b[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_b["Cluster"] = kmeans.fit_predict(Xb)

    df_b, name_map = assign_cluster_names(df_b, "Cluster", sort_metric="MEDIAN_ADG")

    # Must merge using BARN_CLEAN (not BARN)
    barn_perf = barn_perf.merge(
        df_b[["BARN_CLEAN", "Cluster_Label"]],
        on="BARN_CLEAN",
        how="left"
    )


# ================================================================
# LOESS GROWTH CURVE
# ================================================================
def build_loess_growth(df_in: pd.DataFrame):
    tmp = df_in.dropna(subset=["ESTIMATED_WEIGHT", "RAISE_DAY"]).copy()
    grp = tmp.groupby("RAISE_DAY", as_index=False)["ESTIMATED_WEIGHT"].mean()
    grp.rename(columns={"ESTIMATED_WEIGHT": "MEAN_WEIGHT"}, inplace=True)

    loess_res = lowess(grp["MEAN_WEIGHT"], grp["RAISE_DAY"], frac=0.3, return_sorted=True)
    loess_df = pd.DataFrame(loess_res, columns=["RAISE_DAY", "WEIGHT_SMOOTH"])

    return grp.merge(loess_df, on="RAISE_DAY", how="left")


loess_df = build_loess_growth(df)

# ================================================================
# MODEL TRAINING – XGBOOST ONLY
# ================================================================
TARGET = "ESTIMATED_WEIGHT"

FEATURES = [
    "RAISE_DAY",
    "FEED_INTAKE_ACTUAL",
    "RESPIRATORY_PERCENT",
    "RESPIRATORY_PERCENT_FLAG",
    "FEED_X_AGE",
    "BARN_TARGET_ENC",
    "TEMP_DIFF",
    "HEAT_STRESS",
    "RESP_SEVERITY",
    "REASON_SCORE",
]


@st.cache_data
def make_train_test(df_in: pd.DataFrame):
    df_model = df_in.dropna(subset=[TARGET, "RAISE_DAY", "FEED_INTAKE_ACTUAL"]).copy()
    X = df_model[FEATURES]
    y = df_model[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


X_train_raw, X_test_raw, y_train, y_test = make_train_test(df)


def build_pipeline(estimator):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ]
    )


@st.cache_resource
def train_xgb_model(X_train, y_train):
    xgb = build_pipeline(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    )
    xgb.fit(X_train, y_train)
    return xgb


xgb_model = train_xgb_model(X_train_raw, y_train)


def evaluate_model(model, X_eval, y_eval):
    pred = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, pred)
    rmse = np.sqrt(mean_squared_error(y_eval, pred))
    r2 = r2_score(y_eval, pred)
    return mae, rmse, r2, pred


mae_x, rmse_x, r2_x, preds_x = evaluate_model(xgb_model, X_test_raw, y_test)


def predict_weight(feed, age, resp_percent=0.0, resp_flag=0.0, model=None):
    if model is None:
        model = xgb_model

    X = pd.DataFrame(
        {
            "RAISE_DAY": [age],
            "FEED_INTAKE_ACTUAL": [feed],
            "RESPIRATORY_PERCENT": [resp_percent],
            "RESPIRATORY_PERCENT_FLAG": [resp_flag],
            "FEED_X_AGE": [feed * age],
            "BARN_TARGET_ENC": [DEFAULT_BARN_TE],
            "TEMP_DIFF": [DEFAULT_TEMP_DIFF],
            "HEAT_STRESS": [0],
            "RESP_SEVERITY": [resp_percent / 100.0],
            "REASON_SCORE": [DEFAULT_REASON_SCORE],
        }
    )
    return float(model.predict(X)[0])
# ================================================================
# PAGE 1 — EXECUTIVE SUMMARY (KPIs)
# ================================================================
if page == "📌 Executive Summary":
    st.markdown("## 📌 Executive Summary – Farm Performance & Health")
    

    # ---------- Theme-aware colors ----------
    is_dark = st.session_state.plotly_template == "plotly_dark"
    CARD_BG = "#0f172a" if is_dark else "white"
    CARD_TEXT = "white" if is_dark else "#0f172a"
    CARD_SUBTEXT = "#cbd5e1" if is_dark else "#475569"
    CARD_BORDER = "#1e293b" if is_dark else "#e2e8f0"

    # ---------- Core aggregates from raw df ----------
    df_valid = df.dropna(subset=["ESTIMATED_WEIGHT"])
    avg_weight_raw = df_valid["ESTIMATED_WEIGHT"].mean()
    avg_adg_raw = df_valid["ADG"].mean() if "ADG" in df_valid.columns else 0
    respiratory_pct_raw = (
        df["RESPIRATORY_PERCENT"].mean() if "RESPIRATORY_PERCENT" in df.columns else 0
    )
    heatstress_pct_raw = (
        df["HEAT_STRESS"].mean() * 100 if "HEAT_STRESS" in df.columns else 0
    )

    # ---------- Barn-level metrics ----------
    top_unified_value = 0.0
    underperforming_pct = 0.0
    avg_adg = avg_adg_raw
    avg_weight = avg_weight_raw
    respiratory_pct = respiratory_pct_raw
    heatstress_pct = heatstress_pct_raw
    best_adg_barn = worst_adg_barn = None

    if barn_perf is not None and not barn_perf.empty:
        # Sort by unified score
        barn_sorted_unified = barn_perf.sort_values(
            "UNIFIED_SCORE", ascending=False
        )
        top_unified_barn = barn_sorted_unified.iloc[0]
        bottom_unified_barn = barn_sorted_unified.iloc[-1]

        top_unified_value = float(top_unified_barn["UNIFIED_SCORE"])

        # Underperforming barns = below TOP_UNIFIED target
        if "UNIFIED_SCORE" in barn_perf.columns:
            under_mask = barn_perf["UNIFIED_SCORE"] < TARGETS["TOP_UNIFIED"]
            underperforming_pct = under_mask.mean() * 100

        # Use barn-level MEDIAN_ADG and AVG_WEIGHT if available
        if "MEDIAN_ADG" in barn_perf.columns:
            avg_adg = barn_perf["MEDIAN_ADG"].mean()
        if "AVG_WEIGHT" in barn_perf.columns:
            avg_weight = barn_perf["AVG_WEIGHT"].mean()

        # Respiratory rate at barn level if available
        if "RESP_RATE_PCT" in barn_perf.columns:
            respiratory_pct = barn_perf["RESP_RATE_PCT"].mean()
        else:
            respiratory_pct = respiratory_pct_raw

        # Heat stress at barn level if available
        if "HEAT_RATE" in barn_perf.columns:
            heatstress_pct = barn_perf["HEAT_RATE"].mean() * 100
        else:
            heatstress_pct = heatstress_pct_raw

        # Best / worst ADG barns
        if "MEDIAN_ADG" in barn_perf.columns:
            best_adg_barn = barn_perf.loc[barn_perf["MEDIAN_ADG"].idxmax()]
            worst_adg_barn = barn_perf.loc[barn_perf["MEDIAN_ADG"].idxmin()]

    # ================================================================
    #  KPI GAUGE FUNCTION 
    # ================================================================
    
    import plotly.graph_objects as go

    def kpi_gauge(title, value, target, kpi_type="higher_better"):
        """Compact gauge, bright colors, visible titles."""
        
        max_range = max(value, target) * 1.25
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={"text": f"<b style='font-size:1.0rem;'>{title}</b>", "font": {"size": 12}},
            delta={
                "reference": target,
                "increasing": {"color": GAUGE_RED},
                "decreasing": {"color": GAUGE_GREEN}
            },
            gauge={
                "axis": {"range": [0, max_range], "tickfont": {"size": 10}},
                "bar": {"color": "#3b82f6", "thickness": 0.25},
                "steps": [
                    {"range": [0, target*0.8], "color": GAUGE_GREEN},
                    {"range": [target*0.8, target], "color": GAUGE_YELLOW},
                    {"range": [target, max_range], "color": GAUGE_RED},
                ],
                "threshold": {
                    "line": {"color": "#ff0000", "width": 4},
                    "value": target,
                }
            }
        ))

        fig.update_layout(
            height=220,
            margin=dict(l=5, r=5, t=30, b=0),
        )

        st.plotly_chart(fig, width='stretch')

    def adg_card(title, barn_label, adg_value):
        return f"""
        <div style="
            padding: 0.8rem; border-radius: 12px;
            background-color: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            width: 100%;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.9rem; color:{CARD_SUBTEXT};">{title}</div>
            <div style="font-size: 1.6rem; font-weight: 700; color:{CARD_TEXT};">
                {barn_label}
            </div>
            <div style="font-size: 1.1rem; color:{CARD_TEXT};">
                ADG: {adg_value:.3f} kg/day
            </div>
        </div>
        """

   
    # ============================================================
    # BEST & WORST ADG (SAFE DEFINITIONS — ALWAYS AVAILABLE)
    # ============================================================

    if barn_perf is not None and not barn_perf.empty and "MEDIAN_ADG" in barn_perf.columns:

        # Best ADG barn
        best_adg_row = barn_perf.iloc[barn_perf["MEDIAN_ADG"].idxmax()]
        best_adg_barn_label = best_adg_row["BARN_CLEAN"]
        best_adg_value = best_adg_row["MEDIAN_ADG"]

        # Worst ADG barn
        worst_adg_row = barn_perf.iloc[barn_perf["MEDIAN_ADG"].idxmin()]
        worst_adg_barn_label = worst_adg_row["BARN_CLEAN"]
        worst_adg_value = worst_adg_row["MEDIAN_ADG"]

    else:
        # Fallback if barn_perf missing or empty
        best_adg_barn_label = "N/A"
        best_adg_value = 0.0
        worst_adg_barn_label = "N/A"
        worst_adg_value = 0.0

    # ================================================================
    #  KPI LAYOUT — 6 GAUGES + SIDE CARDS
    #   1) Underperforming Barns (%)
    #   2) Average ADG
    #   3) Average Weight
    #   4) Respiratory Exposure
    #   5) Heat Stress Exposure
    #   6) Top Barn Unified Score
    # ================================================================
    st.markdown("## 📊 KPI Performance Dashboard")

    col_g1, col_g2, col_g3, col_cards = st.columns([1.1,1.1,1.1,0.9])

    with col_g1:
        kpi_gauge("Underperforming Barns (%)", underperforming_pct, TARGETS["UNDERPERFORMING_BARNS"], "lower_better")
        kpi_gauge("Heat Stress Exposure (%)", heatstress_pct, TARGETS["HEAT_STRESS"], "lower_better")

    with col_g2:
        kpi_gauge("Average Daily Gain (kg/day)", avg_adg, TARGETS["ADG"], "higher_better")
        kpi_gauge("Respiratory Exposure (%)", respiratory_pct, TARGETS["RESPIRATORY"], "lower_better")

    with col_g3:
        kpi_gauge("Average Weight (kg)", avg_weight, TARGETS["WEIGHT"], "higher_better")
        kpi_gauge("Top Barn Unified Score", top_unified_barn["UNIFIED_SCORE"], TARGETS["TOP_UNIFIED"], "higher_better")

    with col_cards:
        st.markdown(adg_card("🏆 Best ADG Barn", best_adg_barn_label, best_adg_value), unsafe_allow_html=True)
        st.markdown(adg_card("⚠️ Worst ADG Barn", worst_adg_barn_label, worst_adg_value), unsafe_allow_html=True)


    # ================================================================
    # EXPANDER 1 — Unified Barn Score Chart
    # ================================================================
    with st.expander("🏆 Barn Performance – Unified Score", expanded=False):
        if barn_perf is not None and not barn_perf.empty:
            fig_barn = go.Figure()
            fig_barn.add_trace(
                go.Bar(
                    x=barn_perf["BARN_CLEAN"],
                    y=barn_perf["UNIFIED_SCORE"],
                    marker=dict(color="#22c55e"),
                )
            )
            fig_barn.update_layout(
                template=plotly_template,
                height=420,
                xaxis_title="Barn",
                yaxis_title="Unified Score (0–1)",
            )
            st.plotly_chart(fig_barn, width="stretch")
        else:
            st.info("Barn performance data not available.")

    # ================================================================
    # EXPANDER 2 — Health & Issue Overview
    # ================================================================
    with st.expander("🩺 Health & Issue Overview (Drill-down)", expanded=False):
        if "REASON_CATEGORY" in df.columns:
            cat_counts = df["REASON_CATEGORY"].value_counts().reset_index()
            cat_counts.columns = ["Reason Category", "Count"]

            fig_reason = px.bar(
                cat_counts,
                x="Reason Category",
                y="Count",
                color="Reason Category",
                title="Distribution of Reported Issues",
            )
            fig_reason.update_layout(template=plotly_template, height=380, showlegend=False)
            st.plotly_chart(fig_reason, width="stretch")

            focus_reason = st.selectbox(
                "Drill down into issue:",
                options=["All"] + cat_counts["Reason Category"].tolist(),
            )

            if focus_reason != "All":
                df_focus = df[df["REASON_CATEGORY"] == focus_reason]

                barn_issue = (
                    df_focus.groupby("BARN_CLEAN")["RAISE_DAY"]
                    .count()
                    .reset_index()
                    .rename(columns={"RAISE_DAY": "Issue_Count"})
                )

                if not barn_issue.empty:
                    barn_issue.sort_values("Issue_Count", ascending=False, inplace=True)

                    fig_issue = px.bar(
                        barn_issue,
                        x="BARN_CLEAN",
                        y="Issue_Count",
                        title=f"{focus_reason} Cases by Barn",
                        color="Issue_Count",
                        color_continuous_scale="Reds",
                    )
                    fig_issue.update_layout(template=plotly_template, height=380)
                    st.plotly_chart(fig_issue, width="stretch")
                else:
                    st.info("No records found for this issue type.")
        else:
            st.info("No REASON_CATEGORY field in dataset.")

    # ================================================================
    # EXPANDER 3 — Strategic Takeaways
    # ================================================================
    with st.expander("📌 Strategic Takeaways (Executive View)", expanded=False):
        best_unified_label = top_unified_barn["BARN_CLEAN"] if barn_perf is not None and not barn_perf.empty else "N/A"
        worst_unified_label = bottom_unified_barn["BARN_CLEAN"] if barn_perf is not None and not barn_perf.empty else "N/A"

        best_adg_label = best_adg_barn["BARN_CLEAN"] if best_adg_barn is not None else "N/A"
        worst_adg_label = worst_adg_barn["BARN_CLEAN"] if worst_adg_barn is not None else "N/A"
        
        st.markdown(f"""
        ### 🐖 Key Insights

        - 🏆 **Best Overall Barn:** **{best_unified_label}** with the highest unified score of **{top_unified_value:.2f}**.  
        - 📉 **Underperforming Barns:** **{underperforming_pct:.1f}%** fall below the target threshold (**{TARGETS["TOP_UNIFIED"]}**).  
        - 🚀 **Best Growth Barn (ADG):** **{best_adg_label}**, showing superior feed conversion efficiency.  
        - 🐢 **Lowest Growth Barn (ADG):** **{worst_adg_label}**, requiring evaluation of feed strategy, stocking density, or environmental controls.  
        - 😷 **Respiratory Exposure:** Average exposure is **{respiratory_pct:.1f}%**, indicating health risks that may slow weight gain.  
        - 🌡️ **Heat Stress Exposure:** **{heatstress_pct:.1f}%**, suggesting climate-driven performance limitations.

        ### 📌 Recommended Next Steps
        - Prioritise operational checks for **low ADG barns** to identify feed, ventilation, or disease issues.  
        - Improve climate resilience for barns experiencing high **TEMP_DIFF** or **HEAT_STRESS** values.  
        - Benchmark barns against **top performer {best_unified_label}** to replicate best practices.
        """)

    # ================================================================
    # EXPANDER 4 — Developer Mode
    # ================================================================
    if dev_mode:
        with st.expander("🧪 Developer Mode – Barn Performance Table", expanded=False):
            st.dataframe(
                barn_perf.sort_values("UNIFIED_SCORE", ascending=False),
                width="stretch",
            )

   
# ================================================================
# PAGE 2 — DESCRIPTIVE & CORRELATION ANALYTICS
# ================================================================
elif page == "📊 Descriptive Analytics":
    st.markdown("## 📊 Descriptive & Correlation Analytics")
    

    st.markdown(
        """
        This view helps technical users understand **data structure and relationships**:
        - Correlation of weight with feed, age, climate and health  
        - Visual scatter distributions by day, barn, and issue types  
        """
    )

    st.markdown("---")

    # ================================================================
    # EXPANDER 1 — Correlation Heatmap
    # ================================================================
    with st.expander("📌 Correlation Heatmap – Core Features", expanded=False):

        corr_features = [
            "ESTIMATED_WEIGHT",
            "FEED_INTAKE_ACTUAL",
            "RAISE_DAY",
            "RESPIRATORY_PERCENT",
            "RESPIRATORY_PERCENT_FLAG",
            "FEED_X_AGE",
            "BARN_TARGET_ENC",
            "TEMP_DIFF",
            "HEAT_STRESS",
            "RESP_SEVERITY",
            "REASON_SCORE",
            "ADG",
        ]
        corr_features = [c for c in corr_features if c in df.columns]

        corr_df = df[corr_features].corr()

        fig_corr = px.imshow(
            corr_df,
            color_continuous_scale="viridis",
            origin="lower",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix – Weight, Feed, Age, Health & Environment",
        )
        fig_corr.update_layout(
            template=plotly_template,
            height=540,
            coloraxis_colorbar=dict(
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"],
            ),
        )

        # Add correlation values inside heatmap cells
        text_color = "white" if st.session_state.theme_choice == "Dark" else "black"
        for i in range(len(corr_df)):
            for j in range(len(corr_df)):
                fig_corr.add_annotation(
                    x=j,
                    y=i,
                    text=str(round(corr_df.iloc[i, j], 2)),
                    showarrow=False,
                    font=dict(color=text_color, size=11),
                )

        st.plotly_chart(fig_corr, width="stretch")

    # ================================================================
    # EXPANDER 2 — Scatterplots: Weight vs Age & Feed
    # ================================================================
    with st.expander("📈 Scatterplots – Weight vs Age & Feed Intake", expanded=False):

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Weight vs Age (Colored by Issue Category)")
            scat1 = px.scatter(
                df,
                x="RAISE_DAY",
                y="ESTIMATED_WEIGHT",
                color="REASON_CATEGORY" if "REASON_CATEGORY" in df.columns else None,
                title="Observed Weight vs Age",
                opacity=0.5,
            )
            scat1.update_layout(template=plotly_template, height=420)
            st.plotly_chart(scat1, width="stretch")

        with c2:
            st.subheader("Weight vs Feed Intake (Colored by Age)")
            scat2 = px.scatter(
                df,
                x="FEED_INTAKE_ACTUAL",
                y="ESTIMATED_WEIGHT",
                color="RAISE_DAY",
                title="Observed Weight vs Feed Intake",
                opacity=0.6,
            )
            scat2.update_layout(template=plotly_template, height=420)
            st.plotly_chart(scat2, width="stretch")

    # ================================================================
    # EXPANDER 3 — Summary Insights
    # ================================================================
    with st.expander("🧾 Summary Insights (Descriptive)", expanded=False):
        st.markdown(
            """
            - **RAISE_DAY** and **FEED_INTAKE_ACTUAL** are the strongest structural drivers of weight.  
            - **TEMP_DIFF** and **HEAT_STRESS** indicate **climate stress**; higher levels correlate with poorer performance.  
            - **RESP_SEVERITY** and **REASON_SCORE** capture health impacts, which reduce growth.  
            - Patterns show clear **non-linear growth behavior** and **feed sensitivity changes by age**.  
            """
        )

    # ================================================================
    # EXPANDER 4 — Developer Mode: Sample Data
    # ================================================================
    if dev_mode:
        with st.expander("🧪 Developer Mode – Data Preview", expanded=False):
            st.dataframe(df.head(30), width="stretch")



 # ================================================================
# PAGE 3 — PREDICTIVE ANALYTICS (With Expanders)
# ================================================================
elif page == "📈 Predictive Analytics":
    st.markdown("## 📈 Predictive Analytics – XGBoost Weight Forecasting")
    

    st.markdown(    
        f"""
        This page uses the **XGBoost model (R² ≈ {r2_x:.3f})** to:
        - Predict estimated weight based on feed, age, health and environment  
        - Explore feed sensitivity by age  
        - Visualise **Feed × Age × Weight** and **Feed Efficiency** heatmaps  
        """
    )

    # ================================================================
    # EXPANDER 1 — Live Predicted Weight (Gauge + Controls)
    # ================================================================
    with st.expander("🎛 Live Weight Prediction & Dual-Ring Gauge", expanded=False):

        c1, c2, c3 = st.columns(3)
        with c1:
            feed_in = st.slider("Daily Feed Intake (kg)", 1.0, 4.0, 2.2, 0.1)
        with c2:
            age_in = st.slider("Current Age (RAISE_DAY)", 27, 191, 150, 1)
        with c3:
            resp_in = st.slider("Respiratory % (current)", 0.0, 20.0, 3.0, 0.5)

        resp_flag_in = 1.0 if resp_in > 5.0 else 0.0
        pred_w = predict_weight(feed_in, age_in, resp_in, resp_flag_in, xgb_model)

        # ------------------------------------------------------------
        # Efficiency computation (SINGLE COMPUTATION)
        # ------------------------------------------------------------
        eff_live = pred_w / feed_in if feed_in > 0 else 0
        eff_min = 15
        eff_max = 50
        eff_ratio = (eff_live - eff_min) / (eff_max - eff_min)
        eff_ratio = max(0, min(eff_ratio, 1))  # clamp to 0–1
        eff_display = f"{eff_live:.2f} kg/kg feed"

        # Weight scale
        max_weight = 120
        weight_ratio = min(pred_w / max_weight, 1)

        # ------------------------------------------------------------
        # GAUGE FIGURE (Clean dual-ring donut)
        # ------------------------------------------------------------
        fig_gauge = go.Figure()

        # OUTER DONUT – Predicted Weight
        fig_gauge.add_trace(go.Pie(
            values=[weight_ratio, 1 - weight_ratio],
            hole=0.55,
            direction="clockwise",
            sort=False,
            marker=dict(colors=["#22c55e", "#1e293b"]),
            textinfo="none",
            showlegend=False,
            domain={"x": [0, 1], "y": [0, 1]}
        ))

        # INNER DONUT – Feed Efficiency
        fig_gauge.add_trace(go.Pie(
            values=[eff_ratio, 1 - eff_ratio],
            hole=0.80,
            direction="clockwise",
            sort=False,
            marker=dict(colors=["#facc15", "#0f172a"]),
            textinfo="none",
            showlegend=False,
            domain={"x": [0.13, 0.87], "y": [0.13, 0.87]}
        ))

        # TITLE
        fig_gauge.add_annotation(
            x=0.5, y=1.12,
            text="<b>Predicted Weight (kg) – XGBoost</b>",
            font=dict(size=22, color="white"),
            showarrow=False
        )

        # CENTER WEIGHT VALUE
        fig_gauge.add_annotation(
            x=0.5, y=0.52,
            text=f"<b>{pred_w:.1f} kg</b>",
            font=dict(size=44, color="white"),
            showarrow=False
        )

        # EFFICIENCY LABEL
        fig_gauge.add_annotation(
            x=0.5, y=-0.10,
            text=f"<b>Feed Efficiency: {eff_live:.2f} kg/kg</b>",
            font=dict(size=20, color="#facc15"),
            showarrow=False
        )

        fig_gauge.update_layout(
            height=480,
            margin=dict(l=0, r=0, t=60, b=40),
            showlegend=False,
            template=plotly_template,
        )

        st.plotly_chart(fig_gauge, width="stretch")

        st.markdown(
            """
            👉 **Use this to answer:**  
            *“If I change feed and health at this age, what weight can I expect?”*
            """
        )

    # ================================================================
    # EXPANDER 2 — Actual vs Predicted Line Chart
    # ================================================================
    with st.expander("📊 Actual vs Predicted – XGBoost Performance", expanded=False):

        comp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds_x}).reset_index(drop=True)

        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=comp_df.index,
                y=comp_df["Actual"],
                mode="lines",
                name="Actual",
                line=dict(color="#38bdf8", width=2),
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=comp_df.index,
                y=comp_df["Predicted"],
                mode="lines",
                name="Predicted (XGBoost)",
                line=dict(color="#f97316", width=2),
            )
        )
        fig_line.update_layout(
            template=plotly_template,
            height=420,
            xaxis_title="Test sample index",
            yaxis_title="Weight (kg)",
            title=f"Actual vs Predicted – XGBoost (R² = {r2_x:.3f})",
        )
        st.plotly_chart(fig_line, width="stretch")

    # ================================================================
    # EXPANDER 3 — Feed Sensitivity Curves
    # ================================================================
    with st.expander("📉 Feed Sensitivity Curves by Age", expanded=False):

        age_presets = [60, 90, 120, 150, 180]
        selected_ages = st.multiselect(
            "Select age(s):",
            options=age_presets,
            default=age_presets[:4],
            format_func=lambda x: f"Age {x} days",
        )

        feed_vals = np.linspace(1.0, 4.0, 40)

        fig_fs = go.Figure()
        for a in selected_ages:
            df_curve = pd.DataFrame({
                "RAISE_DAY": [a] * len(feed_vals),
                "FEED_INTAKE_ACTUAL": feed_vals,
                "RESPIRATORY_PERCENT": resp_in,
                "RESPIRATORY_PERCENT_FLAG": resp_flag_in,
                "FEED_X_AGE": feed_vals * a,
                "BARN_TARGET_ENC": [DEFAULT_BARN_TE] * len(feed_vals),
                "TEMP_DIFF": [DEFAULT_TEMP_DIFF] * len(feed_vals),
                "HEAT_STRESS": [0] * len(feed_vals),
                "RESP_SEVERITY": [resp_in / 100.0] * len(feed_vals),
                "REASON_SCORE": [DEFAULT_REASON_SCORE] * len(feed_vals),
            })
            preds_curve = xgb_model.predict(df_curve)

            fig_fs.add_trace(go.Scatter(
                x=feed_vals,
                y=preds_curve,
                mode="lines",
                name=f"Age {a} days",
            ))

        fig_fs.update_layout(
            template=plotly_template,
            height=450,
            xaxis_title="Feed Intake (kg/day)",
            yaxis_title="Predicted Weight (kg)",
            title="Feed vs Predicted Weight at Different Ages (XGBoost)",
        )
        st.plotly_chart(fig_fs, width="stretch")

    # ================================================================
    # EXPANDER 4 — Heatmaps (Predicted Weight & Efficiency)
    # ================================================================
    with st.expander("🌡️ Feed × Age → Predicted Weight & Efficiency Heatmaps", expanded=False):

        heat_ages = np.linspace(60, 190, 9, dtype=int)
        heat_feeds = np.linspace(1.0, 4.0, 25)

        # ------- Build prediction grid -------
        grid_rows = [
            {
                "RAISE_DAY": a,
                "FEED_INTAKE_ACTUAL": float(f),
                "RESPIRATORY_PERCENT": resp_in,
                "RESPIRATORY_PERCENT_FLAG": resp_flag_in,
                "FEED_X_AGE": f * a,
                "BARN_TARGET_ENC": DEFAULT_BARN_TE,
                "TEMP_DIFF": DEFAULT_TEMP_DIFF,
                "HEAT_STRESS": 0,
                "RESP_SEVERITY": resp_in / 100.0,
                "REASON_SCORE": DEFAULT_REASON_SCORE,
            }
            for a in heat_ages for f in heat_feeds
        ]

        df_grid = pd.DataFrame(grid_rows)
        df_grid["PRED_WEIGHT"] = xgb_model.predict(df_grid)
        df_grid["EFF"] = df_grid["PRED_WEIGHT"] / df_grid["FEED_INTAKE_ACTUAL"]

        # Normalised efficiency
        eff_min_val = df_grid["EFF"].min()
        eff_max_val = df_grid["EFF"].max()
        df_grid["EFF_NORM"] = (df_grid["EFF"] - eff_min_val) / (eff_max_val - eff_min_val + 1e-9)

        # ---------- Build pivots ----------
        pivot_pred = df_grid.pivot_table(index="RAISE_DAY", columns="FEED_INTAKE_ACTUAL", values="PRED_WEIGHT")
        pivot_eff = df_grid.pivot_table(index="RAISE_DAY", columns="FEED_INTAKE_ACTUAL", values="EFF_NORM")

        weight_z = pivot_pred.values
        eff_z = pivot_eff.values
        ages = pivot_pred.index.values
        feeds = pivot_pred.columns.values

        # NORMALISED weight for classification
        w_min = weight_z.min()
        w_max = weight_z.max()
        w_norm = (weight_z - w_min) / (w_max - w_min + 1e-9)

        # TOOLTIP generation
        hover_pred = []
        hover_eff = []

        for i, age in enumerate(ages):
            row_pred = []
            row_eff = []
            for j, feed in enumerate(feeds):
                w = weight_z[i, j]
                w_n = w_norm[i, j]
                e = eff_z[i, j]

                # Classification text
                cls_pred = (
                    "🔥 Highest predicted weight zone" if w_n >= 0.67
                    else "⚖️ Moderate predicted weight zone" if w_n >= 0.33
                    else "🐌 Lowest predicted weight zone"
                )
                cls_eff = (
                    "🔥 BEST efficiency zone" if e >= 0.67
                    else "⚖️ Moderate efficiency" if e >= 0.33
                    else "🐌 Poor efficiency"
                )

                eff_val = df_grid[
                    (df_grid["RAISE_DAY"] == age)
                    & (df_grid["FEED_INTAKE_ACTUAL"] == feed)
                ]["EFF"].mean()

                row_pred.append(
                    f"Age: {age} days<br>"
                    f"Feed: {feed:.2f} kg/day<br>"
                    f"Predicted weight: {w:.1f} kg<br>"
                    f"{cls_pred}"
                )
                row_eff.append(
                    f"Age: {age} days<br>"
                    f"Feed: {feed:.2f} kg/day<br>"
                    f"Efficiency: {eff_val:.2f}<br>"
                    f"{cls_eff}"
                )

            hover_pred.append(row_pred)
            hover_eff.append(row_eff)

        # ------------ Predicted Weight Heatmap ------------
        fig_heat_pred = go.Figure(data=go.Heatmap(
            z=weight_z,
            x=feeds,
            y=ages,
            colorscale="Viridis",
            colorbar=dict(title="Weight (kg)"),
            hoverinfo="text",
            text=hover_pred,
        ))
        fig_heat_pred.update_layout(
            template=plotly_template,
            height=480,
            xaxis_title="Feed Intake (kg/day)",
            yaxis_title="Age (RAISE_DAY)",
            title="Predicted Weight (kg) by Feed & Age",
        )
        st.plotly_chart(fig_heat_pred, width="stretch")

        # ------------ Efficiency Heatmap ------------
        fig_heat_eff = go.Figure(data=go.Heatmap(
            z=eff_z,
            x=feeds,
            y=ages,
            colorscale="RdYlGn",
            colorbar=dict(title="Normalised efficiency"),
            hoverinfo="text",
            text=hover_eff,
        ))
        fig_heat_eff.update_layout(
            template=plotly_template,
            height=480,
            xaxis_title="Feed Intake (kg/day)",
            yaxis_title="Age (RAISE_DAY)",
            title="Feed Efficiency (kg gain per kg feed)",
        )
        st.plotly_chart(fig_heat_eff, width="stretch")

    # ================================================================
    # EXPANDER 5 — Interpretation
    # ================================================================
    with st.expander("🧠 Interpretation Guide", expanded=False):
        st.markdown(
            """
            **How to interpret the heatmaps & charts:**

            - 🔥 **Bright zones** (Predicted Weight heatmap) identify **heaviest pigs** at given feed × age.
            - 🟩 **Green zones** (Efficiency heatmap) identify **most feed-efficient combinations**.
            - 🟥 **Red zones** reveal where added feed **does not convert well** into weight.
            - Optimal strategy:
              - Increase feed in **bright + green overlapping zones**.
              - Avoid overspending feed in **red zones**, especially at late ages.
            """
        )

    # ================================================================
    # Developer Mode – Model Metrics
    # ================================================================
    if dev_mode:
        with st.expander("🧪 Developer Mode – Model Metrics", expanded=False):
            st.markdown(
                f"""
                **Model Performance (XGBoost)**  
                - MAE: `{mae_x:.3f}`  
                - RMSE: `{rmse_x:.3f}`  
                - R²: `{r2_x:.3f}`
                """
            )

 # ================================================================
# PAGE 7 — EXPLAINABLE AI (SHAP)
# ================================================================
elif page == "🧠 Explainable AI (SHAP)":

    st.markdown("## 🧠 Explainable AI – SHAP for XGBoost")
    

    st.markdown("""
        This page shows **why** the XGBoost model predicts a certain weight:
        - Which features push weight **up or down**
        - How feed, age, barn performance and temperature interact
        - Helps transform ML into **actionable insights**
    """)

    # ------------------------------------------------------------
    # Compute SHAP background sample
    # ------------------------------------------------------------
    X_bg = X_train_raw.copy()
    if len(X_bg) > 400:
        X_bg = X_bg.sample(400, random_state=42)

    pipe = xgb_model
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    reg = pipe.named_steps["regressor"]

    X_imp = imputer.transform(X_bg)
    X_scaled = scaler.transform(X_imp)

    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X_scaled)

    # ------------------------------------------------------------
    # EXPANDER 1 — Summary Dot Plot
    # ------------------------------------------------------------
    with st.expander("📌 SHAP Summary Plot (Dot Plot)", expanded=False):
        plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=FEATURES,
            show=False,
            plot_size=(7, 5),
        )
        st.pyplot(plt.gcf(), clear_figure=True)

    # ------------------------------------------------------------
    # EXPANDER 2 — Mean |SHAP| Bar Plot
    # ------------------------------------------------------------
    with st.expander("📊 Mean |SHAP| Feature Impact (Bar Plot)", expanded=False):
        plt.figure(figsize=(7, 4))
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=FEATURES,
            plot_type="bar",
            show=False,
            plot_size=(7, 4),
        )
        st.pyplot(plt.gcf(), clear_figure=True)

    # ------------------------------------------------------------
    # EXPANDER 3 — Interpretation
    # ------------------------------------------------------------
    with st.expander("🧭 Management Interpretation", expanded=False):
        st.markdown("""
        **Key insights from SHAP:**

        - **RAISE_DAY**, **FEED_INTAKE_ACTUAL**, and **FEED_X_AGE**  
          → structural drivers of growth.

        - **BARN_TARGET_ENC**  
          → highlights barns consistently above/below expected weight.

        - **TEMP_DIFF** and **HEAT_STRESS**  
          → quantify **climate risk impact** on weight gain.

        - **RESP_SEVERITY** and **REASON_SCORE**  
          → represent health burden & management issues that suppress growth.

        Use SHAP to **audit the model**, justify predictions and design farm interventions.
        """)

    # ------------------------------------------------------------
    # EXPANDER 4 — Developer Mode (SAFE)
    # ------------------------------------------------------------
    if dev_mode:
        with st.expander("🧪 Developer Mode – SHAP Debug Info", expanded=False):
            st.write("Background sample size:", len(X_bg))
            st.write("Feature list used for SHAP:", FEATURES)
            st.write("Scaled input shape:", X_scaled.shape)

       
# ================================================================
# PAGE 6 — HARVEST & REVENUE SIMULATOR (WITH EXPANDERS + DEV MODE)
# ================================================================
elif page == "💰 Harvest & Revenue Simulator":

    st.markdown("## 💰 Harvest & Revenue Simulator – XGBoost Model")
    
    st.markdown("""
        Simulate profitability using the **XGBoost predictive model** by adjusting:
        - Feed intake  
        - Days to harvest  
        - Market prices  
        - Herd size  
    """)

    # ============================================================
    # EXPANDER 1 — SIMULATION INPUTS
    # ============================================================
    with st.expander("⚙️ Simulation Inputs", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            feed_cost = st.number_input(
                "Feed Cost per kg (SGD):", min_value=0.5, max_value=5.0,
                value=1.2, step=0.05
            )
            pork_price = st.number_input(
                "Pork Selling Price per kg (SGD):",
                min_value=2.0, max_value=20.0,
                value=6.5, step=0.1,
            )
            days_to_simulate = st.slider(
                "Days until Harvest", min_value=1, max_value=50,
                value=14, step=1
            )

        with c2:
            feed_intake = st.slider(
                "Daily Feed Intake (kg/pig)",
                min_value=1.0, max_value=4.0,
                value=2.3, step=0.1,
            )
            base_age = st.number_input(
                "Current Age (RAISE_DAY)", min_value=30,
                max_value=200, value=120
            )
            herd_size = st.number_input(
                "Number of Pigs", min_value=10,
                max_value=5000, value=200, step=10
            )

    st.markdown("---")

    # ============================================================
    # SIMULATION ENGINE
    # ============================================================
    sim_days = np.arange(0, days_to_simulate + 1)
    sim_age = base_age + sim_days
    sim_feed = np.full_like(sim_days, feed_intake, dtype=float)

    sim_df = pd.DataFrame({
        "RAISE_DAY": sim_age,
        "FEED_INTAKE_ACTUAL": sim_feed,
        "RESPIRATORY_PERCENT": 0.0,
        "RESPIRATORY_PERCENT_FLAG": 0.0,
        "FEED_X_AGE": sim_feed * sim_age,
        "BARN_TARGET_ENC": [DEFAULT_BARN_TE] * len(sim_days),
        "TEMP_DIFF": [DEFAULT_TEMP_DIFF] * len(sim_days),
        "HEAT_STRESS": [0] * len(sim_days),
        "RESP_SEVERITY": [0.0] * len(sim_days),
        "REASON_SCORE": [DEFAULT_REASON_SCORE] * len(sim_days),
    })

    xgb_sim = xgb_model.predict(sim_df)
    sim_df["PRED_WEIGHT"] = xgb_sim


    # ============================================================
    # EXPANDER 2 — PREDICTED WEIGHT CHART
    # ============================================================
    with st.expander("📈 Predicted Weight Curve", expanded=False):

        fig_sim = go.Figure()
        fig_sim.add_trace(
            go.Scatter(
                x=sim_days,
                y=sim_df["PRED_WEIGHT"],
                mode="lines+markers",
                name="Predicted Weight",
                line=dict(width=3)
            )
        )
        fig_sim.update_layout(
            template=plotly_template,
            height=450,
            xaxis_title="Days from Today",
            yaxis_title="Predicted Weight (kg/pig)",
        )
        st.plotly_chart(fig_sim, width="stretch")


    # ============================================================
    # METRICS SUMMARY
    # ============================================================
    final_weight = float(sim_df["PRED_WEIGHT"].iloc[-1])
    total_feed_cost = feed_cost * feed_intake * days_to_simulate * herd_size
    expected_revenue = final_weight * pork_price * herd_size
    profit = expected_revenue - total_feed_cost

    c1, c2, c3 = st.columns(3)
    c1.metric("Final Predicted Weight (per pig)", f"{final_weight:.1f} kg")
    c2.metric("Total Feed Cost", f"${total_feed_cost:,.2f}")
    c3.metric("Net Profit (Batch)", f"${profit:,.2f}")

    # Interpretation
    with st.expander("🧭 Interpretation & Recommendations", expanded=False):
        if profit > 0:
            st.success(
                "✔ **Scenario is profit-positive**.\n"
                "Try adjusting feed or extending harvest days to improve ROI."
            )
        else:
            st.error(
                "⚠ **Scenario is loss-making**.\n"
                "- Reduce days to harvest\n"
                "- Lower feeding cost\n"
                "- Improve health/climate conditions\n"
            )


    # ============================================================
    # EXPANDER 3 — PDF REPORT GENERATION
    # ============================================================
    with st.expander("📄 Download PDF Report", expanded=False):

        pdf_buf = BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4)
        style = ParagraphStyle(name="Normal", fontName="Helvetica",
                               fontSize=10, leading=14)
        story = []

        story.append(Paragraph("<b>Harvest & Revenue Simulation Report</b>", style))
        story.append(Paragraph(f"Feed Intake: {feed_intake:.2f} kg/day/pig", style))
        story.append(Paragraph(f"Days to Harvest: {days_to_simulate}", style))
        story.append(Paragraph(f"Final Predicted Weight: {final_weight:.1f} kg/pig", style))
        story.append(Paragraph(f"Herd Size: {herd_size}", style))
        story.append(Paragraph(f"Feed Cost per kg: ${feed_cost:.2f}", style))
        story.append(Paragraph(f"Pork Price per kg: ${pork_price:.2f}", style))
        story.append(Paragraph(f"Total Feed Cost: ${total_feed_cost:,.2f}", style))
        story.append(Paragraph(f"Expected Revenue: ${expected_revenue:,.2f}", style))
        story.append(Paragraph(f"Net Profit: ${profit:,.2f}", style))

        doc.build(story)

        st.download_button(
            "📥 Download PDF Report",
            pdf_buf.getvalue(),
            file_name="harvest_revenue_simulation_report.pdf",
            mime="application/pdf",
        )

    # ============================================================
    # EXPANDER 4 — DEVELOPER MODE
    # ============================================================
    if dev_mode:
        with st.expander("🧪 Developer Mode – Simulation Debug Info", expanded=False):
            st.write("Sample simulation dataframe:")
            st.dataframe(sim_df.head(), width="stretch")
            st.write("Final predicted weight:", final_weight)
            st.write("Model input columns:", list(sim_df.columns))

# ================================================================
# PAGE 8 — ASK THE FARM AI
# ================================================================
    #st.write("DEBUG: Page Loaded")  # temp
elif page == "🤖 Ask the Farm AI":
    st.markdown("## 🤖 Ask the Farm AI – Guided Q&A + Charts")
    
    st.markdown(
        """
        Ask questions in natural language or use the quick buttons below.  
        The assistant will respond with **charts + text insights**.
        """
    )

    col_q, col_help = st.columns([2, 1])
    with col_q:
        user_q = st.text_area(
            "Type your question (e.g. 'Which barn is best?', 'Show barns with highest respiratory problems'):",
            key="farm_ai_question",
        )
    with col_help:
        st.markdown("#### 🔍 Quick question shortcuts")
        if st.button("🏆 Which barn is top performer?"):
            user_q = "Which barn is top performer?"
        if st.button("⚠️ Which barns underperform the most?"):
            user_q = "Which barns underperform the most?"
        if st.button("🩺 Which barns have highest respiratory problems?"):
            user_q = "Which barns have highest respiratory problems?"
        if st.button("🥕 Show feed efficiency ranking"):
            user_q = "Show feed efficiency ranking"
        if st.button("📈 Show growth curve for top barn"):
            user_q = "Show growth curve for top barn"

    if not user_q:
        st.info("Ask a question or click one of the quick buttons on the right.")
    else:
        q = user_q.lower()

        # 1) Top performer
        if ("top" in q and "barn" in q) or ("best barn" in q):
            if barn_perf is not None and not barn_perf.empty:
                bp_sorted = barn_perf.sort_values("UNIFIED_SCORE", ascending=False)
                top3 = bp_sorted.head(3)
                st.markdown("### 🏆 Top Performing Barns (Unified Score)")
                fig_top = px.bar(
                    top3,
                    x="BARN_CLEAN",
                    y="UNIFIED_SCORE",
                    color="UNIFIED_SCORE",
                    color_continuous_scale="Greens",
                    title="Top 3 Barns by Unified Score",
                )
                fig_top.update_layout(template=plotly_template, height=400)
                st.plotly_chart(fig_top, width="stretch")

                st.markdown(
                    """
                    - These barns combine **strong growth**, **good sample coverage** and **stable health**.
                    - Use them as **best-practice references** for SOPs, feeding programs and climate management.
                    """
                )
            else:
                st.warning("Barn performance metrics not available.")

        # 2) Underperforming barns
        elif "underperform" in q or "worst" in q:
            if barn_perf is not None and not barn_perf.empty:
                bp_sorted = barn_perf.sort_values("UNIFIED_SCORE", ascending=True)
                worst3 = bp_sorted.head(3)
                st.markdown("### ⚠️ Underperforming Barns (Lowest Unified Score)")
                fig_worst = px.bar(
                    worst3,
                    x="BARN_CLEAN",
                    y="UNIFIED_SCORE",
                    color="UNIFIED_SCORE",
                    color_continuous_scale="Reds",
                    title="Bottom 3 Barns by Unified Score",
                )
                fig_worst.update_layout(template=plotly_template, height=400)
                st.plotly_chart(fig_worst, width="stretch")

                st.markdown(
                    """
                    - These barns are **priority targets** for management review.
                    - Check **feed, stocking, climate and health protocols** versus top barns.
                    """
                )
            else:
                st.warning("Barn performance metrics not available.")

        # 3) Respiratory problems
        elif "respiratory" in q or "health" in q or "disease" in q:
            if barn_perf is not None and not barn_perf.empty:
                st.markdown("### 🩺 Barns with Highest Respiratory Issue Rates")
                bp_sorted = barn_perf.sort_values("RESP_RATE_PCT", ascending=False)
                top5 = bp_sorted.head(5)
                fig_resp = px.bar(
                    top5,
                    x="BARN_CLEAN",
                    y="RESP_RATE_PCT",
                    color="RESP_RATE_PCT",
                    color_continuous_scale="Reds",
                    title="Top 5 Barns by Respiratory Cases (%)",
                )
                fig_resp.update_layout(template=plotly_template, height=400)
                st.plotly_chart(fig_resp, width="stretch")

                st.markdown(
                    """
                    - Focus **vaccination, biosecurity and ventilation checks** on these barns.
                    - Compare with barns with **low respiratory rates** to copy good practices.
                    """
                )
            else:
                st.warning("Barn performance metrics not available.")

        # 4) Feed efficiency ranking
        elif "efficiency" in q or "feed efficiency" in q:
            if barn_perf is not None and not barn_perf.empty:
                st.markdown("### 🥕 Feed Efficiency Ranking (ADG / Feed)")
                eff_df = barn_perf.dropna(subset=["FEED_EFF"]).copy()
                if eff_df.empty:
                    st.info("Feed efficiency values not available.")
                else:
                    eff_sorted = eff_df.sort_values("FEED_EFF", ascending=False)
                    fig_eff = px.bar(
                        eff_sorted,
                        x="BARN_CLEAN",
                        y="FEED_EFF",
                        color="FEED_EFF",
                        color_continuous_scale="Greens",
                        title="Barns by Feed Efficiency (higher is better)",
                    )
                    fig_eff.update_layout(template=plotly_template, height=420)
                    st.plotly_chart(fig_eff, width="stretch")

                    st.markdown(
                        """
                        - Barns on the left convert **each kg of feed into more growth**.
                        - Investigate feed type, feeding schedule and stocking density in high-efficiency barns.
                        """
                    )
            else:
                st.warning("Barn performance metrics not available.")

        # 5) Growth curve for top barn
        elif "growth curve" in q or "growth" in q:
            if barn_perf is not None and not barn_perf.empty:
                top_barn_id = (
                    barn_perf.sort_values("UNIFIED_SCORE", ascending=False)["BARN_CLEAN"].iloc[0]
                )
                st.markdown(f"### 📈 Growth Curve – Top Barn **{top_barn_id}**")

                df_top = df[df["BARN"] == top_barn_id].dropna(
                    subset=["RAISE_DAY", "ESTIMATED_WEIGHT"]
                )
                if df_top.empty:
                    st.info(f"No growth data available for barn {top_barn_id}.")
                else:
                    grp_top = (
                        df_top.groupby("RAISE_DAY", as_index=False)["ESTIMATED_WEIGHT"]
                        .mean()
                        .rename(columns={"ESTIMATED_WEIGHT": "MEAN_WEIGHT"})
                    )
                    if len(grp_top) >= 5:
                        loess_res_top = lowess(
                            endog=grp_top["MEAN_WEIGHT"],
                            exog=grp_top["RAISE_DAY"],
                            frac=0.3,
                            return_sorted=True,
                        )
                        loess_top = pd.DataFrame(
                            loess_res_top, columns=["RAISE_DAY", "WEIGHT_SMOOTH"]
                        )
                        curve_top = grp_top.merge(loess_top, on="RAISE_DAY", how="left")

                        fig_gc = go.Figure()
                        fig_gc.add_trace(
                            go.Scatter(
                                x=curve_top["RAISE_DAY"],
                                y=curve_top["MEAN_WEIGHT"],
                                mode="markers",
                                name="Mean Weight",
                                marker=dict(size=6, opacity=0.6, color="#38bdf8"),
                            )
                        )
                        fig_gc.add_trace(
                            go.Scatter(
                                x=curve_top["RAISE_DAY"],
                                y=curve_top["WEIGHT_SMOOTH"],
                                mode="lines",
                                name="Smoothed Growth Curve (LOESS)",
                                line=dict(width=3, color="#facc15"),
                            )
                        )
                        fig_gc.update_layout(
                            template=plotly_template,
                            height=420,
                            xaxis_title="RAISE_DAY (Age in days)",
                            yaxis_title="Weight (kg)",
                        )
                        st.plotly_chart(fig_gc, width="stretch")
                    else:
                        st.info(
                            f"Not enough distinct ages to build a smoothed curve for barn {top_barn_id}."
                        )
            else:
                st.warning("Barn performance metrics not available.")

        else:
            st.info(
                """
                I couldn't match your question to a known pattern.  
                Try examples like:
                - "Which barn is top performer?"
                - "Which barns underperform the most?"
                - "Which barns have highest respiratory problems?"
                - "Show feed efficiency ranking"
                - "Show growth curve for top barn"
                """
            )

  
# ================================================================
# PAGE 5 🏭 PRODUCTION OPERATIONS PAGE (FULLY FIXED + EXPANDERS + DEV MODE)
# ================================================================
elif page == "🏭 Production Operations":

    st.markdown("## 🏭 Production Operations & Barn Performance Insights")
    

    st.markdown("""
        Operational insights to improve feeding strategy, growth performance,  
        health stability, and benchmarking across barns.
    """)

    # ----------------------------------------------------------------
    # 1️⃣ Build Barn Performance Table
    # ----------------------------------------------------------------
    barn_perf = build_barn_performance(df)

    if barn_perf.empty:
        st.error("No barn performance data available.")
        st.stop()

    barn_perf = barn_perf.sort_values("UNIFIED_SCORE", ascending=False).reset_index(drop=True)
    barn_perf["RANK"] = barn_perf.index + 1

    top_barn = barn_perf.iloc[0]
    worst_barn = barn_perf.iloc[-1]

    
    # ================================================================
    # 1️⃣ Barn Performance Summary (NO EXPANDER)
    # ================================================================

    st.markdown("### 📌 Barn Performance Summary")

    barn_perf = barn_perf.sort_values("UNIFIED_SCORE", ascending=False).reset_index(drop=True)
    barn_perf["RANK"] = barn_perf.index + 1

    top_barn = barn_perf.iloc[0]
    worst_barn = barn_perf.iloc[-1]

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "🏆 Top Performing Barn",
            f"{top_barn['BARN_CLEAN']} (Score: {top_barn['UNIFIED_SCORE']:.2f})",
        )

    with colB:
        st.metric(
            "📉 Lowest Performing Barn",
            f"{worst_barn['BARN_CLEAN']} (Score: {worst_barn['UNIFIED_SCORE']:.2f})",
        )



    #st.markdown("---")

    # ============================================================
    # EXPANDER 1 — BARN SCORE TABLE
    # ============================================================
    with st.expander("📊 Barn Performance Score Table", expanded=False):

        # FIXED COLUMN NAMES — matches your build_barn_performance()
        score_table = barn_perf[
    [
        "RANK",
        "BARN_CLEAN",
        "UNIFIED_SCORE",
        "MEDIAN_ADG",
        "SAMPLE_STRENGTH_SCORE",
        "HEALTH_STABILITY_SCORE",
        "RESP_RATE_PCT",
        "AVG_WEIGHT",
        "N_RECORDS",
    ]
]


        st.dataframe(score_table, width=1200)

    st.markdown("---")

    # ============================================================
    # EXPANDER 2 — RADAR COMPARISON
    # ============================================================
    with st.expander("🎯 Barn Comparison Radar Chart", expanded=False):

        barn_list = barn_perf["BARN_CLEAN"].tolist()
        barn_select = st.multiselect(
            "Select up to 3 barns to compare", barn_list, default=barn_list[:3]
        )

        radar_metrics = [
            "MEDIAN_ADG",
            "SAMPLE_STRENGTH_SCORE",
            "HEALTH_STABILITY_SCORE",
            "UNIFIED_SCORE",
        ]

        radar_df = barn_perf[barn_perf["BARN_CLEAN"].isin(barn_select)][["BARN_CLEAN"] + radar_metrics]

        # Normalize 0–1
        for colname in radar_metrics:
            colmin, colmax = barn_perf[colname].min(), barn_perf[colname].max()
            if colmin == colmax:
                radar_df[colname] = 0.5
            else:
                radar_df[colname] = (radar_df[colname] - colmin) / (colmax - colmin)

        fig_radar = go.Figure()
        for _, row in radar_df.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=row[radar_metrics].values,
                    theta=radar_metrics,
                    fill="toself",
                    name=row["BARN_CLEAN"],
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template=st.session_state.plotly_template,
            height=500,
        )

        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("---")

    # ============================================================
    # EXPANDER 3 — ADG DISTRIBUTION
    # ============================================================
    with st.expander("📈 ADG Distribution Across Barns", expanded=False):

        df_adg = df.dropna(subset=["BARN_CLEAN", "ADG"]).copy()
        df_adg["BARN_CLEAN"] = df_adg["BARN_CLEAN"].astype(str)

        fig_adg = px.box(
            df_adg,
            x="BARN_CLEAN",
            y="ADG",
            points="all",
            title="ADG Distribution Across Barns",
            template=st.session_state.plotly_template,
        )

        st.plotly_chart(fig_adg, width="stretch")

    st.markdown("---")

    # ============================================================
    # EXPANDER 4 — LOESS CURVE
    # ============================================================
    with st.expander("🐖 Smoothed Growth Curve (LOESS) by Barn", expanded=False):

        barn_choice = st.selectbox("Select barn to visualize growth curve:", barn_list)
        df_barn = df[df["BARN_CLEAN"] == barn_choice].dropna(subset=["RAISE_DAY", "ESTIMATED_WEIGHT"])

        if df_barn.empty:
            st.warning(f"No valid data for barn {barn_choice}.")
        else:
            grp = df_barn.groupby("RAISE_DAY", as_index=False)["ESTIMATED_WEIGHT"].mean()
            grp.rename(columns={"ESTIMATED_WEIGHT": "MEAN_WEIGHT"}, inplace=True)

            loess_res = lowess(
                endog=grp["MEAN_WEIGHT"],
                exog=grp["RAISE_DAY"],
                frac=0.3,
                return_sorted=True,
            )
            loess_df = pd.DataFrame(loess_res, columns=["RAISE_DAY", "WEIGHT_SMOOTH"])

            fig_loess = go.Figure()
            fig_loess.add_trace(go.Scatter(x=grp["RAISE_DAY"], y=grp["MEAN_WEIGHT"],
                                           mode="markers", name="Observed Mean Weight"))
            fig_loess.add_trace(go.Scatter(x=loess_df["RAISE_DAY"], y=loess_df["WEIGHT_SMOOTH"],
                                           mode="lines", name="Smoothed LOESS Curve"))

            fig_loess.update_layout(
                title=f"Growth Curve for Barn {barn_choice}",
                xaxis_title="Age (days)",
                yaxis_title="Estimated Weight (kg)",
                template=st.session_state.plotly_template,
            )

            st.plotly_chart(fig_loess, width="stretch")

    st.markdown("---")

    # ============================================================
    # EXPANDER 5 — MANAGEMENT INSIGHTS
    # ============================================================
    with st.expander("🧠 Operational Insights", expanded=False):

        st.markdown(f"""
        - 🏆 **Top Barn ({top_barn['BARN_CLEAN']})** shows strong growth and stability.  
        - 📉 **Lowest Barn ({worst_barn['BARN_CLEAN']})** needs feed optimization or health checks.  
        - 🎯 Radar chart helps compare barns across growth, stability, and health.  
        - 🐖 LOESS curves reveal whether barns follow expected growth trajectories.  
        """)

    # ============================================================
    # EXPANDER 6 — DEVELOPER MODE
    # ============================================================
    if dev_mode:
        with st.expander("🧪 Developer Mode – Raw Barn Table", expanded=False):
            st.dataframe(barn_perf, width="stretch")

# ================================================================
# PAGE 4 — K-MEANS SEGMENTATION (GROWTH + HEALTH + BARN)
# ================================================================
elif page == "🧬 K-Means Segmentation":
    st.markdown("## 🧬 K-Means Segmentation – Growth & Health Clusters")
    
    st.markdown(
        """
        Unsupervised learning (**K-Means**) helps discover **hidden patterns**:
        - Identify **high / medium / low performing barns**
        - Find **fast vs slow growers**
        - Detect **health-risk clusters**
        """
    )

    # ------------------------------------------------------------------
    # Ensure BARN_CLEAN exists in df and barn_perf
    # ------------------------------------------------------------------
    if "BARN_CLEAN" not in df.columns and "BARN" in df.columns:
        df["BARN_CLEAN"] = df["BARN"].apply(clean_barn_code)

    if barn_perf is not None and not barn_perf.empty:
        if "BARN_CLEAN" not in barn_perf.columns and "BARN" in barn_perf.columns:
            barn_perf["BARN_CLEAN"] = barn_perf["BARN"].apply(clean_barn_code)

    
    # ------------------------------------------------------------------
    # What to segment?
    # ------------------------------------------------------------------
    mode = st.radio(
        "What would you like to segment?",
        ["Barn performance", "Individual growth", "Health patterns"],
        horizontal=True,
    )

    # ------------------------------------------------------------------
    # Helper: name clusters by ranking on a metric
    # ------------------------------------------------------------------
    def name_clusters_by_metric(center_df, metric, base_names):
        """
        center_df : DataFrame of cluster centers (index = cluster id)
        metric    : column to sort by (higher = better)
        base_names: list of human-friendly names in descending order
        """
        order = center_df[metric].sort_values(ascending=False).index.tolist()
        mapping = {}
        for i, cid in enumerate(order):
            if i < len(base_names):
                mapping[cid] = base_names[i]
            else:
                mapping[cid] = f"Cluster {cid}"
        return mapping

    # ============================================================
    # INDUSTRIAL STANDARD CLUSTER COLORS (GROWTH + HEALTH)
    # ============================================================
    cluster_color_map = {
        # Growth segmentation (Individual)
        "Fast Growers": "#16a34a",              # Green
        "Above-Average Growers": "#84cc16",     # Light Green
        "Moderate Growers": "#eab308",          # Yellow
        "Slow Growers": "#dc2626",              # Red
        "Critical Slow Growers": "#991b1b",     # Dark Red

        # Barn performance segmentation
        "High Performance": "#16a34a",
        "Moderate Performance": "#eab308",
        "Low Performance": "#dc2626",
        "Developing": "#0ea5e9",
        "Critical Risk": "#7e22ce",

        # Health segmentation
        "Low-Risk Pens": "#16a34a",
        "Moderate-Risk Pens": "#eab308",
        "Elevated-Risk Pens": "#f97316",
        "High-Risk Pens": "#dc2626",
    }


    # ============================================================
    # MODE 1 — BARN PERFORMANCE SEGMENTATION
    # ============================================================
    if mode == "Barn performance":
        st.subheader("🏠 Barn Performance Segmentation")

        if barn_perf is None or barn_perf.empty:
            st.warning("Barn performance table is empty.")
        else:
            features = [
                "MEDIAN_ADG",
                "FEED_EFF",
                "RESP_RATE",
                "WEIGHT_STD",
                "N_AGES",
                "N_RECORDS",
            ]

            df_b = barn_perf.dropna(subset=features).copy()
            if df_b.empty:
                st.warning("Not enough barns with complete data.")
            else:
                # ---------------------------------------------
                # K slider
                # ---------------------------------------------
                k = st.slider("Number of clusters (k)", 2, 6, 3)

                # ---------------------------------------------
                # Scale data & fit model
                # ---------------------------------------------
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_b[features])

                

                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                df_b["Cluster"] = kmeans.fit_predict(X_scaled)

                centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)

                # ---------------------------------------------
                # Name clusters by MEDIAN_ADG (higher = better)
                # ---------------------------------------------
                barn_name_pool = [
                    "High Performance",
                    "Moderate Performance",
                    "Low Performance",
                    "Developing",
                    "Critical Risk",
                ]
                cluster_name_map = name_clusters_by_metric(
                    centers, "MEDIAN_ADG", barn_name_pool
                )

                df_b["Cluster_Label"] = df_b["Cluster"].map(cluster_name_map)

                # Propagate labels back to barn_perf & df (by BARN_CLEAN)
                if "BARN_CLEAN" in df_b.columns:
                    lookup = df_b.set_index("BARN_CLEAN")["Cluster_Label"]
                    barn_perf["Cluster_Label"] = barn_perf["BARN_CLEAN"].map(lookup)
                    if "BARN_CLEAN" in df.columns:
                        df["Cluster_Label"] = df["BARN_CLEAN"].map(lookup)

                # ===========================
                # 1) SCATTER: Growth vs Feed
                # ===========================
                with st.expander(
                    "📈 Growth vs Feed Efficiency (Cluster View)", expanded=False
                ):
                    fig_scatter = px.scatter(
                        df_b,
                        x="MEDIAN_ADG",
                        y="FEED_EFF",
                        color="Cluster_Label",
                        hover_data=["BARN_CLEAN", "RESP_RATE", "N_RECORDS"],
                        template=plotly_template,
                        title="Barn Performance Segmentation",
                        color_discrete_map=cluster_color_map,
                    )

                    fig_scatter.update_layout(height=480)
                    st.plotly_chart(fig_scatter, width="stretch")

                # ===========================
                # 2) RADAR: Cluster profiles
                # ===========================
                with st.expander(
                    "🧭 Cluster Profiles (Radar Chart)", expanded=False
                ):
                    centers_norm = (centers - centers.min()) / (
                        centers.max() - centers.min()
                    )
                    centers_norm = centers_norm.fillna(0.5)

                    labels = [
                        "ADG",
                        "Feed Eff",
                        "Resp %",
                        "Weight Var",
                        "Age Cov",
                        "Record Cov",
                    ]
                    feat_map = dict(
                        zip(
                            labels,
                            [
                                "MEDIAN_ADG",
                                "FEED_EFF",
                                "RESP_RATE",
                                "WEIGHT_STD",
                                "N_AGES",
                                "N_RECORDS",
                            ],
                        )
                    )

                    fig_radar = go.Figure()
                    for cid in centers.index:
                        r_vals = [centers_norm.loc[cid, feat_map[l]] for l in labels]
                        label = cluster_name_map.get(cid, f"Cluster {cid}")
                        fig_radar.add_trace(
                            go.Scatterpolar(
                                r=r_vals,
                                theta=labels,
                                fill="toself",
                                name=label,
                            )
                        )

                    fig_radar.update_layout(
                        template=plotly_template,
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        height=520,
                    )
                    st.plotly_chart(fig_radar, width="stretch")

                # ===========================
                # 3) SUMMARY TABLE + INSIGHTS
                # ===========================
                with st.expander("📋 Cluster Summary (Barn Level)", expanded=False):
                    summary = df_b.groupby("Cluster")[features].mean().reset_index()
                    summary["Cluster_Label"] = summary["Cluster"].map(cluster_name_map)
                    st.dataframe(summary, width="stretch")

                with st.expander("🧠 Manager Insights", expanded=False):
                    best = summary.sort_values("MEDIAN_ADG", ascending=False).iloc[0]
                    worst = summary.sort_values("MEDIAN_ADG", ascending=True).iloc[0]

                    best_label = best["Cluster_Label"]
                    worst_label = worst["Cluster_Label"]

                    barns_best = df_b[df_b["Cluster_Label"] == best_label]["BARN_CLEAN"].unique()
                    barns_worst = df_b[df_b["Cluster_Label"] == worst_label]["BARN_CLEAN"].unique()

                    st.markdown(
                        f"""
                        - **Top barn cluster:** `{best_label}`  
                          Barns: **{", ".join(barns_best.astype(str))}**  

                        - **Weakest barn cluster:** `{worst_label}`  
                          Barns: **{", ".join(barns_worst.astype(str))}**  

                        👉 Use **high-performance barns** as benchmarks for feed plan,
                        ventilation, and SOPs. Prioritise **interventions** in the weakest cluster.
                        """
                    )

                    if dev_mode:
                        st.markdown("---")
                        st.markdown("### Raw Cluster Table (Developer Mode)")
                        st.dataframe(df_b.sort_values("Cluster"), width="stretch")

                # ============================================================
                # 4) BARN CLUSTER MOVEMENT ACROSS GROWTH STAGES
                # ============================================================
                with st.expander(
                    "📈 Barn Cluster Movement Across Growth Stages", expanded=False
                ):

                    if "BARN_CLEAN" not in df.columns:
                        st.warning("BARN_CLEAN not found in main data.")
                    elif "Cluster_Label" not in barn_perf.columns:
                        st.warning("Cluster labels not found. Run clustering above first.")
                    else:
                        # 4.1 Map growth stages
                        def map_growth_period(day):
                            if day < 60:
                                return "Nursery (0–59d)"
                            elif day < 90:
                                return "Early Grower (60–89d)"
                            elif day < 120:
                                return "Late Grower (90–119d)"
                            elif day < 150:
                                return "Early Finisher (120–149d)"
                            else:
                                return "Late Finisher (150+d)"

                        stage_order = [
                            "Nursery (0–59d)",
                            "Early Grower (60–89d)",
                            "Late Grower (90–119d)",
                            "Early Finisher (120–149d)",
                            "Late Finisher (150+d)",
                        ]

                        df_bm = df[["BARN_CLEAN", "RAISE_DAY"]].dropna().copy()
                        df_bm["Period"] = df_bm["RAISE_DAY"].apply(map_growth_period)

                        # Merge cluster label (by BARN_CLEAN)
                        df_bm = df_bm.merge(
                            barn_perf[["BARN_CLEAN", "Cluster_Label"]],
                            on="BARN_CLEAN",
                            how="left",
                        )

                        # Transition table
                        movement = (
                            df_bm.groupby(["BARN_CLEAN", "Period"])["Cluster_Label"]
                            .first()
                            .reset_index()
                        )

                        st.markdown("### 🔄 Cluster Transition Table")
                        pivot_m = movement.pivot(
                            index="BARN_CLEAN", columns="Period", values="Cluster_Label"
                        )
                        st.dataframe(pivot_m, width="stretch")

                        # Distribution of cluster types per stage
                        st.markdown("### 📊 Cluster Distribution per Growth Stage")
                        dist_counts = (
                            df_bm.groupby(["Period", "Cluster_Label"])["BARN_CLEAN"]
                            .nunique()
                            .reset_index()
                            .rename(columns={"BARN_CLEAN": "Barn_Count"})
                        )

                        dist_counts = dist_counts[
                            dist_counts["Cluster_Label"].notna()
                            & dist_counts["Period"].notna()
                        ]

                        cluster_palette = {
                            "High Performance": "#22c55e",
                            "Moderate Performance": "#eab308",
                            "Low Performance": "#ef4444",
                            "Developing": "#0ea5e9",
                            "Critical Risk": "#a855f7",
                        }

                        fig_dist = px.bar(
                            dist_counts,
                            x="Period",
                            y="Barn_Count",
                            color="Cluster_Label",
                            barmode="stack",
                            title="Distribution of Cluster Types per Growth Stage",
                            color_discrete_map=cluster_palette,
                        )
                        fig_dist.update_layout(template=plotly_template, height=420)
                        st.plotly_chart(fig_dist, width="stretch")

                        # 4.3 Sankey: Movement across stages
                        st.markdown("### 🌊 Movement of Barns Across Growth Stages")

                        # Helper for rgba color
                        def hex_to_rgba(hex_color, alpha=0.55):
                            hex_color = hex_color.lstrip("#")
                            r = int(hex_color[0:2], 16)
                            g = int(hex_color[2:4], 16)
                            b = int(hex_color[4:6], 16)
                            return f"rgba({r},{g},{b},{alpha})"

                        # Nodes: each (stage, cluster_label)
                        valid_m = movement.dropna(subset=["Cluster_Label", "Period"]).copy()
                        unique_clusters = sorted(valid_m["Cluster_Label"].unique())
                        label_to_id = {}
                        node_labels = []
                        node_colors = []

                        for stage in stage_order:
                            for clabel in unique_clusters:
                                node_id = len(node_labels)
                                label_to_id[(stage, clabel)] = node_id
                                node_labels.append(f"{clabel} – {stage}")
                                node_colors.append(
                                    cluster_palette.get(clabel, "#6b7280")
                                )

                        # Links
                        links = {"source": [], "target": [], "value": [], "color": []}

                        for i in range(len(stage_order) - 1):
                            s1 = stage_order[i]
                            s2 = stage_order[i + 1]

                            m1 = valid_m[valid_m["Period"] == s1]
                            m2 = valid_m[valid_m["Period"] == s2]

                            merged = m1.merge(
                                m2,
                                on="BARN_CLEAN",
                                suffixes=("_old", "_new"),
                            )

                            for _, row in merged.iterrows():
                                old_lbl = row["Cluster_Label_old"]
                                new_lbl = row["Cluster_Label_new"]

                                if pd.isna(old_lbl) or pd.isna(new_lbl):
                                    continue

                                src = label_to_id.get((s1, old_lbl))
                                tgt = label_to_id.get((s2, new_lbl))

                                if src is None or tgt is None:
                                    continue

                                links["source"].append(src)
                                links["target"].append(tgt)
                                links["value"].append(1)
                                links["color"].append(
                                    hex_to_rgba(
                                        cluster_palette.get(old_lbl, "#6b7280")
                                    )
                                )

                        if len(links["source"]) == 0:
                            st.warning("Not enough data to draw movement Sankey.")
                        else:
                            fig_sankey = go.Figure(
                                go.Sankey(
                                    node=dict(label=node_labels, color=node_colors),
                                    link=dict(
                                        source=links["source"],
                                        target=links["target"],
                                        value=links["value"],
                                        color=links["color"],
                                    ),
                                )
                            )
                            fig_sankey.update_layout(
                                title="Barn Cluster Movement Across Growth Stages",
                                height=800,
                                template=plotly_template,
                            )
                            st.plotly_chart(fig_sankey, width="stretch")

                        st.info(
                            """
                            **How to read this view:**
                            - Each node = a *cluster label* at a *growth stage*.  
                            - Flows show how barns move between performance bands as pigs age.  
                            - Look for:
                              - Barns staying in **High Performance → High Performance** (strong SOPs).  
                              - Barns drifting from **High → Moderate → Low** (deterioration, needs action).  
                              - Barns improving from **Low → Moderate → High** (successful interventions).
                            """
                        )
                # Developer diagnostics (safe)
                if dev_mode:
                    safe_run(
                        "K-Means Diagnostics (Barn performance)",
                        lambda: show_k_diagnostics(X_scaled),
                    )
    # ============================================================
    # MODE 2 — INDIVIDUAL GROWTH CLUSTERING
    # ============================================================
    elif mode == "Individual growth":
        st.subheader("🐖 Individual Growth Segmentation")

        req = [
            "RAISE_DAY",
            "ESTIMATED_WEIGHT",
            "FEED_INTAKE_ACTUAL",
            "RESPIRATORY_PERCENT",
            "TEMP_DIFF",
        ]
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.warning(f"Missing required fields for growth segmentation: {missing}")
        else:
            df_g = df.dropna(subset=req).copy()
            df_g["ADG"] = df_g["ESTIMATED_WEIGHT"] / df_g["RAISE_DAY"].replace(0, np.nan)
            df_g = df_g.dropna(subset=["ADG"])

            features = [
                "RAISE_DAY",
                "ESTIMATED_WEIGHT",
                "ADG",
                "FEED_INTAKE_ACTUAL",
                "RESPIRATORY_PERCENT",
                "TEMP_DIFF",
            ]

            X = df_g[features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k = st.slider("Number of clusters (k)", 2, 8, 4)

            # Developer diagnostics
            if dev_mode:
                safe_run(
                    "K-Means Diagnostics (Individual growth)",
                    lambda: show_k_diagnostics(X_scaled),
                )

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_g["Cluster"] = kmeans.fit_predict(X_scaled)

            centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)

            growth_name_pool = [
                "Fast Growers",
                "Above-Average Growers",
                "Moderate Growers",
                "Slow Growers",
                "Critical Slow Growers",
            ]
            cluster_name_map = name_clusters_by_metric(
                centers, "ADG", growth_name_pool
            )

            df_g["Cluster_Label"] = df_g["Cluster"].map(cluster_name_map)

            # Scatter
            with st.expander("📈 Age vs Weight by Growth Cluster", expanded=False):
                sample_df = df_g.sample(min(3000, len(df_g)), random_state=42)
                fig_grow = px.scatter(
                    sample_df,
                    x="RAISE_DAY",
                    y="ESTIMATED_WEIGHT",
                    color="Cluster_Label",
                    opacity=0.6,
                    template=plotly_template,
                    labels={...},
                    color_discrete_map=cluster_color_map,
                )

                fig_grow.update_layout(height=480)
                st.plotly_chart(fig_grow, width="stretch")

            # Summary
            with st.expander("📋 Growth Cluster Summary", expanded=False):
                sum_g = df_g.groupby("Cluster")[features].mean().reset_index()
                sum_g["Cluster_Label"] = sum_g["Cluster"].map(cluster_name_map)
                st.dataframe(sum_g, width="stretch")

            # Insights
            with st.expander(
                "🧠 Growth Mode Insights (Fast-Grower vs Slow-Grower Profiles)",
                expanded=False,
            ):
                fast = sum_g.sort_values("ADG", ascending=False).iloc[0]
                slow = sum_g.sort_values("ADG", ascending=True).iloc[0]

                fast_label = fast["Cluster_Label"]
                slow_label = slow["Cluster_Label"]

                barns_fast = df_g[df_g["Cluster_Label"] == fast_label]["BARN_CLEAN"].unique()
                barns_slow = df_g[df_g["Cluster_Label"] == slow_label]["BARN_CLEAN"].unique()

                st.markdown(
                    f"""
                    - **Fastest-growing cluster:** `{fast_label}`  
                      • High ADG at younger ages  
                      • Typically better feed intake and lower stress  
                      • Barns: **{", ".join(barns_fast.astype(str))}**  

                    - **Slowest-growing cluster:** `{slow_label}`  
                      • Low ADG and often higher respiratory % or temperature variation  
                      • Barns: **{", ".join(barns_slow.astype(str))}**  

                    👉 **Action:**  
                    - Copy feed & environment practices from **fast growers**.  
                    - Investigate **slow growers** for sub-optimal feed, crowding, or disease.
                    """
                )

    # ============================================================
    # MODE 3 — HEALTH PATTERN CLUSTERING
    # ============================================================
    else:
        st.subheader("🩺 Health Pattern Segmentation")

        req = [
            "RESPIRATORY_PERCENT",
            "HEAT_STRESS",
            "TEMP_DIFF",
            "REASON_SCORE",
            "ESTIMATED_WEIGHT",
            "RAISE_DAY",
        ]
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.warning(f"Missing required health fields: {missing}")
        else:
            df_h = df.dropna(subset=req).copy()
            df_h["ADG"] = df_h["ESTIMATED_WEIGHT"] / df_h["RAISE_DAY"].replace(0, np.nan)
            df_h = df_h.dropna(subset=["ADG"])

            features = [
                "RESPIRATORY_PERCENT",
                "HEAT_STRESS",
                "TEMP_DIFF",
                "REASON_SCORE",
                "ADG",
            ]

            X = df_h[features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k = st.slider("Number of clusters (k)", 2, 6, 3)

            # Developer diagnostics
            if dev_mode:
                safe_run(
                    "K-Means Diagnostics (Health patterns)",
                    lambda: show_k_diagnostics(X_scaled),
                )

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_h["Cluster"] = kmeans.fit_predict(X_scaled)

            centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)

            health_name_pool = [
                "High-Risk Pens",
                "Elevated-Risk Pens",
                "Moderate-Risk Pens",
                "Low-Risk Pens",
                "Very Low-Risk Pens",
            ]
            cluster_name_map = name_clusters_by_metric(
                centers, "RESPIRATORY_PERCENT", health_name_pool
            )

            df_h["Cluster_Label"] = df_h["Cluster"].map(cluster_name_map)

            # Scatter
            with st.expander("🌡️ Respiratory % vs Temperature Swing", expanded=False):
                sample_h = df_h.sample(min(3000, len(df_h)), random_state=42)
                fig_health = px.scatter(
                    sample_h,
                    x="TEMP_DIFF",
                    y="RESPIRATORY_PERCENT",
                    color="Cluster_Label",
                    opacity=0.7,
                    template=plotly_template,
                    labels={...},
                    color_discrete_map=cluster_color_map,
                )

                fig_health.update_layout(height=480)
                st.plotly_chart(fig_health, width="stretch")

            # Summary
            with st.expander("📋 Health Cluster Summary", expanded=False):
                sum_h = df_h.groupby("Cluster")[features].mean().reset_index()
                sum_h["Cluster_Label"] = sum_h["Cluster"].map(cluster_name_map)
                st.dataframe(sum_h, width="stretch")

            # Insights
            with st.expander(
                "🧠 Health Mode Insights (High-Risk vs Low-Risk Pens)", expanded=False
            ):
                high = sum_h.sort_values(
                    "RESPIRATORY_PERCENT", ascending=False
                ).iloc[0]
                low = sum_h.sort_values(
                    "RESPIRATORY_PERCENT", ascending=True
                ).iloc[0]

                high_label = high["Cluster_Label"]
                low_label = low["Cluster_Label"]

                barns_high = df_h[df_h["Cluster_Label"] == high_label]["BARN_CLEAN"].unique()
                barns_low = df_h[df_h["Cluster_Label"] == low_label]["BARN_CLEAN"].unique()

                st.markdown(
                    f"""
                    - **Highest-risk cluster:** `{high_label}`  
                      • High respiratory %  
                      • Often combined with heat stress and high TEMP_DIFF  
                      • Barns: **{", ".join(barns_high.astype(str))}**  

                    - **Healthiest cluster:** `{low_label}`  
                      • Low respiratory symptoms  
                      • Stable temperatures  
                      • Better ADG  
                      • Barns: **{", ".join(barns_low.astype(str))}**  

                    👉 **Action:**  
                    - Prioritise vaccination, ventilation and isolation in **high-risk pens**.  
                    - Use **low-risk pens** as *gold-standard* protocols.
                    """
                )


# ================================================================
# PAGE 9 — DATA DICTIONARY
# ================================================================
elif page == "📚 Data Dictionary":
    st.markdown("## 📚 Data Dictionary")
    
    st.markdown(
        """
        This page consolidates **all important fields** used in the dashboard:
        - Raw farm operations (batches, barns, population, feed)
        - Growth & performance metrics (weight, ADG, feed efficiency)
        - Health & climate fields (respiratory %, heat stress, temperature)
        - Barn-level KPIs and ML features (unified scores, clusters, encodings)

        Where available, definitions come from your external Excel data dictionary.  
        Remaining fields are auto-documented based on their role in the pipeline.
        """
    )

    # Build enhanced dictionary from df + barn_perf + Excel
    enhanced_dd = build_enhanced_data_dictionary(
        df,
        barn_perf if "barn_perf" in globals() else None,
        dict_path="Data_Dictionary (Sharing).xlsx",
    )

    if enhanced_dd is None or enhanced_dd.empty:
        st.error("❌ Unable to build data dictionary. Please check that the data and Excel file are available.")
        st.stop()

    # Show quick overview
    st.markdown(
        f"**Total fields documented:** `{len(enhanced_dd)}`  "
        f"• **Categories:** `{len(enhanced_dd['Category'].unique())}`"
    )
    st.markdown("---")

    # Sectioned accordion by Category
    for category in enhanced_dd["Category"].unique():
        subset = enhanced_dd[enhanced_dd["Category"] == category]

        with st.expander(f"📂 {category}  ({len(subset)})", expanded=(category == "Raw Farm Operations")):
            # Reorder columns for nicer view
            view = subset[["Field", "Description", "Datatype", "Table / Source"]]
            safe_df = enhanced_dd.copy()
            safe_df = safe_df.astype({col: str for col in safe_df.columns})
            st.dataframe(safe_df, width="stretch")

            #st.dataframe(view, width="stretch")

    st.markdown("---")
    st.markdown(
        """
        ✅ **Tip for management:**  
        Use this page when new users ask *“What does this field mean?”* or  
        *“Where does this KPI come from?”* – it explains both **business meaning**  
        and **technical origin** at a glance.
        """
    )

# ================================================================
# PAGE — DATA LINEAGE
# ================================================================
elif page == "🔗 Data Lineage":
    st.markdown("## 🔗 Data Lineage – From Raw Data to Decisions")
    
    st.markdown(
        """
        This view explains **how data flows** from raw CSV files  
        → engineered features → barn performance table → ML models → dashboard KPIs.

        It answers:
        - *“Where do these KPIs come from?”*  
        - *“How are health & risk indicators built?”*  
        - *“What’s the link between raw farm data and ML predictions?”*
        """
    )

    # ------------------------------------------------------------
    # 1) High-level lineage diagram (Graphviz)
    # ------------------------------------------------------------
    try:
        dot = r"""
        digraph G {
            rankdir=LR;
            node [shape=box, style="filled,rounded", fontsize=11,
                  color="#0f172a", fontcolor="white"];

            raw     [label="Raw Source Data\n(farm_data_sample.csv & others)"];
            feature [label="Feature Engineering\n(ADG, FEED_X_AGE,\nTEMP_DIFF, HEAT_STRESS,\nRESPIRATORY flags, etc.)"];
            barn    [label="Barn Performance Table\n(barn_perf)"];
            ml      [label="ML Models\nXGBoost Weight Predictor\nK-Means Cluster Labels"];
            dash    [label="Dashboard Pages\nExecutive Summary, Production Ops,\nK-Means, SHAP, Simulator"];

            raw     -> feature;
            feature -> barn;
            feature -> ml;
            barn    -> ml;
            barn    -> dash;
            ml      -> dash;
        }
        """
        st.graphviz_chart(dot)
    except Exception as e:
        st.warning(f"Graphviz diagram could not be rendered: {e}")

    st.markdown("---")

    # ------------------------------------------------------------
    # 2) Accordion: Detailed layer-by-layer lineage
    # ------------------------------------------------------------

    # Layer 1 – Raw source
    with st.expander("① Raw Source Data (CSV / Excel Inputs)", expanded=False):
        st.markdown(
            """
            **Examples of raw fields (from `farm_data_sample.csv` and related sources):**
            - `FARM_ID`, `BARN` – identify farm and housing/barn
            - `RAISE_DAY`, `RAISE_WEEK` – age of animals in days / weeks
            - `BEGIN_POP`, `ANIMAL_IN`, `DEATH_CULLING_TOTAL` – population & movement
            - `FEED_INTAKE_ACTUAL` – actual feed given per animal
            - `TEMP_DIFF`, `HEAT_STRESS` inputs – temperature variation / heat flags
            - `RESPIRATORY_PERCENT` – % animals with respiratory symptoms

            These columns are mostly stored in the main **`df`** DataFrame.
            """
        )
        st.markdown("**Sample of raw columns in `df`:**")
        st.write(sorted(df.columns.tolist())[:25])

    # Layer 2 – Feature Engineering
    with st.expander("② Feature Engineering (Record-level Features)", expanded=False):
        st.markdown(
            """
            On top of raw fields, additional **derived features** are created inside the app, e.g.:

            - `ADG` – Average Daily Gain = `ESTIMATED_WEIGHT / RAISE_DAY`
            - `FEED_X_AGE` – interaction between feed intake and age
            - `TEMP_DIFF` refinement, `HEAT_STRESS` binary/continuous indicators
            - `RESPIRATORY_PERCENT_FLAG` – flags pens with high respiratory %  
            - Encoded fields like `BARN_TARGET_ENC` (target encoding of barns)

            These engineered features are used for:
            - **Descriptive analytics** (Performance & health views)
            - **Model features** for XGBoost & K-Means
            """
        )

        derived_cols = [c for c in df.columns if any(k in c.upper() for k in ["ADG", "HEAT", "TEMP", "RESP", "ENC", "FEED_X"])]
        if derived_cols:
            st.markdown("**Key engineered columns detected in `df`:**")
            st.write(sorted(derived_cols))
        else:
            st.info("No engineered feature columns were auto-detected (check naming in df).")

    # Layer 3 – Barn Performance Table
    with st.expander("③ Barn Performance Table (`barn_perf`)", expanded=False):
        if barn_perf is None or barn_perf.empty:
            st.warning("`barn_perf` is empty or not available – please check build_barn_performance(df).")
        else:
            st.markdown(
                """
                The **barn performance table** aggregates record-level data (df) by `BARN_CLEAN`:

                - `MEDIAN_ADG` – typical growth rate for each barn  
                - `AVG_WEIGHT` – mean weight per barn  
                - `RESP_RATE_PCT`, `HEAT_RATE` – health & climate exposure metrics  
                - `SAMPLE_STRENGTH_SCORE`, `AGE_COVERAGE_SCORE` – data quality / coverage  
                - `HEALTH_STABILITY_SCORE` – stability of health indicators over time  
                - `UNIFIED_SCORE` – combined performance score used for ranking barns  

                This is the table used for:
                - **Executive Summary KPIs**
                - **Production Operations page**
                - **Barn-level segmentation in K-Means page**
                """
            )
            st.markdown("**Columns in `barn_perf`:**")
            st.write(sorted(barn_perf.columns.tolist()))

    # Layer 4 – ML Models
    with st.expander("④ ML Models & Segmentation", expanded=False):
        st.markdown(
            """
            Two main ML components are used:

            1. **XGBoost Weight Prediction Model**
               - Input features (examples):  
                 `RAISE_DAY`, `FEED_INTAKE_ACTUAL`, `ADG`, `FEED_X_AGE`,  
                 `BARN_TARGET_ENC`, `TEMP_DIFF`, `HEAT_STRESS`, `RESP_SEVERITY`, `REASON_SCORE`
               - Outputs: `PRED_WEIGHT` (predicted weight per pig) used in:
                 - Daily Risk Predictor
                 - Harvest & Revenue Simulator

            2. **K-Means Clustering (Segmentation)**
               - Barn performance clustering (High / Moderate / Low / Critical)
               - Individual growth clustering (Fast / Slow growers)
               - Health pattern clustering (High-risk vs Low-risk pens)

               K-Means outputs:
               - `Cluster` – numeric cluster id
               - `Cluster_Label` – human label (e.g. “High Performance”, “High-Risk Pens”)

            These clusters and predictions drive the **K-Means Segmentation**  
            page and parts of the **Executive Summary** and **Operations** pages.
            """
        )

    # Layer 5 – Dashboard Consumption
    with st.expander("⑤ Dashboard KPIs & Pages", expanded=False):
        st.markdown(
            """
            Finally, all upstream layers feed into the dashboard:

            - **📌 Executive Summary**
              - Uses `barn_perf` (UNIFIED_SCORE, MEDIAN_ADG, AVG_WEIGHT, RESP_RATE_PCT, HEAT_RATE)
              - Combines with ML outputs for risk & heat stress insights

            - **🏭 Production Operations**
              - Deep-dive into `barn_perf` by barn
              - Radar charts, LOESS growth curves, ADG distributions

            - **🧬 K-Means Segmentation**
              - Uses engineered features and ML cluster labels
              - Shows cluster-level patterns and movement across growth stages

            - **🧠 Explainable AI (SHAP)**
              - Uses model inputs & SHAP values to explain XGBoost predictions

            - **💰 Harvest & Revenue Simulator**
              - Uses the XGBoost model to simulate future weight & profitability
            """
        )

    st.markdown("---")
    st.success(
        "This lineage view, together with the 📚 Data Dictionary page, gives both "
        "management and auditors a transparent trail from **raw data** to **decisions**."
    )

  # ================================================================
# PAGE — AI / ML Journey Roadmap
# ================================================================
elif page == "🤖 AI/ML Journey":

    st.markdown("## 🤖 AI / ML Journey – From Descriptive to Predictive Farm Intelligence")

    st.markdown(
        """
        Modern livestock operations evolve through **four maturity stages** in analytics.
        This roadmap shows how your Farm ML Dashboard supports each step.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Stage 1 — Descriptive Analytics
    # ------------------------------------------------------------
    st.markdown("### 🟦 Stage 1: **Descriptive Analytics**")
    st.markdown(
        """
        **Answering: _What happened?_**  
        The system summarizes historical barn & animal performance:
        
        - KPI dashboards (ADG, Average Weight, Respiratory Exposure, Heat Stress)
        - Barn ranking (best & worst performers)
        - Unified barn health/performance scoring
        - Correlation & distribution analytics  

        **Value:**  
        - Improves visibility  
        - Identifies underperforming barns  
        - Highlights operational gaps  
        """
    )

    st.info("📊 *Your dashboard already includes a full descriptive analytics suite.*")

    st.markdown("---")

    # ------------------------------------------------------------
    # Stage 2 — Diagnostic Analytics
    # ------------------------------------------------------------
    st.markdown("### 🟩 Stage 2: **Diagnostic Analytics**")
    st.markdown(
        """
        **Answering: _Why did it happen?_**  

        This layer uncovers **root causes of poor performance or health issues**, using:
        - K-Means segmentation (growth clusters, risk clusters)
        - Radar charts showing multi-attribute comparison
        - Heatmaps identifying feature interactions (age, temperature swing, feed intake, etc.)
        - Barn stability & record coverage scores

        **Value:**  
        - Provides causal understanding  
        - Helps target interventions  
        - Supports SOP improvements  
        """
    )

    st.info("🧬 *Your Growth & Health Segmentation module delivers diagnostic insights.*")

    st.markdown("---")

    # ------------------------------------------------------------
    # Stage 3 — Predictive Analytics
    # ------------------------------------------------------------
    st.markdown("### 🟨 Stage 3: **Predictive Analytics**")
    st.markdown(
        """
        **Answering: _What will happen?_**  

        Machine Learning models forecast key outcomes:
        - **Weight prediction model** (XGBoost)
        - Future ADG estimation
        - Risk scoring (respiratory % and heat stress predictors)
        - Daily Risk Predictor tool (Model V2)

        **Value:**  
        - Enables early-warning alerts  
        - Anticipates upcoming weight deviations  
        - Highlights barns requiring proactive checks  
        """
    )

    st.info("⚠️ *Your dashboard includes ML models for weight forecasting & risk estimation.*")

    st.markdown("---")

    # ------------------------------------------------------------
    # Stage 4 — Prescriptive Analytics (Farm AI)
    # ------------------------------------------------------------
    st.markdown("### 🟧 Stage 4: **Prescriptive Analytics (Farm AI)**")
    st.markdown(
        """
        **Answering: _What should we do about it?_**  

        This is the **future state** where the system recommends optimal actions automatically:

        - Feed optimization recommendations  
        - AI-assisted health intervention scheduling  
        - Early-warning barn alerts  
        - Automatic SOP deviations detection  
        - Resource allocation (manpower, ventilation, cleaning cycles)

        **Value:**  
        - Drives consistent performance  
        - Reduces mortality  
        - Improves feed ROI  
        - Minimizes disease outbreaks  
        """
    )

    st.success("🤖 *Your ‘Ask the Farm AI’ page begins this prescriptive capability.*")

    st.markdown("---")

    # ------------------------------------------------------------
    # Full AI Journey Diagram (ASCII)
    # ------------------------------------------------------------
    st.markdown("### 🗺️ End-to-End Farm AI Journey Map")
    st.markdown(
        """
        ```
        [ Descriptive ]  →  [ Diagnostic ]  →  [ Predictive ]  →  [ Prescriptive (AI) ]
            KPIs            Root Cause            Forecasting           Recommendations
            Trends          Segmentation          Risk Models           Action Engine
            Ranking         Heatmaps              Future Outcomes       Auto Alerts
        ```
        """
    )

    st.markdown("---")

    st.success(
        "🚀 Your dashboard is already at **Stage 3 (Predictive)** with foundations for Stage 4 (AI-driven decisions)."
    )
    

   
    st.markdown("---")

    st.markdown("### 🧪 Model Development Journey ")
    st.caption(
        "How the weight prediction model evolved from baseline Linear Regression "
        "to the final XGBoost model (R² ≈ 0.821) used in production."
    )

    render_model_dev_journey_table()

import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# Initialize connection
conn = st.connection("gsheets", type=GSheetsConnection)

st.divider()
#st.header("🐄 Log New Health Data")
#st.markdown("## <h2 style='color: #31333F;'>🐄 Log New Health Data</h2>", unsafe_allow_html=True)
# Custom CSS to force label visibility
st.markdown("""
    <style>
    /* Force form labels to be dark and bold */
    .stWidgetLabel p {
        color: #1A1A1A !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }
    /* Force header color */
    h2 {
        color: #1A1A1A !important;
    }
    /* Ensure the input text itself is dark */
    input {
        color: #1A1A1A !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Now your existing form code...
with st.form("health_entry_form"):
    st.header("🐄 Log New Health Data")
    # ... rest of your form inputs
    col1, col2 = st.columns(2)
    
    with col1:
        animal_id = st.text_input("Animal ID (e.g., COW-001)")
        health_score = st.number_input("Health Score", min_value=0, max_value=100, value=75)
    
    with col2:
        date_logged = st.date_input("Date of Observation")
        status = st.selectbox("Status", ["Healthy", "Monitoring", "Treatment Required"])
        
    notes = st.text_area("Additional Notes")
    
    submit = st.form_submit_button("Save to Google Sheets")

# Logic to append data
if submit:
    if animal_id:
        # 1. Read existing data from your sheet
        existing_data = conn.read(worksheet="Sheet1")
        
        # 2. Create a DataFrame for the new entry
        new_row = pd.DataFrame([{
            "Date": date_logged.strftime("%Y-%m-%d"),
            "Animal_ID": animal_id,
            "Health_Score": health_score,
            "Status": status,
            "Notes": notes
        }])
        
        # 3. Concatenate and update the sheet
        updated_df = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(worksheet="Sheet1", data=updated_df)
        
        st.success(f"Successfully logged data for {animal_id}!")
        # Clear cache so the dashboard updates with the new data
        st.cache_data.clear()
    else:
        st.warning("Please enter an Animal ID before saving.")



