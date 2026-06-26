import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Fuel Analytics Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Design Tokens ───────────────────────
BG          = "#0D1117"
SURFACE     = "#161B22"
SURFACE2    = "#1C2333"
BORDER      = "#30363D"
ACCENT      = "#F78166"          # warm coral — fuel/heat
ACCENT2     = "#58A6FF"          # cool blue — prediction/data
ACCENT3     = "#3FB950"          # green — positive
ACCENT_WARN = "#D29922"          # amber — warning
TEXT        = "#E6EDF3"
TEXT_MUTED  = "#8B949E"
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif", size=13),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    margin=dict(t=40, b=40, l=40, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    colorway=[ACCENT, ACCENT2, ACCENT3, ACCENT_WARN, "#BC8CFF", "#39D353"],
)

# ─── Custom CSS ──────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* Root */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {BG} !important;
      color: {TEXT};
      font-family: 'Inter', sans-serif;
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}

  /* Typography */
  h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }}

  /* Cards */
  .kpi-card {{
      background: {SURFACE};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 20px 24px;
      position: relative;
      overflow: hidden;
  }}
  .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: var(--accent-color, {ACCENT});
      border-radius: 12px 12px 0 0;
  }}
  .kpi-label {{
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {TEXT_MUTED};
      margin-bottom: 8px;
  }}
  .kpi-value {{
      font-size: 32px;
      font-weight: 700;
      color: {TEXT};
      line-height: 1;
      font-variant-numeric: tabular-nums;
  }}
  .kpi-delta {{
      font-size: 12px;
      margin-top: 6px;
      color: {TEXT_MUTED};
  }}
  .kpi-delta.pos {{ color: {ACCENT3}; }}
  .kpi-delta.neg {{ color: {ACCENT}; }}

  /* Section headers */
  .section-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 32px 0 16px;
      padding-bottom: 10px;
      border-bottom: 1px solid {BORDER};
  }}
  .section-header .icon {{
      font-size: 18px;
  }}
  .section-header h3 {{
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      color: {TEXT};
  }}

  /* Chart container */
  .chart-card {{
      background: {SURFACE};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
  }}

  /* Insight pills */
  .insight-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 16px;
  }}
  .insight-pill {{
      background: {SURFACE2};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 13px;
  }}
  .insight-pill .label {{ color: {TEXT_MUTED}; font-size: 11px; margin-bottom: 4px; }}
  .insight-pill .val {{ color: {TEXT}; font-weight: 600; }}

  /* Forecast box */
  .forecast-box {{
      background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE2} 100%);
      border: 1px solid {ACCENT2};
      border-radius: 12px;
      padding: 24px;
      position: relative;
      overflow: hidden;
  }}
  .forecast-box::after {{
      content: '🔮';
      position: absolute;
      right: 20px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 48px;
      opacity: 0.15;
  }}

  /* Warning / success / info banners */
  .banner {{
      border-radius: 8px;
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 500;
      margin: 8px 0;
      display: flex;
      align-items: center;
      gap: 8px;
  }}
  .banner.success {{ background: rgba(63,185,80,0.12); border: 1px solid rgba(63,185,80,0.3); color: {ACCENT3}; }}
  .banner.warning {{ background: rgba(210,153,34,0.12); border: 1px solid rgba(210,153,34,0.3); color: {ACCENT_WARN}; }}
  .banner.danger  {{ background: rgba(247,129,102,0.12); border: 1px solid rgba(247,129,102,0.3); color: {ACCENT}; }}
  .banner.info    {{ background: rgba(88,166,255,0.12); border: 1px solid rgba(88,166,255,0.3); color: {ACCENT2}; }}

  /* Data table */
  [data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

  /* Hero */
  .hero {{
      background: {SURFACE};
      border: 1px solid {BORDER};
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
  }}
  .hero-title {{
      font-size: 28px;
      font-weight: 700;
      color: {TEXT};
      letter-spacing: -0.03em;
      line-height: 1.1;
  }}
  .hero-subtitle {{
      color: {TEXT_MUTED};
      font-size: 14px;
      margin-top: 4px;
  }}
  .hero-accent {{
      font-size: 52px;
      filter: drop-shadow(0 0 20px {ACCENT}88);
  }}

  /* Mono numbers */
  .mono {{ font-family: 'JetBrains Mono', monospace; }}

  /* Divider */
  hr {{ border-color: {BORDER} !important; margin: 24px 0; }}

  /* Streamlit default overrides */
  .stMetric {{ display: none; }}
  div[data-testid="metric-container"] {{ display: none; }}


  /* Futuristic animations */
  .hero{
      animation: floatHero 6s ease-in-out infinite;
      position:relative;
      overflow:hidden;
  }
  .hero::before{
      content:'';
      position:absolute;
      top:0;left:-120%;
      width:60%;height:100%;
      background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
      animation:shine 5s linear infinite;
  }
  .hero-accent{
      animation:pulseGlow 2.5s ease-in-out infinite;
  }
  .kpi-card,.chart-card,.forecast-box{
      transition:all .28s ease;
  }
  .kpi-card:hover,.chart-card:hover,.forecast-box:hover{
      transform:translateY(-6px) scale(1.01);
      box-shadow:0 0 24px rgba(88,166,255,.22);
      border-color:#58A6FF;
  }
  .section-header{
      animation:fadeUp .8s ease both;
  }
  .forecast-box{
      animation:borderPulse 3s infinite;
  }
  @keyframes fadeUp{
      from{opacity:0;transform:translateY(18px);}
      to{opacity:1;transform:translateY(0);}
  }
  @keyframes pulseGlow{
      0%,100%{transform:scale(1);filter:drop-shadow(0 0 8px #F78166);}
      50%{transform:scale(1.08);filter:drop-shadow(0 0 20px #58A6FF);}
  }
  @keyframes floatHero{
      0%,100%{transform:translateY(0);}
      50%{transform:translateY(-4px);}
  }
  @keyframes shine{
      from{left:-120%;}
      to{left:160%;}
  }
  @keyframes borderPulse{
      0%,100%{box-shadow:0 0 10px rgba(88,166,255,.18);}
      50%{box-shadow:0 0 28px rgba(88,166,255,.35);}
  }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

def kpi_card(label, value, delta=None, delta_label="", accent=ACCENT, unit=""):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {abs(delta):.2f} {delta_label}</div>'
    return f"""
    <div class="kpi-card" style="--accent-color: {accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value mono">{value}<span style="font-size:14px;color:{TEXT_MUTED};font-weight:400"> {unit}</span></div>
      {delta_html}
    </div>"""

def banner(msg, kind="info"):
    icons = {"success": "✅", "warning": "⚠️", "danger": "🔴", "info": "ℹ️"}
    return f'<div class="banner {kind}">{icons[kind]} {msg}</div>'

def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
      <span class="icon">{icon}</span>
      <h3>{title}</h3>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Fuel.csv")
    except FileNotFoundError:
        st.error("⛽ **Fuel.csv not found.** Place it in the same directory as this script.")
        st.stop()

    df.columns = ["Qty", "KM", "Date", "Battery", "Cost"]

    # Fix: actual CSV uses 4-digit year (DD-MM-YYYY), not 2-digit
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True)

    # Sort by date, then flag any row where KM is not higher than the previous.
    # This is the only reliable signal that a date was entered incorrectly.
    df = df.sort_values("Date").reset_index(drop=True)
    df["_date_mismatch"] = df["KM"].diff().fillna(1) <= 0

    # ── Derived fields ──
    df["Distance"]       = df["KM"].diff()
    df["Mileage"]        = df["Distance"] / df["Qty"]
    df["Cost_per_Litre"] = df["Cost"] / df["Qty"]
    df["Days_Between"]   = df["Date"].diff().dt.days
    df["KM_per_Day"]     = df["Distance"] / df["Days_Between"]
    df["Cumulative_Cost"]= df["Cost"].cumsum()
    df["Cumulative_KM"]  = df["Distance"].cumsum()
    df["Year"]           = df["Date"].dt.year
    df["Month"]          = df["Date"].dt.month
    df["Month_Name"]     = df["Date"].dt.strftime("%b %Y")

    # Drop first row (NaN from diff)
    df = df.dropna().reset_index(drop=True)

    # Outlier flag (mileage > 3σ from mean)
    z = (df["Mileage"] - df["Mileage"].mean()) / df["Mileage"].std()
    df["Is_Outlier"] = z.abs() > 3

    # Indian season
    def get_season(m):
        if m in [12, 1, 2]:    return "Winter"
        elif m in [3, 4, 5]:   return "Pre-Summer"
        elif m in [6, 7, 8, 9]:return "Monsoon"
        else:                   return "Post-Monsoon"
    df["Season"] = df["Month"].apply(get_season)

    df["Time_Index"] = np.arange(len(df))
    return df

df = load_data()

# Data quality warnings — only fires if KM goes backwards after sorting by date
if "_date_mismatch" in df.columns and df["_date_mismatch"].any():
    bad_rows = df[df["_date_mismatch"]][["Date", "KM"]]
    km_list = ", ".join(bad_rows["KM"].astype(str).tolist())
    st.warning(f"⚠️ **{len(bad_rows)} row(s) have a date that appears wrong** — the odometer reading goes backwards at KM: {km_list}. The date on that row is likely a typo. Fix it in Fuel.csv and re-deploy.")

# ─────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT};margin-bottom:16px'>⚙️ Filters</h3>", unsafe_allow_html=True)
    years = sorted(df["Year"].unique())
    sel_years = st.multiselect("Years", years, default=years)
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    show_outliers = st.toggle("Show outlier flags", value=True)
    forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30)
    st.markdown("---")
    st.markdown(f"<span style='color:{TEXT_MUTED};font-size:12px'>Last updated: {df['Date'].max().date()}</span>", unsafe_allow_html=True)

# Apply filters
fdf = df[df["Year"].isin(sel_years)].copy()
if len(date_range) == 2:
    fdf = fdf[(fdf["Date"].dt.date >= date_range[0]) & (fdf["Date"].dt.date <= date_range[1])]

# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-title">Fuel Intelligence Dashboard</div>
    <div class="hero-subtitle">
      {len(fdf)} refills &nbsp;·&nbsp;
      {fdf['Date'].min().strftime('%d %b %Y')} → {fdf['Date'].max().strftime('%d %b %Y')} &nbsp;·&nbsp;
      {fdf['Year'].nunique()} year(s) of data
    </div>
  </div>
  <div class="hero-accent">⛽</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────
section("📊", "Key Performance Indicators")

# Compute deltas (first half vs second half)
mid = len(fdf) // 2
h1, h2 = fdf.iloc[:mid], fdf.iloc[mid:]
d_mileage = h2["Mileage"].mean() - h1["Mileage"].mean()
d_cpl     = h2["Cost_per_Litre"].mean() - h1["Cost_per_Litre"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(kpi_card(
        "Total Fuel Consumed", f"{fdf['Qty'].sum():.1f}", unit="L", accent=ACCENT
    ), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card(
        "Total Spend", f"₹{fdf['Cost'].sum():,.0f}", unit="", accent=ACCENT_WARN
    ), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card(
        "Avg Mileage", f"{fdf['Mileage'].mean():.2f}",
        delta=d_mileage, delta_label="km/L vs earlier",
        unit="km/L", accent=ACCENT2
    ), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card(
        "Avg Cost / Litre", f"₹{fdf['Cost_per_Litre'].mean():.2f}",
        delta=d_cpl, delta_label="vs earlier",
        unit="", accent=ACCENT_WARN
    ), unsafe_allow_html=True)
with c5:
    st.markdown(kpi_card(
        "Avg Refill Interval", f"{fdf['Days_Between'].mean():.1f}",
        unit="days", accent=ACCENT3
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────
section("🔮", "Next Refill Forecast")

# ── Rolling-trend exponential smoothing forecast ─────────────────────────────
# Instead of a flat average, we use exponential smoothing so recent refills
# carry more weight, and the rate of change (trend) is tracked separately.

def exp_smooth_with_trend(series, alpha=0.4, beta=0.3):
    """Double exponential smoothing (Holt's method)."""
    s = [series.iloc[0]]
    t = [series.iloc[1] - series.iloc[0] if len(series) > 1 else 0]
    for i in range(1, len(series)):
        s_new = alpha * series.iloc[i] + (1 - alpha) * (s[-1] + t[-1])
        t_new = beta * (s_new - s[-1]) + (1 - beta) * t[-1]
        s.append(s_new)
        t.append(t_new)
    level, trend = s[-1], t[-1]
    return level, trend

# Smooth km/day and distance per tank — these drive the forecast
km_day_level,    km_day_trend    = exp_smooth_with_trend(fdf["KM_per_Day"],  alpha=0.35, beta=0.25)
dist_level,      dist_trend      = exp_smooth_with_trend(fdf["Distance"],    alpha=0.35, beta=0.25)
qty_level,       _               = exp_smooth_with_trend(fdf["Qty"],         alpha=0.35, beta=0.25)
cpl_level,       cpl_trend       = exp_smooth_with_trend(fdf["Cost_per_Litre"], alpha=0.35, beta=0.25)

# Project km/day into the future with trend dampening (φ = 0.9)
phi = 0.9
def project_km_day(h):
    """Damped-trend forecast for km/day h steps ahead."""
    damping = sum(phi**j for j in range(1, h+1))
    return km_day_level + km_day_trend * damping

# Find days until tank empties: integrate projected daily km
cumulative_km = 0.0
days_until = 0
while cumulative_km < dist_level and days_until < 365:
    days_until += 1
    cumulative_km += project_km_day(days_until)

refill_date    = fdf["Date"].iloc[-1] + pd.Timedelta(days=days_until)
predicted_cost = qty_level * cpl_level

# Build forward projection curve
fwd_days   = list(range(1, forecast_days + 1))
fwd_km_day = [project_km_day(h) for h in fwd_days]
fwd_cum_km = np.cumsum(fwd_km_day)
remaining  = np.clip(dist_level - fwd_cum_km, 0, None)
future_dates = pd.date_range(fdf["Date"].iloc[-1], periods=forecast_days + 1, freq="D")[1:]

# Historical km/day with smoothed overlay for the chart
smooth_series = []
s = fdf["KM_per_Day"].iloc[0]
t_val = 0.0
for v in fdf["KM_per_Day"]:
    s_new = 0.35 * v + 0.65 * (s + t_val)
    t_val = 0.3 * (s_new - s) + 0.7 * t_val
    s = s_new
    smooth_series.append(s)
fdf["KM_per_Day_Smooth"] = smooth_series

col1, col2 = st.columns([1.4, 1])

with col1:
    st.markdown(f"""
    <div class="forecast-box">
      <div style="font-size:12px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:16px">
        Forecast — Holt's Damped-Trend Smoothing
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div>
          <div style="color:{TEXT_MUTED};font-size:11px;margin-bottom:4px">NEXT REFILL DATE</div>
          <div style="color:{ACCENT2};font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace">
            {refill_date.strftime('%d %b %Y')}
          </div>
        </div>
        <div>
          <div style="color:{TEXT_MUTED};font-size:11px;margin-bottom:4px">DAYS REMAINING</div>
          <div style="color:{TEXT};font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace">
            ~{days_until} days
          </div>
        </div>
        <div>
          <div style="color:{TEXT_MUTED};font-size:11px;margin-bottom:4px">EXPECTED COST</div>
          <div style="color:{ACCENT_WARN};font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace">
            ₹{predicted_cost:,.0f}
          </div>
        </div>
        <div>
          <div style="color:{TEXT_MUTED};font-size:11px;margin-bottom:4px">SMOOTHED KM/DAY</div>
          <div style="color:{TEXT};font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace">
            {km_day_level:.1f} km
          </div>
        </div>
        <div style="grid-column:span 2">
          <div style="color:{TEXT_MUTED};font-size:11px;margin-bottom:6px">KM/DAY TREND</div>
          <div style="color:{'#3FB950' if km_day_trend >= 0 else '#F78166'};font-size:13px;font-weight:600">
            {'▲ Increasing' if km_day_trend >= 0 else '▼ Decreasing'} usage
            ({km_day_trend:+.2f} km/day per refill interval)
          </div>
          <div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px">
            Fuel price trend: {cpl_trend:+.2f} ₹/L per interval
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with col2:
    fig_fwd = go.Figure()
    fig_fwd.add_trace(go.Scatter(
        x=future_dates, y=remaining,
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.12)",
        line=dict(color=ACCENT2, width=2),
        name="Est. range remaining",
        hovertemplate="%{x|%d %b}: ~%{y:.0f} km left<extra></extra>"
    ))
    fig_fwd.add_hline(y=0, line_color=ACCENT, line_dash="dot",
                      annotation_text="Refill needed", annotation_font_color=ACCENT)
    fig_fwd.update_layout(**PLOTLY_THEME, height=220, title="Projected Tank Range",
                          yaxis_title="Est. km remaining", showlegend=False)
    st.plotly_chart(fig_fwd, use_container_width=True)

# KM/Day history with smoothed overlay
fig_kmd = go.Figure()
fig_kmd.add_trace(go.Scatter(
    x=fdf["Date"], y=fdf["KM_per_Day"],
    mode="markers", name="Actual km/day",
    marker=dict(size=6, color=ACCENT2, opacity=0.5),
    hovertemplate="%{x|%d %b %Y}: %{y:.1f} km/day<extra></extra>"
))
fig_kmd.add_trace(go.Scatter(
    x=fdf["Date"], y=fdf["KM_per_Day_Smooth"],
    mode="lines", name="Smoothed trend",
    line=dict(color=ACCENT3, width=2),
    hovertemplate="Smoothed: %{y:.1f} km/day<extra></extra>"
))
# Forward projection
fwd_smooth_dates = [fdf["Date"].iloc[-1]] + list(future_dates[:15])
fwd_smooth_vals  = [km_day_level] + [project_km_day(h) for h in range(1, 16)]
fig_kmd.add_trace(go.Scatter(
    x=fwd_smooth_dates, y=fwd_smooth_vals,
    mode="lines", name="Forecast (damped)",
    line=dict(color=ACCENT_WARN, width=2, dash="dash"),
    hovertemplate="Forecast: %{y:.1f} km/day<extra></extra>"
))
fig_kmd.update_layout(
    **PLOTLY_THEME, height=260,
    title="Daily Usage Rate — History & Forecast",
    yaxis_title="km / day", hovermode="x unified",
)
st.plotly_chart(fig_kmd, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────
# MILEAGE OVER TIME  +  REGRESSION
# ─────────────────────────────────────────
section("📈", "Mileage Trend & Regression")

# ── Multi-feature Ridge regression ──────────────────────────────────────────
# Features: time index, days between refills, cost per litre, season (encoded)
season_dummies = pd.get_dummies(fdf["Season"], prefix="S", drop_first=True).astype(float)
feature_df = pd.concat([
    fdf[["Time_Index", "Days_Between", "Cost_per_Litre"]],
    season_dummies,
], axis=1).fillna(0)

X_multi = feature_df.values
y_mileage = fdf["Mileage"].values

# Ridge with standard scaling — more robust than plain OLS with correlated features
multi_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
multi_model.fit(X_multi, y_mileage)
fdf["Predicted_Mileage"] = multi_model.predict(X_multi)
r2 = r2_score(y_mileage, fdf["Predicted_Mileage"])

# Cross-validated R² (leave-one-out proxy with cv=min(5,n))
cv_k = min(5, len(fdf))
cv_scores = cross_val_score(multi_model, X_multi, y_mileage, cv=cv_k, scoring="r2")
cv_r2 = cv_scores.mean()

# Pure time-based slope for trend direction (simple linear on index)
linear_model = LinearRegression()
linear_model.fit(fdf[["Time_Index"]], y_mileage)
slope = linear_model.coef_[0]

# Feature importance (Ridge coefficients after scaling)
ridge_step = multi_model.named_steps["ridge"]
scaler_step = multi_model.named_steps["standardscaler"]
feat_names = feature_df.columns.tolist()
feat_importance = pd.DataFrame({
    "Feature": feat_names,
    "Coefficient": ridge_step.coef_,
    "Abs": np.abs(ridge_step.coef_),
}).sort_values("Abs", ascending=True)

fig_mileage = go.Figure()
# Actual mileage with markers
fig_mileage.add_trace(go.Scatter(
    x=fdf["Date"], y=fdf["Mileage"],
    mode="lines+markers",
    name="Actual Mileage",
    line=dict(color=ACCENT2, width=2),
    marker=dict(
        size=8, color=np.where(fdf["Is_Outlier"], ACCENT, ACCENT2),
        symbol=np.where(fdf["Is_Outlier"], "diamond", "circle"),
        line=dict(width=1, color=BG)
    ),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Mileage: %{y:.2f} km/L<extra></extra>"
))
# Multi-feature model predicted line
fig_mileage.add_trace(go.Scatter(
    x=fdf["Date"], y=fdf["Predicted_Mileage"],
    mode="lines", name=f"Model fit (R²={r2:.2f})",
    line=dict(color=ACCENT, width=2, dash="dot"),
    hovertemplate="Predicted: %{y:.2f} km/L<extra></extra>"
))
# Outlier annotation
outliers = fdf[fdf["Is_Outlier"]]
if not outliers.empty and show_outliers:
    fig_mileage.add_trace(go.Scatter(
        x=outliers["Date"], y=outliers["Mileage"],
        mode="markers", name="Outlier (>3σ)",
        marker=dict(size=14, color=ACCENT, symbol="diamond",
                    line=dict(width=2, color=TEXT)),
        hovertemplate="<b>Outlier</b><br>%{x|%d %b %Y}: %{y:.2f} km/L<extra></extra>"
    ))

fig_mileage.update_layout(
    **PLOTLY_THEME,
    height=340,
    yaxis_title="Mileage (km/L)",
    title="Actual vs Multi-Feature Predicted Mileage",
    hovermode="x unified",
)
st.plotly_chart(fig_mileage, use_container_width=True)

# ── Feature importance bar ───────────────────────────────────────────────────
fig_feat = go.Figure(go.Bar(
    x=feat_importance["Coefficient"],
    y=feat_importance["Feature"],
    orientation="h",
    marker=dict(
        color=feat_importance["Coefficient"].apply(
            lambda v: ACCENT3 if v > 0 else ACCENT
        ),
        line=dict(width=0),
    ),
    hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>",
))
fig_feat.update_layout(
    **PLOTLY_THEME, height=240,
    title=f"Feature Impact on Mileage (Ridge | CV R²={cv_r2:.2f})",
    xaxis_title="Coefficient (scaled)",
    showlegend=False,
)
st.plotly_chart(fig_feat, use_container_width=True)

# Diagnostics
col_a, col_b = st.columns(2)
with col_a:
    if slope > 0.02:
        st.markdown(banner(f"Mileage improving: +{slope:.3f} km/L per refill · Model R²={r2:.2f} · CV R²={cv_r2:.2f}", "success"), unsafe_allow_html=True)
    elif slope < -0.02:
        st.markdown(banner(f"Mileage declining: {slope:.3f} km/L per refill · Model R²={r2:.2f} · CV R²={cv_r2:.2f}. Consider servicing.", "danger"), unsafe_allow_html=True)
    else:
        st.markdown(banner(f"Mileage stable (slope={slope:.3f}) · Model R²={r2:.2f} · CV R²={cv_r2:.2f}", "info"), unsafe_allow_html=True)
with col_b:
    cv_metric = fdf["Mileage"].std() / fdf["Mileage"].mean()
    if cv_metric < 0.05:
        st.markdown(banner(f"Highly consistent fuel efficiency (CV={cv_metric:.3f})", "success"), unsafe_allow_html=True)
    elif cv_metric < 0.10:
        st.markdown(banner(f"Moderately consistent efficiency (CV={cv_metric:.3f})", "warning"), unsafe_allow_html=True)
    else:
        st.markdown(banner(f"High variability in mileage (CV={cv_metric:.3f}) — check driving patterns or outliers", "danger"), unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
# COST ANALYSIS
# ─────────────────────────────────────────
section("💰", "Cost Analysis")

col1, col2 = st.columns(2)

with col1:
    # Cost per litre over time
    fig_cpl = go.Figure()
    fig_cpl.add_trace(go.Scatter(
        x=fdf["Date"], y=fdf["Cost_per_Litre"],
        fill="tozeroy",
        fillcolor=f"rgba(247,129,102,0.12)",
        line=dict(color=ACCENT, width=2),
        name="₹/Litre",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:.2f}/L<extra></extra>"
    ))
    fig_cpl.update_layout(**PLOTLY_THEME, height=280,
                          yaxis_title="Cost per Litre (₹)", title="Fuel Price Over Time")
    st.plotly_chart(fig_cpl, use_container_width=True)

with col2:
    # Yearly spend
    yearly = fdf.groupby("Year").agg(
        Total_Cost=("Cost", "sum"),
        Refills=("Cost", "count"),
        Avg_Mileage=("Mileage", "mean"),
    ).reset_index()

    fig_yr = go.Figure()
    fig_yr.add_trace(go.Bar(
        x=yearly["Year"].astype(str), y=yearly["Total_Cost"],
        marker=dict(
            color=yearly["Total_Cost"],
            colorscale=[[0, SURFACE2], [1, ACCENT_WARN]],
            line=dict(width=0),
        ),
        text=[f"₹{v:,.0f}" for v in yearly["Total_Cost"]],
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{x}</b><br>Spent: ₹%{y:,.0f}<extra></extra>"
    ))
    fig_yr.update_layout(**PLOTLY_THEME, height=280,
                         yaxis_title="Total Spend (₹)", title="Yearly Fuel Expenditure",
                         showlegend=False)
    st.plotly_chart(fig_yr, use_container_width=True)

# Cumulative spend
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=fdf["Date"], y=fdf["Cumulative_Cost"],
    mode="lines",
    fill="tozeroy",
    fillcolor=f"rgba(210,153,34,0.10)",
    line=dict(color=ACCENT_WARN, width=2),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Cumulative: ₹%{y:,.0f}<extra></extra>",
    name="Cumulative Spend"
))
fig_cum.update_layout(**PLOTLY_THEME, height=240,
                      yaxis_title="Cumulative Spend (₹)",
                      title="Running Total Fuel Expenditure")
st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────
# SEASONAL ANALYSIS
# ─────────────────────────────────────────
section("🌦", "Seasonal Performance")

season_order = ["Winter", "Pre-Summer", "Monsoon", "Post-Monsoon"]
seasonal = fdf.groupby("Season").agg(
    Avg_Mileage=("Mileage", "mean"),
    Avg_Days=("Days_Between", "mean"),
    Avg_CPL=("Cost_per_Litre", "mean"),
    Count=("Mileage", "count"),
).reindex([s for s in season_order if s in fdf["Season"].unique()]).reset_index()

season_colors = {
    "Winter":       "#58A6FF",
    "Pre-Summer":   "#F78166",
    "Monsoon":      "#3FB950",
    "Post-Monsoon": "#D29922",
}

col1, col2, col3 = st.columns(3)
with col1:
    fig_s1 = px.bar(
        seasonal, x="Season", y="Avg_Mileage",
        color="Season", color_discrete_map=season_colors,
        text=seasonal["Avg_Mileage"].round(2),
    )
    fig_s1.update_traces(textposition="outside", textfont_color=TEXT)
    fig_s1.update_layout(**PLOTLY_THEME, title="Avg Mileage by Season", showlegend=False,
                         height=280, yaxis_title="km/L")
    st.plotly_chart(fig_s1, use_container_width=True)

with col2:
    fig_s2 = px.bar(
        seasonal, x="Season", y="Avg_Days",
        color="Season", color_discrete_map=season_colors,
        text=seasonal["Avg_Days"].round(1),
    )
    fig_s2.update_traces(textposition="outside", textfont_color=TEXT)
    fig_s2.update_layout(**PLOTLY_THEME, title="Avg Days Between Refills", showlegend=False,
                         height=280, yaxis_title="Days")
    st.plotly_chart(fig_s2, use_container_width=True)

with col3:
    fig_s3 = px.bar(
        seasonal, x="Season", y="Avg_CPL",
        color="Season", color_discrete_map=season_colors,
        text=seasonal["Avg_CPL"].round(2),
    )
    fig_s3.update_traces(textposition="outside", textfont_color=TEXT)
    fig_s3.update_layout(**PLOTLY_THEME, title="Avg Fuel Price by Season", showlegend=False,
                         height=280, yaxis_title="₹/Litre")
    st.plotly_chart(fig_s3, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────
section("🔗", "Correlation Matrix")

corr_cols = ["Qty", "Distance", "Mileage", "Cost", "Cost_per_Litre", "Days_Between", "KM_per_Day"]
corr = fdf[corr_cols].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns, y=corr.index,
    colorscale=[
        [0.0, "#1a3a5c"],
        [0.5, SURFACE2],
        [1.0, "#7f1a0f"],
    ],
    zmid=0,
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=11, color=TEXT),
    hovertemplate="<b>%{x} × %{y}</b><br>r = %{z:.3f}<extra></extra>",
))
fig_corr.update_layout(**PLOTLY_THEME, height=380, title="Pearson Correlation — All Metrics")
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────
with st.expander("📄 View full processed dataset"):
    display_cols = ["Date", "KM", "Distance", "Qty", "Mileage",
                    "Cost", "Cost_per_Litre", "Days_Between", "KM_per_Day",
                    "Season", "Year", "Is_Outlier"]
    display_cols = [c for c in display_cols if c in fdf.columns]
    st.dataframe(
        fdf[display_cols].style.format({
            "Mileage": "{:.2f}",
            "Cost_per_Litre": "₹{:.2f}",
            "KM_per_Day": "{:.2f}",
            "Distance": "{:.1f}",
            "Cost": "₹{:,.0f}",
        }).background_gradient(subset=["Mileage"], cmap="RdYlGn"),
        use_container_width=True,
        height=320,
    )

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:32px 0 16px;color:{TEXT_MUTED};font-size:12px">
  Fuel Intelligence Dashboard &nbsp;·&nbsp;
  Data: {len(fdf)} refills from {fdf['Date'].min().strftime('%b %Y')} to {fdf['Date'].max().strftime('%b %Y')} &nbsp;·&nbsp;
  Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
