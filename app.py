import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecasting_model import train_model, train_linear_regression, train_arima_model, train_sarima_model
from evaluation import evaluate_model, evaluate_arima_model
from metadata_utils import get_dept_name

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

# Custom CSS for a more "premium" feel
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Retail Performance & Forecasting Dashboard")

# Load data
@st.cache_data
def get_data():
    df = pd.read_csv("retail_sales_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = get_data()

# --- Sidebar Configuration ---
st.sidebar.header("🛠️ Dashboard Controls")

store = st.sidebar.selectbox("Select Store Location", sorted(df["Store"].unique()))

# Realistic Department Selection: Show Names instead of IDs
all_depts_in_store = sorted(df[df["Store"] == store]["Dept"].unique())
dept_options = {get_dept_name(d): d for d in all_depts_in_store}
selected_dept_name = st.sidebar.selectbox("Select Department / Category", options=list(dept_options.keys()))
dept = dept_options[selected_dept_name]

date_range = st.sidebar.date_input(
    "Analysis Timeframe",
    [df["Date"].min(), df["Date"].max()]
)

# Filtering logic
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["Store"] == store) &
        (df["Dept"] == dept) &
        (df["Date"] >= pd.to_datetime(start_date)) &
        (df["Date"] <= pd.to_datetime(end_date))
    ]
else:
    filtered_df = df[(df["Store"] == store) & (df["Dept"] == dept)]

# --- Top Row: Key Performance Indicators (KPIs) ---
st.header(f"🏪 Performance Overview: Store {store}")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_store_sales = df[df["Store"] == store]["Weekly_Sales"].sum()
avg_weekly_sales = df[(df["Store"] == store) & (df["Dept"] == dept)]["Weekly_Sales"].mean()
max_spike = df[(df["Store"] == store) & (df["Dept"] == dept)]["Weekly_Sales"].max()
holiday_impact = df[(df["Store"] == store) & (df["IsHoliday"] == 1)]["Weekly_Sales"].mean() / df[(df["Store"] == store) & (df["IsHoliday"] == 0)]["Weekly_Sales"].mean()

with kpi1:
    st.metric("Total Store Revenue", f"${total_store_sales:,.0f}")
with kpi2:
    st.metric(f"{selected_dept_name} Avg", f"${avg_weekly_sales:,.2f}")
with kpi3:
    st.metric("Peak Weekly Sales", f"${max_spike:,.0f}")
with kpi4:
    st.metric("Holiday Sales Lift", f"{holiday_impact:.1f}x")

st.divider()

# --- Main Analysis Area ---
st.subheader(f"📅 Sales Trends & Forecast for {selected_dept_name}")

# Filter out markdowns for clean display
display_df = filtered_df.drop(columns=[f"MarkDown{i}" for i in range(1, 6)], errors="ignore")
with st.expander("🔍 View Raw Weekly Data"):
    st.dataframe(display_df, use_container_width=True)

# Sales trend plot
sales_trend = filtered_df.groupby("Date")["Weekly_Sales"].sum()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sales_trend.index, sales_trend.values, marker='o', linestyle='-', color='#1f77b4', markersize=4)
ax.fill_between(sales_trend.index, 0, sales_trend.values, alpha=0.1, color='#1f77b4')
ax.set_ylabel("Weekly Sales ($)")
ax.set_title(f"Historical Sales Trend: {selected_dept_name}")
st.pyplot(fig)

# --- Forecasting Row ---
st.divider()
col_forecast, col_ml = st.columns([2, 1])

with col_forecast:
    st.subheader("📈 Time-Series Forecasting (SARIMA)")
    if len(sales_trend) > 10:
        arima_fit, ts_data = train_arima_model(filtered_df)
        arima_forecast = arima_fit.forecast(steps=12)

        with st.spinner("Analyzing seasons..."):
            sarima_fit, _ = train_sarima_model(filtered_df)
            sarima_forecast = sarima_fit.forecast(steps=12)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(ts_data.index, ts_data.values, label="Historical", color='#1f77b4')
        
        future_dates = pd.date_range(start=ts_data.index[-1], periods=13, freq="W")[1:]
        ax3.plot(future_dates, arima_forecast, label="ARIMA (No Seasonality)", linestyle='--', color='gray')
        ax3.plot(future_dates, sarima_forecast, label="SARIMA (Holiday-Aware)", linestyle='-', color='#d62728', linewidth=2)
        
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.warning("Insufficient data for forecasting.")

with col_ml:
    st.subheader("🤖 Smart Predictions")
    rf_model, _, _ = train_model(df)
    lr_model, _, _ = train_linear_regression(df)
    
    if not filtered_df.empty:
        sample = filtered_df.iloc[[0]]
        features = ["Store","Dept","Temperature","Fuel_Price","CPI","Unemployment","IsHoliday","Size"]
        X_sample = sample[features]
        
        rf_pred = rf_model.predict(X_sample)[0]
        lr_pred = lr_model.predict(X_sample)[0]
        
        st.metric("Complex Pattern Model (RF)", f"${rf_pred:,.2f}")
        st.metric("Direct Baseline (Linear)", f"${lr_pred:,.2f}")
        st.caption(f"Comparing against actual: ${sample['Weekly_Sales'].values[0]:,.2f}")

# --- Global Analytics ---
st.divider()
st.header("🏆 Regional Performance Rankings")
perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.subheader("Top Stores (Highest Profitability)")
    store_sales = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(store_sales)
    
with perf_col2:
    st.subheader("Top Departments (Volume)")
    dept_sales = df.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False).head(5)
    # Use names for chart labels
    dept_label_map = {d: get_dept_name(d) for d in dept_sales.index}
    dept_sales.index = [f"{dept_label_map[d]}" for d in dept_sales.index]
    st.bar_chart(dept_sales)

with st.expander("ℹ️ Why are there 99 departments?"):
    st.write("""
    In massive retail chains like Walmart, "Departments" are highly granular to track every specific corner of the store. 
    IDs 1-99 cover everything from **Bakery (80)** to **Photo Labs (8)** and even **Fuel Stations (59)**. 
    This granularity is what allows the AI to predict exactly how many heads of lettuce (Produce) vs. how many flat-screen TVs (Electronics) should be on shelves.
    """)