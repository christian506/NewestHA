import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Page setup
st.set_page_config(page_title="Dysentery in Lebanon Dashboard", layout="wide")
st.title("📊 Dysentery Dashboard – Lebanon")
st.markdown("Interactive dashboard to explore trends, patterns, and forecasts of dysentery cases using public health data.")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("Dysentery_data.csv")
    df = df[df["Number of cases"].notna()].copy()
    df["Governorate"] = df["refArea"].str.extract(r'/resource/([^/]+)$')[0].str.replace('_', ' ')
    df["Year"] = df["refPeriod"].str.extract(r'(\d{4})')
    df["Month"] = df["refPeriod"].str.extract(r'(\d{2})-\d{4}')[0]
    df["Date"] = pd.to_datetime(df["Year"] + "-" + df["Month"], errors='coerce')
    return df[["Date", "Year", "Month", "Governorate", "Number of cases"]].dropna()

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
years = sorted(df["Year"].unique())
governorates = sorted(df["Governorate"].unique())

selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
selected_govs = st.sidebar.multiselect("Select Governorate(s)", governorates, default=governorates)

df_filtered = df[df["Year"].isin(selected_years) & df["Governorate"].isin(selected_govs)]

# ===================
# DASHBOARD START
# ===================

# Top KPIs
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("🦠 Total Cases", f"{int(df_filtered['Number of cases'].sum())}")
kpi2.metric("📅 Months Covered", df_filtered["Date"].nunique())
kpi3.metric("📍 Governorates", df_filtered["Governorate"].nunique())

st.markdown("---")

# Row 1: Monthly Trend + Governorate Chart
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    trend = df_filtered.groupby("Date")["Number of cases"].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(trend["Date"], trend["Number of cases"], marker='o', color='crimson')
    ax1.set_title("Monthly Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cases")
    ax1.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with row1_col2:
    gov_data = df_filtered.groupby("Governorate")["Number of cases"].sum().sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(gov_data.index, gov_data.values, color='skyblue')
    ax2.set_title("Cases by Governorate")
    ax2.set_xlabel("Cases")
    st.pyplot(fig2)

# Row 2: Seasonality + Forecast
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    df_filtered["Month Name"] = pd.to_datetime(df_filtered["Month"], format="%m").dt.strftime("%B")
    monthly_avg = df_filtered.groupby("Month Name")["Number of cases"].mean()
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    monthly_avg = monthly_avg.reindex(month_order)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(monthly_avg.index, monthly_avg.values, color='orange')
    ax3.set_title("Seasonal Pattern (Avg. Monthly)")
    ax3.set_ylabel("Avg. Cases")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

with row2_col2:
    monthly_cases = df_filtered.groupby("Date")["Number of cases"].sum().reset_index()
    monthly_cases = monthly_cases.dropna().sort_values("Date")
    monthly_cases["Date_Ordinal"] = monthly_cases["Date"].map(pd.Timestamp.toordinal)
    X = monthly_cases["Date_Ordinal"].values.reshape(-1, 1)
    y = monthly_cases["Number of cases"].values
    model = LinearRegression().fit(X, y)

    last_date = monthly_cases["Date"].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(future_ordinals)

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Cases": future_preds})
    combined = pd.concat([
        monthly_cases[["Date", "Number of cases"]].rename(columns={"Number of cases": "Cases"}),
        forecast_df.rename(columns={"Predicted Cases": "Cases"})
    ])

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(combined["Date"], combined["Cases"], marker='o', linestyle='--')
    ax4.axvline(x=last_date, color='gray', linestyle=':', label="Today")
    ax4.set_title("Forecast: Next 6 Months")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Cases")
    ax4.grid(True)
    ax4.legend(["Cases", "Forecast"])
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("📊 Developed for **MSBA350 – Healthcare Analytics** | Data: Ministry of Public Health, Lebanon")
