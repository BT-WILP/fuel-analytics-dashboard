import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fuel Analytics Dashboard", layout="wide")

st.title("🚗 Fuel Analytics Dashboard")

# -----------------------------
# Load Data from GitHub Repo
# -----------------------------
try:
    df = pd.read_csv("Fuel.csv")
except:
    st.error("Fuel.csv not found in repository.")
    st.stop()

# Clean column names
df.columns = ["Qty", "KM", "Date", "Battery", "Cost"]

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y")

# Sort by Date
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# Derived Calculations
# -----------------------------

# Distance travelled since last refill
df["Distance"] = df["KM"].diff()

# Mileage (km per litre)
df["Mileage"] = df["Distance"] / df["Qty"]

# Cost per litre
df["Cost_per_Litre"] = df["Cost"] / df["Qty"]

# Cost per km
df["Cost_per_KM"] = df["Cost"] / df["Distance"]

# Days between refills
df["Days_Between"] = df["Date"].diff().dt.days

# Fuel per dot
df["Fuel_per_Dot"] = df["Qty"] / df["Battery"]

# Drop first row (NaN due to diff)
df = df.dropna().reset_index(drop=True)

# -----------------------------
# KPI Section
# -----------------------------
st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Fuel Used (L)", round(df["Qty"].sum(), 2))
col2.metric("Total Money Spent (INR)", round(df["Cost"].sum(), 2))
col3.metric("Average Mileage (km/L)", round(df["Mileage"].mean(), 2))
col4.metric("Average Cost per KM (INR)", round(df["Cost_per_KM"].mean(), 2))

col5, col6 = st.columns(2)

col5.metric("Average Days Between Refills", round(df["Days_Between"].mean(), 2))
col6.metric("Average Fuel per Dot (L)", round(df["Fuel_per_Dot"].mean(), 2))

st.divider()

# -----------------------------
# Display Raw + Processed Data
# -----------------------------
st.subheader("📄 Complete Processed Dataset (Read-Only)")
st.dataframe(df)



