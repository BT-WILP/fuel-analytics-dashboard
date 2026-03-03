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

# Days between refills
df["Days_Between"] = df["Date"].diff().dt.days

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
col4.metric("Average Days Between Refills", round(df["Days_Between"].mean(), 2))

st.divider()

# -----------------------------
# Display Raw + Processed Data
# -----------------------------
st.subheader("📄 Complete Processed Dataset (Read-Only)")
st.dataframe(df)


# =====================================================
# 📈 Trend Analysis
# =====================================================

st.header("📈 Trend Analysis")

# Mileage Trend
st.subheader("Mileage Over Time")
st.line_chart(df.set_index("Date")["Mileage"])

# Cost per Litre Trend
st.subheader("Fuel Price (Cost per Litre) Over Time")
st.line_chart(df.set_index("Date")["Cost_per_Litre"])

# =====================================================
# 📅 Yearly Cost Breakdown
# =====================================================

st.header("📅 Yearly Fuel Expenditure")

df["Year"] = df["Date"].dt.year
yearly_cost = df.groupby("Year")["Cost"].sum()

st.bar_chart(yearly_cost)

# =====================================================
# 🌦 Seasonal Performance Analysis
# =====================================================

st.header("🌦 Seasonal Performance Analysis")

def get_season(month):
    if month in [11, 12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5, 6]:
        return "Summer"
    else:
        return "Monsoon"

df["Season"] = df["Date"].dt.month.apply(get_season)

seasonal_mileage = df.groupby("Season")["Mileage"].mean()
seasonal_days = df.groupby("Season")["Days_Between"].mean()

st.subheader("Average Mileage by Season")
st.bar_chart(seasonal_mileage)

st.subheader("Average Days Between Refills by Season")
st.bar_chart(seasonal_days)


# =====================================================
# 📉 Mileage Trend Regression
# =====================================================

from sklearn.linear_model import LinearRegression
import numpy as np

st.header("📉 Mileage Trend Regression Analysis")

# Create time index
df["Time_Index"] = np.arange(len(df))

X = df[["Time_Index"]]
y = df["Mileage"]

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]

st.write(f"Regression Slope (Mileage change per refill): {round(slope, 4)} km/L per refill")

if slope > 0:
    st.success("Mileage is improving over time.")
elif slope < 0:
    st.warning("Mileage is declining over time.")
else:
    st.info("Mileage is stable over time.")

# Plot regression line
df["Predicted_Mileage"] = model.predict(X)

st.line_chart(df.set_index("Date")[["Mileage", "Predicted_Mileage"]])


# =====================================================
# 📊 Mileage Stability Analysis
# =====================================================

st.header("📊 Mileage Stability Analysis")

mileage_std = df["Mileage"].std()
mileage_mean = df["Mileage"].mean()

coefficient_of_variation = mileage_std / mileage_mean

st.write(f"Standard Deviation of Mileage: {round(mileage_std, 3)}")
st.write(f"Coefficient of Variation: {round(coefficient_of_variation, 4)}")

if coefficient_of_variation < 0.05:
    st.success("Fuel efficiency is highly stable.")
elif coefficient_of_variation < 0.10:
    st.info("Fuel efficiency is moderately stable.")
else:
    st.warning("Fuel efficiency shows noticeable variability.")


# =====================================================
# 🔗 Correlation Analysis
# =====================================================

import seaborn as sns
import matplotlib.pyplot as plt

st.header("🔗 Correlation Matrix")

corr_columns = ["Qty", "Distance", "Mileage", "Cost", "Cost_per_Litre", "Days_Between"]

corr_matrix = df[corr_columns].corr()

fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# =====================================================
# 🔮 Forecast Next Refill Date
# =====================================================

st.header("🔮 Next Refill Forecast")

# Average km per day
df["KM_per_Day"] = df["Distance"] / df["Days_Between"]
avg_km_per_day = df["KM_per_Day"].mean()

# Average distance per tank
avg_distance_per_tank = df["Distance"].mean()

# Remaining distance until next refill
last_km = df["KM"].iloc[-1]
predicted_days_until_empty = avg_distance_per_tank / avg_km_per_day

predicted_refill_date = df["Date"].iloc[-1] + pd.Timedelta(days=predicted_days_until_empty)

st.write(f"Average KM per Day: {round(avg_km_per_day, 2)} km/day")
st.write(f"Expected Days Until Next Refill: {round(predicted_days_until_empty, 1)} days")
st.write(f"Predicted Next Refill Date: {predicted_refill_date.date()}")

