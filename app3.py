import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# ✅ Set page config at the top
st.set_page_config(page_title="🚚 Amazon Market Basket & Delivery Analysis", page_icon="📦")

# 🚀 Function to load and preprocess data
@st.cache_data
def load_data():
    file_path = "amazon.csv"  # Ensure this file is in your GitHub repo

    if not os.path.exists(file_path):
        st.error("⚠️ File not found! Please upload 'amazon.csv' to your GitHub repository.")
        return None

    df = pd.read_csv(file_path)

    # ✅ Strip spaces from column names
    df.columns = df.columns.str.strip()

    # ✅ Print column names for debugging
    st.write("✅ Available columns:", df.columns.tolist())

    # ✅ Ensure 'Order_Date' exists and convert to datetime
    if "Order_Date" not in df.columns:
        st.error("❌ Column 'Order_Date' not found in dataset! Please check column names.")
        return None

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")

    # ✅ Convert 'Time_taken(min)' to numeric
    if "Time_taken(min)" in df.columns:
        df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.extract(r"(\d+)").astype(float)
    else:
        st.error("❌ Column 'Time_taken(min)' not found in dataset! Please check the dataset.")
        return None

    return df

# ✅ Load data
df = load_data()

if df is None:
    st.stop()

st.title("📦 Amazon Market Basket & Delivery Analysis")
st.write("### Sample Data")
st.dataframe(df.head())

# ✅ Sidebar Filters
st.sidebar.title("🔍 Filter Options")

# Ensure 'Order_Date' column exists
if "Order_Date" not in df.columns:
    st.error("❌ Column 'Order_Date' not found in dataset.")
    st.stop()

selected_date = st.sidebar.selectbox("Select Order Date", options=df["Order_Date"].dt.date.unique())

# Ensure 'Road_traffic_density' column exists
if "Road_traffic_density" not in df.columns:
    st.error("❌ Column 'Road_traffic_density' not found in dataset.")
    st.stop()

selected_traffic = st.sidebar.multiselect(
    "Select Traffic Density",
    options=df["Road_traffic_density"].dropna().unique(),
    default=df["Road_traffic_density"].dropna().unique()
)

min_delivery_time = st.sidebar.slider(
    "Minimum Delivery Time (mins)",
    min_value=int(df["Time_taken(min)"].min()),
    max_value=int(df["Time_taken(min)"].max()),
    value=int(df["Time_taken(min)"].median())
)

num_records = st.sidebar.slider("Number of Records to Display", min_value=1, max_value=50, value=10)

# ✅ Filter Data
filtered_data = df[
    (df["Order_Date"].dt.date == selected_date) &
    (df["Time_taken(min)"] >= min_delivery_time) &
    (df["Road_traffic_density"].isin(selected_traffic))
].head(num_records)

st.write(f"**Filtered Results for Date {selected_date} with Min Delivery Time {min_delivery_time} mins:**")
st.dataframe(filtered_data)

# ✅ Visualization Tabs
tabs = st.tabs(["📊 Delivery Time Distribution", "🚦 Traffic Impact on Delivery"])

with tabs[0]:
    st.subheader("📊 Delivery Time Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_data["Time_taken(min)"], bins=10, color='blue', alpha=0.7)
    ax.set_xlabel("Time Taken (mins)")
    ax.set_ylabel("Number of Deliveries")
    ax.set_title("Distribution of Delivery Time")
    st.pyplot(fig)

with tabs[1]:
    st.subheader("🚦 Traffic Density vs Delivery Time")
    traffic_summary = filtered_data.groupby("Road_traffic_density")["Time_taken(min)"].mean().reset_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(traffic_summary["Road_traffic_density"], traffic_summary["Time_taken(min)"], color='red')
    ax2.set_xlabel("Traffic Density")
    ax2.set_ylabel("Avg. Delivery Time (mins)")
    ax2.set_title("Impact of Traffic on Delivery Time")
    st.pyplot(fig2)

# ✅ Clustering: Customer Segmentation
st.write("### 🏷️ Customer Segmentation")

features = df[['Time_taken(min)']].fillna(0)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['customer_segment'] = kmeans.fit_predict(scaled_features)

fig, ax = plt.subplots()
sns.scatterplot(x=df['Time_taken(min)'], y=df['customer_segment'], hue=df['customer_segment'], palette='viridis', ax=ax)
st.pyplot(fig)

# ✅ Association Rule Mining (Market Basket Analysis)
st.write("### 🛒 Market Basket Analysis")

if all(col in df.columns for col in ['user_id', 'product_name', 'category']):
    basket = df.groupby(['user_id', 'product_name'])['category'].count().unstack().reset_index().fillna(0)
    basket.set_index('user_id', inplace=True)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)

    if frequent_itemsets.empty:
        st.warning("No frequent itemsets found with the current min_support. Try lowering it further.")
    else:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        st.write("### Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    st.warning("⚠️ Required columns ('user_id', 'product_name', 'category') are missing for Market Basket Analysis.")

st.write("🚀 Data-driven insights made easy!")




