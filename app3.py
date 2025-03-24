import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# ✅ Set Streamlit page config
st.set_page_config(page_title="🚚 Delivery Analysis", page_icon="📦")

# 🚀 Load the dataset from GitHub or local
@st.cache_data
def load_data():
    file_path = "amazon.csv"  # Update with your actual CSV filename

    if not os.path.exists(file_path):
        st.error("⚠️ Dataset not found! Please upload the correct dataset.")
        return None

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Convert Order_Date to datetime
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")

    # Extract numeric values from Time_taken(min)
    df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.extract(r"(\d+)").astype(float)

    return df

df = load_data()
if df is None:
    st.stop()

st.title("📦 Delivery & Market Basket Analysis")

# ✅ Sidebar Filters
st.sidebar.title("🔍 Filters")
selected_date = pd.to_datetime(st.sidebar.selectbox("Select Order Date", options=df["Order_Date"].unique()))
selected_traffic = st.sidebar.multiselect("Select Traffic Density", options=df["Road_traffic_density"].unique(), default=df["Road_traffic_density"].unique())
min_delivery_time = st.sidebar.slider("Minimum Delivery Time (mins)", min_value=int(df["Time_taken(min)"].min()), max_value=int(df["Time_taken(min)"].max()), value=int(df["Time_taken(min)"].median()))
num_records = st.sidebar.slider("Number of Records", min_value=1, max_value=50, value=10)

# ✅ Data Filtering
filtered_data = df[(df["Order_Date"] == selected_date) & (df["Time_taken(min)"] >= min_delivery_time) & (df["Road_traffic_density"].isin(selected_traffic))]
filtered_data = filtered_data.head(num_records)
st.write(f"**Filtered Results for {selected_date.date()}**")
st.dataframe(filtered_data)

# ✅ Delivery Time Distribution
st.subheader("📊 Delivery Time Distribution")
fig, ax = plt.subplots()
ax.hist(filtered_data["Time_taken(min)"], bins=10, color='blue', alpha=0.7)
ax.set_xlabel("Time Taken (mins)")
ax.set_ylabel("Deliveries")
ax.set_title("Delivery Time Distribution")
st.pyplot(fig)

# ✅ Traffic vs Delivery Time
st.subheader("🚦 Traffic Density vs Delivery Time")
traffic_summary = filtered_data.groupby("Road_traffic_density")["Time_taken(min)"].mean().reset_index()
fig2, ax2 = plt.subplots()
ax2.bar(traffic_summary["Road_traffic_density"], traffic_summary["Time_taken(min)"], color='red')
ax2.set_xlabel("Traffic Density")
ax2.set_ylabel("Avg. Delivery Time (mins)")
ax2.set_title("Traffic Impact on Delivery")
st.pyplot(fig2)

# ✅ Clustering: Customer Segmentation
st.subheader("🏷️ Customer Segmentation")
features = df[['Time_taken(min)']].fillna(0)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df['customer_segment'] = kmeans.fit_predict(scaled_features)
fig, ax = plt.subplots()
sns.scatterplot(x=df['Time_taken(min)'], y=df['customer_segment'], hue=df['customer_segment'], palette='viridis', ax=ax)
st.pyplot(fig)

# ✅ Market Basket Analysis
st.subheader("🛒 Market Basket Analysis")
basket = df.groupby(['Delivery_person_ID', 'Type_of_order'])['Type_of_vehicle'].count().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)
frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)

if frequent_itemsets.empty:
    st.warning("No frequent itemsets found. Try lowering the support threshold.")
else:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    st.write("### Association Rules")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

st.write("🚀 Data-driven insights made easy!")


