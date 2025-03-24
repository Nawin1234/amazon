import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")  # Update with your file path
    return df

df = load_data()

# Data Preprocessing
df['Time_taken(min)'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)

df.columns = df.columns.str.strip()

st.sidebar.title("Options")
if st.sidebar.button("Show Data Sample"):
    st.write("### Delivery Data Sample:", df.head())

# Exploratory Data Analysis
st.title("Delivery Data Analysis")
st.subheader("Exploratory Data Analysis")
if st.sidebar.button("Show Data Insights"):
    st.write("### Data Statistics:")
    st.write(df.describe())
    
    fig, ax = plt.subplots()
    sns.histplot(df['Time_taken(min)'].dropna(), bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# Association Rule Mining
st.subheader("Association Rule Mining")
if st.sidebar.button("Run Association Analysis"):
    basket = df.groupby(['City', 'Type_of_order'])['Time_taken(min)'].count().unstack().fillna(0)
    frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
    st.write(rules.head())

# K-Means Clustering
st.subheader("K-Means Clustering")
k = st.sidebar.slider("Select number of clusters", 2, 10, 3)
if st.sidebar.button("Run K-Means"):
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['Time_taken(min)']].dropna())
    
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Time_taken(min)'], y=df['Vehicle_condition'], hue=df['cluster'], palette='viridis', ax=ax)
    st.pyplot(fig)

st.write("### End of Analysis")

