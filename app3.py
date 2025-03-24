import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from mlxtend.frequent_patterns import apriori, association_rules

# ✅ Streamlit Page Config
st.set_page_config(page_title="🛒 E-Commerce Product Analysis", page_icon="📊")

# 🚀 Function to load and preprocess data
@st.cache_data
def load_data():
    file_path = "amazon.csv"  # Ensure this file is in your GitHub repo

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("⚠️ File not found! Please upload 'amazon.csv' to your GitHub repository.")
        return None

    # ✅ Strip spaces from column names
    df.columns = df.columns.str.strip()

    # ✅ Print column names for debugging
    st.write("✅ Available columns:", df.columns.tolist())

    # ✅ Convert numeric columns
    for col in ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ✅ Load Data
df = load_data()
if df is None:
    st.stop()

# ✅ App Title
st.title("🛒 E-Commerce Product & Review Analysis")
st.write("Analyze product pricing, ratings, and user reviews!")

# ✅ Sidebar Filters
st.sidebar.title("🔍 Filters")

# 🔹 Filter by Category
categories = df["category"].dropna().unique() if "category" in df.columns else []
selected_category = st.sidebar.selectbox("Select Category", options=["All"] + list(categories))

# 🔹 Filter by Rating
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0, 0.1)

# 🔹 Apply Filters
filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]
filtered_df = filtered_df[filtered_df["rating"] >= min_rating]

st.write(f"Showing **{len(filtered_df)}** products from category **{selected_category}** with rating **≥ {min_rating}**")
st.dataframe(filtered_df[["product_name", "category", "discounted_price", "actual_price", "rating", "rating_count"]].head(10))

# ✅ Visualization Tabs
tabs = st.tabs(["📊 Price vs Discount", "⭐ Top Rated Products", "📝 Review Word Cloud"])

with tabs[0]:
    st.subheader("📊 Price vs Discount Analysis")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x="actual_price", y="discounted_price", hue="discount_percentage", palette="coolwarm", size="discount_percentage", sizes=(20, 200), alpha=0.7, ax=ax)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Discounted Price")
    st.pyplot(fig)

with tabs[1]:
    st.subheader("⭐ Top Rated Products")
    top_rated = filtered_df.nlargest(10, "rating")[["product_name", "category", "rating", "rating_count"]]
    st.dataframe(top_rated)

with tabs[2]:
    st.subheader("📝 Review Word Cloud")
    if "review_content" in df.columns:
        review_text = " ".join(df["review_content"].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(review_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 'review_content' column not found in dataset!")

# ✅ Association Rule Mining (Market Basket Analysis)
st.write("### 🛒 Market Basket Analysis")
if all(col in df.columns for col in ["user_id", "product_name"]):
    basket = df.groupby(["user_id", "product_name"])["category"].count().unstack().reset_index().fillna(0)
    basket.set_index("user_id", inplace=True)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        st.write("### Association Rules")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    else:
        st.warning("No frequent itemsets found with the current min_support.")
else:
    st.warning("⚠️ 'user_id' and 'product_name' columns are required for Market Basket Analysis.")

st.write("🚀 Data-driven insights made easy!")





