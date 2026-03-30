import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Streamlytics", page_icon="🎬", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("feature_engineered_netflix.csv")
    except:
        return pd.read_csv("cleaned_netflix_titles.csv")

df = load_data()

# -------------------------------
# OPTIONAL UPLOAD DATA
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload your dataset")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<style>
.big-title {font-size:48px;font-weight:bold;color:#E50914;}
.subtitle {font-size:20px;color:#aaa;}
</style>

<div class="big-title">🎬 Streamlytics</div>
<div class="subtitle">Netflix Content Strategy Analyzer</div>
""", unsafe_allow_html=True)

st.write("App started successfully 🚀")

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.title("🎛️ Filters Panel")
st.sidebar.markdown("---")

type_filter = st.sidebar.selectbox("Select Type", df['type'].dropna().unique())

year_min = int(df['release_year'].min())
year_max = int(df['release_year'].max())

year_filter = st.sidebar.slider(
    "Select Release Year",
    year_min,
    year_max,
    (year_min, year_max)
)

# -------------------------------
# SEARCH
# -------------------------------
search = st.text_input("🔍 Search Title")

# -------------------------------
# FILTER DATA
# -------------------------------
filtered_df = df[
    (df['type'] == type_filter) &
    (df['release_year'].between(year_filter[0], year_filter[1]))
]

if search:
    filtered_df = filtered_df[
        filtered_df['title'].str.contains(search, case=False, na=False)
    ]

# -------------------------------
# KPI SECTION
# -------------------------------
st.markdown("## 📊 Key Insights")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎥 Total Titles", len(filtered_df))
col2.metric("🎬 Movies", len(filtered_df[filtered_df['type'] == "Movie"]))
col3.metric("📺 TV Shows", len(filtered_df[filtered_df['type'] == "TV Show"]))
col4.metric("🌍 Countries", filtered_df['country'].nunique())

# -------------------------------
# DATA TABLE + DOWNLOAD
# -------------------------------
st.markdown("## 📂 Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    "⬇️ Download Filtered Data",
    filtered_df.to_csv(index=False),
    file_name="filtered_data.csv"
)

# -------------------------------
# CHART 1 - YEAR
# -------------------------------
st.markdown("## 📅 Content by Release Year")

fig1, ax1 = plt.subplots(figsize=(10,5))
sns.countplot(x='release_year', data=filtered_df, ax=ax1)
plt.xticks(rotation=90)
st.pyplot(fig1)

# -------------------------------
# CHART 2 - COUNTRY
# -------------------------------
st.markdown("## 🌍 Top Countries")

top_countries = filtered_df['country'].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(10,5))
top_countries.plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# -------------------------------
# AI SECTION
# -------------------------------
st.markdown("## 🤖 AI Insights")

if 'cluster' in df.columns:
    fig3, ax3 = plt.subplots()
    sns.countplot(x='cluster', data=df, ax=ax3)
    st.pyplot(fig3)
    st.success("✅ Machine Learning clustering applied successfully")
else:
    st.info("ℹ️ No cluster column found. Add ML clustering to dataset.")

# -------------------------------
# RECOMMENDATION SYSTEM
# -------------------------------
st.markdown("## 🎯 Recommendations")

title = st.selectbox("Select a title", df['title'].dropna().unique())

if title:
    selected_type = df[df['title'] == title]['type'].values[0]
    recs = df[df['type'] == selected_type].head(5)

    st.write("### Similar Content:")
    st.dataframe(recs[['title', 'type', 'country']])
