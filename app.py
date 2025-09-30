import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Dataset Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔬 CORD-19 Research Dataset Analysis")
st.markdown("""
This application provides a basic analysis of the COVID-19 Open Research Dataset (CORD-19) metadata.
The dataset contains information about research papers related to COVID-19.
""")

# Sidebar for user controls
st.sidebar.header("Analysis Controls")
st.sidebar.markdown("Customize the analysis using the options below:")

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        # Load the dataset
        df = pd.read_csv('metadata.csv', low_memory=False)
        
        # Data cleaning and preprocessing
        # Convert publication date to datetime
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        
        # Extract year from publication date
        df['publication_year'] = df['publish_time'].dt.year
        
        # Create abstract word count column
        df['abstract_word_count'] = df['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Handle missing values for important columns
        df['journal'] = df['journal'].fillna('Unknown Journal')
        df['abstract'] = df['abstract'].fillna('No abstract available')
        
        return df
    except FileNotFoundError:
        st.error("❌ metadata.csv file not found. Please make sure it's in the same directory as this script.")
        st.info("💡 You can download the metadata.csv file from: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
        return None

# Load the data
df = load_data()

if df is not None:
    # Display basic dataset information
    st.header("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Papers", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Overall Missing Data", f"{missing_percentage:.1f}%")
    
    # Data preview
    st.subheader("📋 Data Preview")
    preview_rows = st.slider("Number of rows to display:", 5, 20, 10)
    st.dataframe(df.head(preview_rows))
    
    # Data structure information
    st.subheader("🔍 Data Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        dtype_info = df.dtypes.reset_index()
        dtype_info.columns = ['Column', 'Data Type']
        st.dataframe(dtype_info, height=300)
    
    with col2:
        st.write("**Missing Values:**")
        missing_info = df.isnull().sum().reset_index()
        missing_info.columns = ['Column', 'Missing Values']
        missing_info['Percentage'] = (missing_info['Missing Values'] / len(df) * 100).round(2)
        st.dataframe(missing_info, height=300)
    
    # Basic statistics
    st.subheader("📈 Basic Statistics")
    
    # Select numerical column for statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'abstract_word_count' in numerical_cols:
        selected_col = st.selectbox("Select numerical column for statistics:", numerical_cols)
        st.write(df[selected_col].describe())
    
    # Analysis Section
    st.header("📊 Analysis Results")
    
    # Publication trends over time
    st.subheader("📅 Publications Over Time")
    
    # Filter by year range
    min_year = int(df['publication_year'].min())
    max_year = int(df['publication_year'].max())
    year_range = st.slider("Select year range:", min_year, max_year, (min_year, max_year))
    
    # Filter data by selected year range
    filtered_df = df[(df['publication_year'] >= year_range[0]) & (df['publication_year'] <= year_range[1])]
    
    # Plot publications over time
    yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_counts.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Number of Publications Over Time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Top journals
    st.subheader("🏆 Top Publishing Journals")
    
    top_n = st.slider("Number of top journals to display:", 5, 20, 10)
    top_journals = filtered_df['journal'].value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_journals.plot(kind='bar', ax=ax)
    ax.set_title(f'Top {top_n} Journals by Number of Publications')
    ax.set_xlabel('Journal')
    ax.set_ylabel('Number of Publications')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    # Word cloud of titles
    st.subheader("☁️ Word Cloud of Paper Titles")
    
    # Combine all titles
    all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
    
    # Clean the text
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
    word_freq = Counter(words)
    
    # Remove common stop words
    stop_words = {'this', 'that', 'with', 'from', 'have', 'were', 'been', 'their', 
                 'which', 'study', 'research', 'using', 'based', 'results', 'method'}
    filtered_words = {word: count for word, count in word_freq.items() if word not in stop_words}
    
    if filtered_words:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(filtered_words)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Most Frequent Words in Paper Titles')
        st.pyplot(fig)
    else:
        st.info("No sufficient title data available for word cloud generation.")
    
    # Distribution of paper counts by source
    st.subheader("📚 Distribution of Papers by Source")
    
    source_counts = filtered_df['source_x'].value_counts().head(10)  # Using source_x column
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribution of Papers by Source')
    st.pyplot(fig)
    
    # Abstract word count distribution
    st.subheader("📝 Abstract Word Count Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(filtered_df['abstract_word_count'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Abstract Word Counts')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Number of Papers')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Interactive data exploration
    st.header("🔍 Interactive Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Filter Data by Journal")
        selected_journal = st.selectbox("Select a journal:", ['All'] + filtered_df['journal'].value_counts().head(20).index.tolist())
        
        if selected_journal != 'All':
            journal_df = filtered_df[filtered_df['journal'] == selected_journal]
            st.write(f"**Papers from {selected_journal}:** {len(journal_df)}")
            st.dataframe(journal_df[['title', 'publication_year', 'abstract_word_count']].head(10))
    
    with col2:
        st.subheader("Search Papers by Keyword")
        search_term = st.text_input("Enter keyword to search in titles:")
        
        if search_term:
            search_results = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
            st.write(f"**Found {len(search_results)} papers containing '{search_term}'**")
            if len(search_results) > 0:
                st.dataframe(search_results[['title', 'journal', 'publication_year']].head(10))
    
    # Download cleaned data
    st.sidebar.header("📥 Data Export")
    if st.sidebar.button("Download Cleaned Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="cord19_cleaned_data.csv",
            mime="text/csv"
        )

else:
    st.info("""
    ## Instructions to run this application:
    
    1. Download the `metadata.csv` file from the CORD-19 dataset
    2. Place it in the same directory as this script
    3. Install the required packages: `pip install streamlit pandas matplotlib seaborn wordcloud`
    4. Run the application: `streamlit run app.py`
    
    **Download link for metadata.csv:** 
    [CORD-19 Dataset on Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
    import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    # Check which dataset is available
    if os.path.exists('metadata.csv'):
        df = pd.read_csv('metadata.csv', low_memory=False)
        st.sidebar.success("✅ Loaded full CORD-19 dataset")
    elif os.path.exists('sample_metadata.csv'):
        df = pd.read_csv('sample_metadata.csv')
        st.sidebar.warning("⚠️ Using sample data. Download full dataset for complete analysis.")
    else:
        st.error("""
        No dataset found. Please:
        1. Download metadata.csv from [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
        2. Place it in this directory
        """)
        return None
    
    # Your data processing code here...
    return df
    """)
