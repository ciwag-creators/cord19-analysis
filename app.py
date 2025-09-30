import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import os

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

# Function to create sample dataset
def create_sample_dataset():
    """Create a sample dataset from the full metadata.csv file"""
    try:
        if os.path.exists('metadata.csv'):
            # Read first 5000 rows from the full dataset
            metadata = pd.read_csv("metadata.csv", nrows=5000)
            # Save the sample to a new file
            metadata.to_csv("metadata_sample.csv", index=False)
            st.sidebar.success("✅ Sample dataset created successfully!")
            return True
        else:
            st.sidebar.error("❌ metadata.csv not found. Cannot create sample.")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ Error creating sample: {str(e)}")
        return False

# Sidebar for user controls
st.sidebar.header("Analysis Controls")
st.sidebar.markdown("Customize the analysis using the options below:")

# Add sample dataset creation to sidebar
st.sidebar.header("📁 Data Management")
if st.sidebar.button("Create Sample Dataset (5000 rows)"):
    with st.sidebar:
        with st.spinner("Creating sample dataset..."):
            create_sample_dataset()

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        # Check which dataset is available (in order of preference)
        if os.path.exists('metadata.csv'):
            df = pd.read_csv('metadata.csv', low_memory=False)
            st.sidebar.success("✅ Loaded full CORD-19 dataset")
        elif os.path.exists('metadata_sample.csv'):
            df = pd.read_csv('metadata_sample.csv', low_memory=False)
            st.sidebar.info("📊 Using sample dataset (5000 rows)")
        elif os.path.exists('sample_metadata.csv'):
            df = pd.read_csv('sample_metadata.csv')
            st.sidebar.warning("⚠️ Using demo sample data")
        else:
            st.error("""
            ❌ No dataset found. Please:
            
            1. **Option 1**: Download `metadata.csv` from [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
            2. **Option 2**: Click 'Create Sample Dataset' button above (requires metadata.csv)
            3. **Option 3**: The app will use built-in demo data
            """)
            # Create a minimal demo dataset as fallback
            demo_data = {
                'cord_uid': ['demo001', 'demo002', 'demo003'],
                'title': [
                    'COVID-19 Transmission Dynamics Study',
                    'Vaccine Efficacy Analysis 2021',
                    'Public Health Response to Pandemic'
                ],
                'authors': ['Smith J et al.', 'Johnson A et al.', 'Lee K et al.'],
                'journal': ['Lancet', 'Nature', 'Science'],
                'publish_time': ['2020-03-15', '2021-02-20', '2022-01-10'],
                'abstract': [
                    'Study of transmission patterns in urban areas during COVID-19 pandemic.',
                    'Analysis of vaccine effectiveness across different population groups.',
                    'Evaluation of public health measures implemented during the pandemic.'
                ],
                'source_x': ['PubMed', 'arXiv', 'WHO']
            }
            df = pd.DataFrame(demo_data)
            st.sidebar.warning("⚠️ Using minimal demo data")
        
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
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Display basic dataset information
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Overall Missing Data", f"{missing_percentage:.1f}%")
    
    with col4:
        # Show which dataset is being used
        if os.path.exists('metadata.csv') and len(df) > 10000:
            dataset_type = "Full Dataset"
        elif os.path.exists('metadata_sample.csv'):
            dataset_type = "Sample (5000 rows)"
        else:
            dataset_type = "Demo Data"
        st.metric("Dataset Type", dataset_type)
    
    # Data preview
    st.subheader("📋 Data Preview")
    preview_rows = st.slider("Number of rows to display:", 5, 20, 10, key="preview_slider")
    st.dataframe(df.head(preview_rows))
    
    # Show dataset source information
    with st.expander("📁 Dataset Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Available Data Files:**")
            files = []
            if os.path.exists('metadata.csv'):
                size = os.path.getsize('metadata.csv') / (1024 * 1024 * 1024)  # GB
                files.append(f"metadata.csv ({size:.2f} GB)")
            if os.path.exists('metadata_sample.csv'):
                size = os.path.getsize('metadata_sample.csv') / (1024 * 1024)  # MB
                files.append(f"metadata_sample.csv ({size:.2f} MB)")
            if os.path.exists('sample_metadata.csv'):
                size = os.path.getsize('sample_metadata.csv') / 1024  # KB
                files.append(f"sample_metadata.csv ({size:.2f} KB)")
            
            if files:
                for file in files:
                    st.write(f"• {file}")
            else:
                st.write("No data files found")
        
        with col2:
            st.write("**Data Source:**")
            if os.path.exists('metadata.csv'):
                st.write("Using original CORD-19 dataset")
            elif os.path.exists('metadata_sample.csv'):
                st.write("Using 5000-row sample from CORD-19")
            else:
                st.write("Using built-in demo data")
    
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
    if numerical_cols:
        selected_col = st.selectbox("Select numerical column for statistics:", numerical_cols, key="stats_select")
        st.write(df[selected_col].describe())
    else:
        st.info("No numerical columns found for statistics.")
    
    # Analysis Section
    st.header("📊 Analysis Results")
    
    # Publication trends over time
    st.subheader("📅 Publications Over Time")
    
    # Filter by year range
    valid_years = df[df['publication_year'].notna()]['publication_year']
    if not valid_years.empty:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
        year_range = st.slider("Select year range:", min_year, max_year, (min_year, max_year), key="year_slider")
        
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
    else:
        st.info("No publication year data available.")
        filtered_df = df
    
    # Top journals
    st.subheader("🏆 Top Publishing Journals")
    
    top_n = st.slider("Number of top journals to display:", 5, 20, 10, key="journal_slider")
    top_journals = filtered_df['journal'].value_counts().head(top_n)
    
    if not top_journals.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_journals.plot(kind='bar', ax=ax)
        ax.set_title(f'Top {top_n} Journals by Number of Publications')
        ax.set_xlabel('Journal')
        ax.set_ylabel('Number of Publications')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.info("No journal data available.")
    
    # Word cloud of titles
    st.subheader("☁️ Word Cloud of Paper Titles")
    
    # Combine all titles
    all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
    
    if all_titles.strip():
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
    else:
        st.info("No title data available for word cloud generation.")
    
    # Distribution of paper counts by source
    st.subheader("📚 Distribution of Papers by Source")
    
    if 'source_x' in filtered_df.columns:
        source_counts = filtered_df['source_x'].value_counts().head(10)
        if not source_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribution of Papers by Source')
            st.pyplot(fig)
        else:
            st.info("No source data available.")
    else:
        st.info("Source column not found in dataset.")
    
    # Abstract word count distribution
    st.subheader("📝 Abstract Word Count Distribution")
    
    if 'abstract_word_count' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(filtered_df['abstract_word_count'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_title('Distribution of Abstract Word Counts')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Number of Papers')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Abstract word count data not available.")
    
    # Interactive data exploration
    st.header("🔍 Interactive Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Filter Data by Journal")
        journal_options = ['All'] + filtered_df['journal'].value_counts().head(20).index.tolist()
        selected_journal = st.selectbox("Select a journal:", journal_options, key="journal_select")
        
        if selected_journal != 'All':
            journal_df = filtered_df[filtered_df['journal'] == selected_journal]
            st.write(f"**Papers from {selected_journal}:** {len(journal_df)}")
            if len(journal_df) > 0:
                st.dataframe(journal_df[['title', 'publication_year', 'abstract_word_count']].head(10))
            else:
                st.info("No papers found for selected journal.")
    
    with col2:
        st.subheader("Search Papers by Keyword")
        search_term = st.text_input("Enter keyword to search in titles:", key="search_input")
        
        if search_term:
            search_results = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
            st.write(f"**Found {len(search_results)} papers containing '{search_term}'**")
            if len(search_results) > 0:
                st.dataframe(search_results[['title', 'journal', 'publication_year']].head(10))
            else:
                st.info("No papers found with the search term.")
    
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
    
    1. **Download the data**: Get `metadata.csv` from [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
    2. **Place it** in the same directory as this script
    3. **Install packages**: `pip install streamlit pandas matplotlib seaborn wordcloud`
    4. **Run the app**: `streamlit run app.py`
    5. **Optional**: Click "Create Sample Dataset" in the sidebar to generate a smaller 5000-row version
    
    The app will work with:
    - Full dataset (if metadata.csv is available)
    - Sample dataset (5000 rows, created from full dataset)
    - Built-in demo data (if no files are available)
    """)
