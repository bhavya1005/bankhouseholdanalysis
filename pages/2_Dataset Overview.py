import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer

# ---------------------------
# 1. App Configuration
# ---------------------------
st.set_page_config(
    page_title="Dataset Overview",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# 2. Custom CSS for Styling
# ---------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Audrey&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Raleway', sans-serif;
        }

        .title {
            text-align: center;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 50px;
            color: #003366;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(
                to bottom,
                #f8e7e7, /* Soft Pink */
                #fff9e6, /* Soft Yellow */
                #e6f9f1, /* Soft Green */
                #e6f0ff  /* Soft Blue */
            );
        }
    </style>
    <h1 class="title">Dataset Overview</h1>
""", unsafe_allow_html=True)

# ---------------------------
# 3. Load Dataset
# ---------------------------
@st.cache_data
def load_dataset(filepath):
    return pd.read_csv(filepath)

# ---------------------------
# 4. Data Cleaning Function
# ---------------------------
@st.cache_data
def clean_dataset(df):
    # Handle missing numerical data
    numerical_cols = ['Income', 'Credit_Score', 'Age']
    imputer = SimpleImputer(strategy='mean')
    for col in numerical_cols:
        if col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])

    # Handle categorical data
    categorical_cols = ['Last_Name', 'State']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype('category')
    return df

# Load and clean dataset
data_filepath = "customerdataset.csv"  # Path to your dataset
raw_df = load_dataset(data_filepath)
cleaned_df = clean_dataset(raw_df.copy())

# ---------------------------
# 5. Dataset Page Content
# ---------------------------
st.markdown("""
### Welcome to the Dataset Overview Page

This page provides a comprehensive view of the dataset, including raw and cleaned data, summary statistics, and key insights through visualizations.
""")

# **Section 1: Display Original Dataset**
st.subheader("📋 Original Dataset")
st.markdown("""
Below is the raw dataset that forms the foundation of this analysis. You can explore the structure and data it contains:
""")
st.write(raw_df.head(100))
st.markdown(f"**Shape of the Dataset:** {raw_df.shape}")

# **Section 2: Missing Value Analysis**
st.subheader("📉 Missing Value Analysis")
missing_data = raw_df.isnull().sum()[raw_df.isnull().sum() > 0]
if not missing_data.empty:
    st.markdown("**Number of Missing Values in Each Column:**")
    st.write(missing_data)

    st.markdown("""
    The above table highlights the number of missing values in key columns. These gaps often arise due to incomplete customer forms, system errors, or synchronization delays. 
    To ensure the dataset is ready for analysis, we address these missing values systematically:
    
    - **Numerical Columns** (`Income`, `Credit_Score`, `Age`): Missing values are imputed with the column's mean to preserve overall data trends.
    - **Categorical Columns** (`Last_Name`, `State`): Missing values are replaced with the placeholder `'Unknown'` for consistency.
    """)
else:
    st.success("No missing data found!")

# **Section 3: Cleaned Dataset**
st.subheader("✅ Cleaned Dataset")
st.markdown("""
The cleaned dataset ensures all missing data is handled and all variables are consistent. Below is the cleaned dataset for reference:
""")
st.write(cleaned_df.head(100))
st.markdown(f"**Shape of the Dataset:** {cleaned_df.shape}")

# **Section 4: Summary Statistics**
st.subheader("📊 Summary Statistics")
st.markdown("""
Summary statistics provide an overview of the numerical variables in the dataset, helping us understand the distribution, central tendencies, and variability of key metrics:
""")
st.write(cleaned_df.describe())

# **Section 5: Correlation Heatmap**
st.subheader("🔥 Correlation Heatmap")
st.markdown("""
The correlation heatmap helps identify relationships between numerical features. Strong correlations can indicate trends that are useful for analysis or predictive modeling:
""")
corr_matrix = cleaned_df[['Income', 'Credit_Score', 'Age']].corr()
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu",
    title="Correlation Heatmap"
)
st.plotly_chart(fig_corr)
st.markdown("""
- A **moderate positive correlation** between `Income` and `Credit_Score` suggests that customers with higher incomes tend to have better credit scores.
- A **weak correlation** between `Age` and other variables indicates minimal dependency on customer age.
""")

# **Section 6: Visualizations**
st.subheader("📈 Key Insights Through Visualizations")

# Average Income by State
st.markdown("### Average Income by State")
avg_income_state = cleaned_df.groupby('State')['Income'].mean().reset_index()
fig_avg_income = px.bar(avg_income_state, x='State', y='Income', title="Average Income by State", color='Income')
st.plotly_chart(fig_avg_income)
st.markdown("""
This chart highlights the average income across different states, providing insights into geographic income distribution. 
Banks can identify high-income states for targeted marketing strategies.
""")

# Income vs. Credit Score
st.markdown("### Income vs. Credit Score")
fig_income_credit = px.scatter(
    cleaned_df,
    x='Income',
    y='Credit_Score',
    color='State',
    title="Income vs. Credit Score",
    labels={'Income': 'Annual Income', 'Credit_Score': 'Credit Score'}
)
st.plotly_chart(fig_income_credit)
st.markdown("""
This scatter plot illustrates the relationship between `Income` and `Credit_Score`. Clusters in the plot may represent state-specific trends or highlight outliers.
""")

# Age Distribution
st.markdown("### Age Distribution")
fig_age_dist = px.histogram(cleaned_df, x='Age', nbins=20, title="Age Distribution", color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig_age_dist)
st.markdown("""
The age distribution helps banks understand the dominant age groups in their customer base, enabling tailored financial products and services.
""")

# Credit Score Distribution
st.markdown("### Credit Score Distribution")
fig_credit_dist = px.histogram(cleaned_df, x='Credit_Score', nbins=20, title="Credit Score Distribution", color_discrete_sequence=['#EF553B'])
st.plotly_chart(fig_credit_dist)
st.markdown("""
The credit score distribution provides an overview of customers' financial health, with a focus on identifying customers with excellent or poor creditworthiness.
""")
