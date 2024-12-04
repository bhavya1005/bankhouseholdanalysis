import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Introduction", layout="wide")

# Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Audrey&display=swap');
        
        /* Apply Audrey font globally */
        html, body, [class*="css"] {
            font-family: 'Raleway', sans-serif;
        }

        /* Title styling with Bebas Neue */
        .center-title {
            text-align: center;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 70px; /* Large heading */
            color: #003366; /* Dark blue color */
        }

        /* Background color */
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
    <h1 class="center-title">Bank Household Analysis Dashboard</h1>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
Welcome to the **Bank Household Analysis Dashboard**!

This project dives into the fascinating process by which large banks identify households based on customer data. By analyzing demographic and financial metrics, this dashboard aims to uncover meaningful patterns, trends, and relationships that drive customer behavior and financial health. These insights can help shape banking strategies, improve decision-making, and enhance customer experience.

### Why Household Analysis Matters:
Understanding households is crucial for banks to:
- Tailor financial products and services to customer needs.
- Accurately assess credit risk and income levels.
- Identify cross-selling opportunities within a household.
- Streamline account management for families or shared accounts.

This dashboard provides a **comprehensive overview** of the dataset, including:
- **Dataset Overview**: Explore the raw data and get a sense of its structure and content.
- **Summary Statistics**: Gain quick insights into the key metrics with aggregated statistics.
- **Customer Distribution**: Visualize customer demographics and their geographic spread.
- **Correlation Heatmap**: Analyze relationships between numerical features to identify potential trends and dependencies.

### How to Use This Dashboard:
Use the **sidebar navigation** to explore:
1. **Dataset Overview**: Examine the raw dataset and summary statistics.
2. **Initial Data Analysis**: Visualize customer distributions and identify key trends.
3. **Correlation Insights**: Understand interdependencies between demographic and financial metrics.

This dashboard simulates real-world data challenges by introducing **missingness at random**, replicating the inconsistencies banks often face. With this approach, we ensure the analysis reflects real-world scenarios.
""")

# Paragraph on Missingness Introduction
st.subheader("Why Introduce Missingness?")
st.markdown("""
In real-world scenarios, datasets are rarely complete. Customers may leave certain fields blank due to privacy concerns or simple oversight. 
To replicate these inconsistencies, we have introduced missing values at random in our dataset. This ensures our analysis and insights are realistic and robust, 
reflecting actual challenges faced in data analysis.
""")

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("customerdataset.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

data = load_data()

if not data.empty:
    # Dataset Overview
    st.subheader("Dataset Overview")
    st.markdown("""
    Below is the dataset used for the Bank Household Analysis. Here's what each column represents:

    - **First_Name**: The first name of the customer.
    - **Last_Name**: The last name of the customer, often used to identify families or households.
    - **Address**: The full residential address of the customer.
    - **State**: The U.S. state where the customer resides.
    - **Pincode**: The postal code associated with the customer's address.
    - **Income**: The annual income of the customer in USD.
    - **Phone_Number**: The customer's phone number.
    - **Credit_Score**: A numerical value representing the creditworthiness of the customer.
    - **Age**: The age of the customer in years.

    The table below displays the dataset for a closer look:
    """)
    st.dataframe(data)

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Initial Data Analysis
    st.subheader("Initial Data Analysis")

    # Total number of customers per state
    st.markdown("#### Total Number of Customers per State")
    total_customers = data['State'].value_counts().reset_index()
    total_customers.columns = ['State', 'Number of Customers']
    fig1 = px.bar(total_customers, x='State', y='Number of Customers', title="Total Customers Per State")
    st.plotly_chart(fig1, use_container_width=True)

    # Average number of customers per state
    st.markdown("#### Average Number of Customers per State")
    avg_customers = total_customers['Number of Customers'].mean()
    st.write(f"The average number of customers per state is approximately **{avg_customers:.2f}**.")

    # Correlation Heatmap
    st.markdown("#### Correlation Heatmap")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numerical_columns].corr()

    # Seaborn heatmap
    fig2, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig2)

    # Explanation of the heatmap insights
    st.markdown("""
    ### Insights from the Correlation Heatmap:

    1. **Income and Credit Score**:
       - There is a moderate positive correlation (**0.57**) between `Income` and `Credit_Score`. This indicates that individuals with higher incomes generally have better credit scores, reflecting financial stability.

    2. **Age and Income**:
       - A weak positive correlation (**0.23**) is observed between `Age` and `Income`. This suggests that income tends to increase with age, likely due to career progression, but the effect is not very strong.

    3. **Age and Credit Score**:
       - The correlation between `Age` and `Credit_Score` is weaker (**0.13**) but positive, indicating a slight tendency for older individuals to have better credit scores, possibly due to longer credit histories.

    4. **Pincode Independence**:
       - `Pincode` shows no meaningful correlation with any other variable, suggesting that location data is not directly influencing `Income`, `Credit_Score`, or `Age` in this dataset.

    ### Why is this Important?
    Understanding these relationships helps us identify patterns in the data:
    - Variables with strong correlations (e.g., `Income` and `Credit_Score`) might be used together in predictive models.
    - Weak or no correlation (e.g., `Pincode`) indicates independence, so these variables might not interact significantly with others.
    """)
else:
    st.warning("The dataset is empty or could not be loaded.")

# GitHub Link
st.sidebar.markdown("""
---
**Created by [Bhavya Chawla](https://github.com/bhavya1005)** 
""")
