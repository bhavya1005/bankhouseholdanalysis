import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# 1. App Configuration
# ---------------------------
st.set_page_config(
    page_title="Household Insights",
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
            font-size: 60px;
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
    <h1 class="title">🏠 Household Insights</h1>
""", unsafe_allow_html=True)

# ---------------------------
# 3. Load Household Dataset
# ---------------------------
@st.cache_data
def load_household_data(df):
    """
    Create a household-level dataset by grouping customers with the same last name and address.
    Filter households with more than one member.
    """
    household_df = df.groupby(['Last_Name', 'Address', 'State'], as_index=False).agg({
        'Income': 'sum',  # Sum income for all members of a household
        'Credit_Score': 'mean',  # Average credit score for the household
        'Age': 'mean',  # Average age for the household
        'First_Name': 'count',  # Count the number of members in the household
    }).rename(columns={
        'Income': 'Total_Income',
        'Credit_Score': 'Average_Credit_Score',
        'Age': 'Average_Age',
        'First_Name': 'Household_Size',
    })
    # Filter households with more than one member
    household_df = household_df[household_df['Household_Size'] > 1]
    return household_df

# Load customer-level dataset
data_filepath = "customerdataset.csv"  # Replace with your dataset file path
raw_df = pd.read_csv(data_filepath)

# Generate household-level dataset
household_df = load_household_data(raw_df)

# ---------------------------
# 4. Household Page Content
# ---------------------------
st.subheader("Understanding Households")
st.markdown("""
In this analysis, households are identified by grouping customers who share the same last name and residential address.
Households with more than one member are considered for this page, as they provide meaningful insights for banks.

### Why Identify Households?
- **Customer Segmentation**: Households can be treated as a unit for financial product offerings.
- **Better Insights**: Aggregated data reveals financial capacity and creditworthiness at the household level.
- **Targeted Marketing**: Banks can design services tailored to household needs, such as loans or family-oriented accounts.
""")

# **Section 1: Display Household Dataset**
st.subheader("📋 Household Dataset")
st.markdown("""
The table below summarizes household-level information, including the total income, average credit score, 
average age, and household size for each group.
""")
st.dataframe(household_df.head(100))  # Display the first 100 rows
st.markdown(f"**Total Households (with >1 member):** {household_df.shape[0]}")

# ---------------------------
# 5. Visualizations with Explanations
# ---------------------------

# **Visualization 1: Number of Households by State**
st.subheader("📍 Number of Households by State")
st.markdown("""
The bar chart below shows the distribution of households across different states. 
This helps identify regions with a higher concentration of customers, guiding regional marketing strategies.
""")
household_state_counts = household_df['State'].value_counts().reset_index()
household_state_counts.columns = ['State', 'Number_of_Households']
fig_households_state = px.bar(
    household_state_counts.sort_values('Number_of_Households', ascending=False),
    x='Number_of_Households',
    y='State',
    title="Number of Households by State",
    labels={'Number_of_Households': 'Number of Households', 'State': 'State'},
    orientation='h',
    color='Number_of_Households',
    color_continuous_scale='Viridis',
)
st.plotly_chart(fig_households_state, use_container_width=True)
st.markdown("""
**Insights:**
- States with a higher number of households indicate larger customer bases.
- These states can be prioritized for household-targeted services or marketing efforts.
""")

# **Visualization 2: Household Size Distribution**
st.subheader("👨‍👩‍👧 Household Size Distribution")
st.markdown("""
This histogram visualizes the distribution of household sizes (i.e., the number of members in a household).
Larger households may have unique financial needs, such as family-oriented savings accounts or loans.
""")
fig_household_size = px.histogram(
    household_df,
    x='Household_Size',
    nbins=10,
    title="Distribution of Household Sizes",
    labels={'Household_Size': 'Household Size'},
    color_discrete_sequence=['#636EFA'],
)
st.plotly_chart(fig_household_size, use_container_width=True)
st.markdown("""
**Insights:**
- Most households in the dataset have 2-4 members, indicating a predominance of small family units.
- Larger households (>5 members) are less common but may require customized financial products.
""")

# **Visualization 3: Income vs. Household Size**
st.subheader("💸 Income vs. Household Size")
st.markdown("""
This scatter plot examines how total household income varies with household size. 
Larger households often have higher incomes due to multiple earners, but this can vary by state.
""")
fig_income_household_size = px.scatter(
    household_df,
    x='Household_Size',
    y='Total_Income',
    size='Total_Income',
    color='State',
    hover_data=['Address'],
    title="Income vs. Household Size",
    labels={'Household_Size': 'Household Size', 'Total_Income': 'Total Household Income'},
    color_continuous_scale='Viridis',
)
st.plotly_chart(fig_income_household_size, use_container_width=True)
st.markdown("""
**Insights:**
- Larger households generally have higher incomes, likely due to multiple contributors.
- Outliers (low-income large households or high-income small households) can indicate unique cases worth exploring.
""")

# **Visualization 4: Average Credit Score by Household Size**
st.subheader("📈 Average Credit Score by Household Size")
st.markdown("""
This line chart explores the relationship between household size and the average credit score.
Understanding creditworthiness across different household sizes can guide loan approvals and financial risk assessments.
""")
avg_credit_score = household_df.groupby('Household_Size')['Average_Credit_Score'].mean().reset_index()
fig_credit_score = px.line(
    avg_credit_score,
    x='Household_Size',
    y='Average_Credit_Score',
    title="Average Credit Score by Household Size",
    labels={'Household_Size': 'Household Size', 'Average_Credit_Score': 'Average Credit Score'},
    markers=True,
)
st.plotly_chart(fig_credit_score, use_container_width=True)
st.markdown("""
**Insights:**
- Average credit scores tend to decrease slightly as household size increases, potentially due to financial strain in larger households.
- Smaller households (2-3 members) tend to maintain higher credit scores, indicating better credit management.
""")
