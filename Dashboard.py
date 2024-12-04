import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Bank Household Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for a polished and visually appealing design
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Audrey&display=swap');

        /* General App Styling */
        html, body, [class*="css"] {
            font-family: 'Audrey', sans-serif;
            background: linear-gradient(
                to bottom,
                #f8e7e7, /* Soft Pink */
                #fff9e6, /* Soft Yellow */
                #e6f9f1, /* Soft Green */
                #e6f0ff  /* Soft Blue */
            );
        }

        /* Title Styling */
        .title {
            text-align: center;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 60px;
            color: #003366; /* Dark Blue */
            margin-bottom: 20px;
        }

        /* Subtitle Styling */
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #555555; /* Subtle Gray */
            margin-bottom: 40px;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                to bottom,
                #e6f0ff, /* Soft Blue */
                #f8e7e7  /* Soft Pink */
            );
            color: #003366; /* Dark Blue Text */
            border-right: 2px solid #cccccc; /* Subtle Sidebar Border */
        }

        [data-testid="stSidebar"] h2 {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 24px;
            color: #003366; /* Dark Blue */
        }

        /* Instructions Styling */
        .instructions {
            font-size: 18px;
            color: #333333; /* Slightly Darker Gray */
            margin-bottom: 20px;
        }

        /* Additional Information Styling */
        .additional-info {
            font-size: 16px;
            color: #666666; /* Light Gray for Secondary Text */
            margin-top: 40px;
        }

        /* Footer Styling */
        hr {
            border: none;
            border-top: 1px solid #cccccc;
            margin: 40px 0;
        }

        .footer {
            text-align: center;
            font-size: 14px;
            color: #888888;
            margin-top: 20px;
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
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h1 class='title'>Bank Household Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Empowering financial insights through household-level data analysis.</p>", unsafe_allow_html=True)

# Introduction Section
st.markdown("""
Welcome to the **Bank Household Analysis Dashboard**! This application offers a comprehensive platform for analyzing household-level customer data in banking. Through advanced visualizations, machine learning models, and detailed data aggregation, the dashboard helps financial institutions make informed decisions.

This dashboard serves as a practical tool for:
- Understanding customer demographics.
- Aggregating household-level financial insights.
- Making predictive analyses to guide banking strategies.

### How to Navigate
The dashboard is structured into intuitive sections accessible from the **sidebar**:
1. **Introduction**: Learn about the project goals and overview.
2. **Dataset Overview**: Explore the raw dataset, including key statistics.
3. **Household Insights**: Analyze financial and demographic metrics aggregated at the household level.
4. **Analysis and Results**: Delve into modeling results, including regression and classification insights.
5. **Final Results**: Summarize the project's findings, challenges, and actionable insights.

### Dashboard Features
- **Visual Insights**: Dynamic charts and visualizations to uncover patterns in the data.
- **Predictive Models**: Regression and classification models for income and creditworthiness predictions.
- **Actionable Takeaways**: Data-driven recommendations tailored for the banking sector.
""")

# Sidebar Instructions
st.markdown("<p class='instructions'>Use the sidebar to navigate through various sections of the dashboard.</p>", unsafe_allow_html=True)

# Additional Information Section
st.markdown("""
<div class="additional-info">
<b>Notes:</b>
- The dataset powers all the analyses in the dashboard. Ensure it's up-to-date and properly formatted.
- For any issues, refer to the project documentation or refresh the app.
- For technical details, please visit the project repository.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr>
<p class="footer">
Developed for a midterm project on data-driven financial insights. Explore the full documentation and app functionality for a comprehensive analysis.
</p>
""", unsafe_allow_html=True)

# GitHub Link
st.sidebar.markdown("""
---
**Created by [Bhavya Chawla](https://github.com/bhavya1005)** 
""")