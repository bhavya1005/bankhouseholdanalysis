import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Bank Household Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for a polished look
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
            color: #003366; /* Dark blue */
            margin-bottom: 20px;
        }

        /* Subtitle Styling */
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #555555; /* Subtle gray */
            margin-bottom: 40px;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                to bottom,
                #e6f0ff, /* Soft Blue */
                #f8e7e7  /* Soft Pink */
            );
            color: #003366; /* Dark blue text */
            border-right: 2px solid #cccccc; /* Subtle border for sidebar */
        }

        /* Instructions Styling */
        .instructions {
            font-size: 18px;
            color: #333333; /* Slightly darker gray */
            margin-bottom: 20px;
        }

        /* Additional Information Styling */
        .additional-info {
            font-size: 16px;
            color: #666666; /* Light gray for secondary text */
            margin-top: 40px;
        }

        /* Footer Styling */
        hr {
            border: none;
            border-top: 1px solid #cccccc;
            margin: 40px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h1 class='title'>Bank Household Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Explore how banks analyze household data for informed decision-making.</p>", unsafe_allow_html=True)

# Introduction Section
st.markdown("""
Welcome to the **Bank Household Analysis Dashboard**! This application provides a comprehensive overview of household-level customer data, showcasing advanced visualizations, analysis, and modeling techniques. By exploring the data, banks can gain insights to make better decisions about product offerings, customer segmentation, and financial risk management.

### How to Use the Dashboard
To navigate through the dashboard, use the **sidebar** on the left. The app is divided into the following sections:

1. **Introduction**: An overview of the project and its objectives.
2. **Dataset Overview**: Examine the raw dataset and understand its structure.
3. **Household Insights**: Analyze households based on aggregated metrics.
4. **Analysis and Results**: View advanced modeling, results, and key takeaways.

### Key Features
Through this dashboard, you will:
- Understand household patterns through data aggregation and analysis.
- Gain insights into financial behaviors and creditworthiness.
- Explore advanced regression and classification models to predict income and credit risk.
""")

# Instructions for Navigation
st.markdown("<p class='instructions'>Navigate through the sidebar to explore the different sections of the dashboard.</p>", unsafe_allow_html=True)

# Additional Information Section
st.markdown("""
<div class="additional-info">
<b>Notes:</b>
- All sections are interconnected and rely on the same dataset.
- If you experience issues, refresh the app or verify the dataset location.
- For further details about the methodology, visit the project documentation in the repository.
</div>
""", unsafe_allow_html=True)

# Footer Message
st.markdown("""
<hr>
<p style="text-align: center; font-size: 14px; color: #888888;">
This dashboard was developed as part of a data analysis project to explore household-level financial insights.
</p>
""", unsafe_allow_html=True)
