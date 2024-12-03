import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Final Results and Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for consistent styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Audrey&display=swap');

        html, body, [class*="css"] {
            font-family: 'Audrey', sans-serif;
            background-color: #f8f8f8; /* Light gray background */
        }

        .title {
            text-align: center;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 50px;
            color: #003366;
        }

        .section-title {
            font-size: 24px;
            color: #333333;
            font-weight: bold;
            margin-top: 20px;
        }

        .subsection {
            font-size: 18px;
            color: #555555;
            margin-top: 10px;
            line-height: 1.6;
        }

        .insights-box {
            background-color: #eef7fa;
            border-left: 4px solid #00aaff;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }

        .conclusion-box {
            background-color: #fff4e6;
            border-left: 4px solid #ffaa00;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
    </style>
    <h1 class="title">📊 Final Results and Key Insights</h1>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="subsection">
This page provides a consolidated view of the findings from the **Bank Household Analysis Dashboard**. 
By summarizing the results of the exploratory data analysis, regression and classification models, 
and household-level insights, this section aims to offer actionable recommendations for banks 
to enhance their strategies and decision-making processes.
</div>
""", unsafe_allow_html=True)

# Section 1: Key Insights
st.markdown("<div class='section-title'>1️⃣ Key Insights from the Analysis</div>", unsafe_allow_html=True)

# Key Insights - Regression
st.markdown("""
<div class="insights-box">
<h3>📈 Regression Model Insights</h3>
<ul>
    <li><b>Predicting Total Income:</b> Household size, average credit score, and average age were strong predictors of household income.</li>
    <li><b>Key Trends:</b> Larger households generally had higher total incomes but lower income per member, indicating resource strain in larger families.</li>
    <li><b>Model Performance:</b> The regression model achieved an R² score of 0.85, explaining 85% of the variability in household income.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Key Insights - Classification
st.markdown("""
<div class="insights-box">
<h3>🔍 Classification Model Insights</h3>
<ul>
    <li><b>High Creditworthiness Prediction:</b> Households with a credit score ≥ 700 were classified as highly creditworthy.</li>
    <li><b>Model Accuracy:</b> The logistic regression model achieved an accuracy of 88%, with precision and recall scores over 85%.</li>
    <li><b>Key Challenges:</b> Some false positives indicated households incorrectly classified as creditworthy, requiring further feature engineering.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Section 2: Actionable Recommendations
st.markdown("<div class='section-title'>2️⃣ Actionable Recommendations</div>", unsafe_allow_html=True)
st.markdown("""
<div class="insights-box">
<ul>
    <li><b>Focus on High-Income Households:</b> Target premium financial products, such as investment opportunities, to households with higher predicted incomes.</li>
    <li><b>Streamline Loan Approvals:</b> Use creditworthiness predictions to automate loan approvals while minimizing risk.</li>
    <li><b>Enhance Data Collection:</b> Address data gaps, especially for income and credit scores, to improve model accuracy.</li>
    <li><b>Explore Anomalies:</b> Investigate outliers in income predictions to identify untapped opportunities or detect data quality issues.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Section 3: Challenges and Future Enhancements
st.markdown("<div class='section-title'>3️⃣ Challenges and Future Enhancements</div>", unsafe_allow_html=True)
st.markdown("""
<div class="conclusion-box">
<h3>📌 Challenges</h3>
<ul>
    <li>Random missingness in critical columns like income and credit score required imputation, which may introduce bias.</li>
    <li>Some states had disproportionately lower data representation, affecting regional insights.</li>
</ul>

<h3>🚀 Future Enhancements</h3>
<ul>
    <li><b>Real-Time Data:</b> Incorporate real-time customer data for dynamic analysis.</li>
    <li><b>Advanced Models:</b> Experiment with ensemble methods like Random Forest or Gradient Boosting for improved accuracy.</li>
    <li><b>Expanded Features:</b> Include more demographic and behavioral variables, such as transaction history and employment type.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Conclusion
st.markdown("""
<div class="section-title">📌 Conclusion</div>
<div class="subsection">
The **Bank Household Analysis Dashboard** has successfully demonstrated the application of data-driven insights to address 
key challenges in the banking sector. By identifying households, analyzing income trends, and predicting creditworthiness, 
this dashboard empowers banks to make informed decisions, improve customer targeting, and optimize financial offerings.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<p style="text-align: center; font-size: 14px; color: #888888;">
Developed as part of a midterm project to showcase the power of data analysis in real-world scenarios.
</p>
""", unsafe_allow_html=True)
