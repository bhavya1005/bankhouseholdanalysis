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
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Raleway&display=swap');

        html, body, [class*="css"] {
            font-family: 'Raleway', sans-serif;
        }

        .title {
            text-align: center;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 50px;
            color: #003366;
        }

        .section-title {
            font-size: 28px;
            color: #333333;
            font-weight: bold;
            margin-top: 20px;
        }

        .subsection {
            font-size: 18px;
            color: #555555;
            margin-top: 10px;
            line-height: 1.6;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
        }

        .insights-box {
            background-color: #e6f7ff;
            border-left: 4px solid #007acc;
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
    <h1 class="title">📊 Final Results and Key Insights</h1>
""", unsafe_allow_html=True)

#Introduction
st.markdown("""
<div style="background-color: #f0f8ff; border-left: 6px solid #003366; padding: 20px; border-radius: 10px;">
<h2 style="text-align: center; font-family: 'Bebas Neue', sans-serif; color: #003366; font-size: 36px;">
    Welcome to the Final Results and Insights Page
</h2>
<p style="text-align: justify; font-family: Raleway, sans-serif; color: #333333; font-size: 16px;">
    This page serves as the culmination of the <strong>Bank Household Analysis Dashboard</strong>, 
    bringing together the most critical findings and actionable insights derived from our analysis. 
    By exploring predictive models, household-level trends, and classification outcomes, 
    it provides a comprehensive understanding of how data-driven strategies can enhance decision-making for financial institutions.
</p>
<p style="text-align: justify; font-family: Raleway, sans-serif; color: #333333; font-size: 16px;">
    <strong>Here’s what this section offers:</strong>
    <ul style="font-family: Raleway, sans-serif; color: #003366; font-size: 16px; line-height: 1.8;">
        <li><strong>Key Metrics Overview:</strong> A quick glance at the performance metrics of the models, highlighting strengths and areas for improvement.</li>
        <li><strong>Insights from Analysis:</strong> Detailed observations from the regression and classification models, shedding light on income trends and creditworthiness.</li>
        <li><strong>Actionable Recommendations:</strong> Practical strategies for banks to leverage these insights effectively.</li>
        <li><strong>Challenges and Future Directions:</strong> A roadmap for overcoming limitations and enhancing future analyses.</li>
    </ul>
</p>
<p style="text-align: justify; font-family: Raleway, sans-serif; color: #333333; font-size: 16px;">
    Dive into the sections below to uncover how the analysis can drive smarter financial strategies, 
    streamline processes, and create impactful outcomes for households and institutions alike.
</p>
</div>
""", unsafe_allow_html=True)


# Section 1: Key Metrics Overview
st.markdown("<div class='section-title'>1️⃣ Key Metrics Overview</div>", unsafe_allow_html=True)

# Enhanced Metric Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background-color:#e6f7ff; border-radius:10px; padding:15px; text-align:center; border-left:5px solid #007acc;">
        <h2 style="color:#003366;">📈 R² Score (Regression)</h2>
        <p style="font-size:22px; color:#333333; margin:5px 0;">0.85</p>
        <p style="color:#007acc; font-size:18px;">Strong Predictor</p>
        <p style="color:#555555; font-size:14px;">Explains 85% of the variance in household income using demographic and financial features.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color:#fff4e6; border-radius:10px; padding:15px; text-align:center; border-left:5px solid #ffaa00;">
        <h2 style="color:#cc5500;">📉 RMSE (Regression)</h2>
        <p style="font-size:22px; color:#333333; margin:5px 0;">102,702</p>
        <p style="color:#ffaa00; font-size:18px;">Real-World Acceptable</p>
        <p style="color:#555555; font-size:14px;">Reflects challenges in extreme income predictions, yet aligns with real-world scenarios.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background-color:#e6ffe6; border-radius:10px; padding:15px; text-align:center; border-left:5px solid #00cc66;">
        <h2 style="color:#004d00;">✅ Classification Accuracy</h2>
        <p style="font-size:22px; color:#333333; margin:5px 0;">88%</p>
        <p style="color:#00cc66; font-size:18px;">High Precision Achieved</p>
        <p style="color:#555555; font-size:14px;">Reliable in identifying high-creditworthiness households with precision and recall exceeding 85%.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="subsection">
These metrics provide a snapshot of the model's performance. While the regression model shows strong explanatory power, the classification model excels in predicting high-creditworthiness households. The RMSE highlights areas for improvement in handling outliers and extreme income values.
</div>
""", unsafe_allow_html=True)

# Section 2: Key Insights with Expanders
st.markdown("<div class='section-title'>2️⃣ Key Insights from Analysis</div>", unsafe_allow_html=True)

# Regression Insights
with st.expander("📈 Regression Insights", expanded=True):
    st.markdown("""
    <div class="insights-box">
    <ul>
        <li><b>Predicting Total Income:</b> Household size, average credit score, and average age emerged as significant predictors of income.</li>
        <li><b>Key Patterns:</b> Larger households generally showed higher total income but lower income per member, revealing a strain on resources.</li>
        <li><b>Future Improvements:</b> Including additional features like education or employment type could refine income predictions.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Classification Insights
with st.expander("🔍 Classification Insights", expanded=True):
    st.markdown("""
    <div class="insights-box">
    <ul>
        <li><b>High Creditworthiness Prediction:</b> Households with a credit score ≥ 700 were effectively classified as highly creditworthy.</li>
        <li><b>Model Performance:</b> Achieved high precision and recall scores, ensuring reliable identification of low-risk households.</li>
        <li><b>Key Observations:</b> False positives indicate the need for improved feature engineering to minimize risk misclassification.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Section 3: Recommendations
st.markdown("<div class='section-title'>3️⃣ Actionable Recommendations</div>", unsafe_allow_html=True)
st.markdown("""
<div class="insights-box">
<ul>
    <li><b>Premium Products:</b> Target high-income households for investment opportunities and premium credit products.</li>
    <li><b>Loan Automation:</b> Use creditworthiness predictions to streamline loan approvals and minimize default risks.</li>
    <li><b>Enhanced Data Collection:</b> Address gaps in key features like income and credit score for better model performance.</li>
    <li><b>Outlier Analysis:</b> Investigate income prediction outliers to uncover untapped business opportunities or anomalies.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Section 4: Challenges and Future Enhancements
st.markdown("<div class='section-title'>4️⃣ Challenges and Future Enhancements</div>", unsafe_allow_html=True)
st.markdown("""
<div class="conclusion-box">
<h3>📌 Challenges</h3>
<ul>
    <li>Handling missing data introduced potential biases in key features like income and credit scores.</li>
    <li>State-level data imbalances affected the generalizability of certain insights.</li>
</ul>

<h3>🚀 Future Enhancements</h3>
<ul>
    <li><b>Real-Time Analytics:</b> Incorporate dynamic data feeds for up-to-date analysis and predictions.</li>
    <li><b>Advanced Models:</b> Experiment with ensemble methods like Random Forest or XGBoost to enhance accuracy.</li>
    <li><b>Behavioral Insights:</b> Add behavioral data such as spending patterns to improve creditworthiness prediction.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Section 5: Conclusion
st.markdown("""
<div style='background-color: #e8f5e9; border-left: 6px solid #2e7d32; padding: 20px; border-radius: 10px;'>
    <h2 style='color: #2e7d32; font-family: Raleway, sans-serif;'>📌 Conclusion</h2>
    <p style='font-size: 16px; color: #4caf50; font-family: Raleway, sans-serif;'>
        The <b>Bank Household Analysis Dashboard</b> has showcased the transformative potential of data-driven approaches 
        in addressing critical challenges in the financial sector. By leveraging predictive models, we successfully 
        analyzed household-level trends, forecasted income patterns, and assessed creditworthiness with actionable precision.
    </p>
    <p style='font-size: 16px; color: #4caf50; font-family: Raleway, sans-serif;'>
        <b>Key Achievements:</b>
    </p>
    <ul style='font-size: 16px; color: #388e3c; font-family: Raleway, sans-serif;'>
        <li>Developed a regression framework to predict household income with substantial explanatory power.</li>
        <li>Built a classification model capable of accurately identifying high-creditworthiness households, 
        enabling better risk management strategies.</li>
        <li>Unveiled meaningful correlations between demographic and financial attributes, highlighting key drivers 
        of household financial behavior.</li>
    </ul>
    <p style='font-size: 16px; color: #4caf50; font-family: Raleway, sans-serif;'>
        Despite challenges such as data imbalances and inherent uncertainties in predictions, this project sets a 
        <strong>solid foundation</strong> for future enhancements. By integrating real-world data, incorporating 
        advanced modeling techniques, and expanding feature sets, the framework can evolve into a robust decision-support system for banks.
    </p>
    <p style='font-size: 16px; color: #4caf50; font-family: Raleway, sans-serif;'>
        Ultimately, this project demonstrates how thoughtful analysis, combined with machine learning, 
        can empower financial institutions to optimize services, reduce risks, and foster stronger customer relationships.
    </p>
</div>
""", unsafe_allow_html=True)

