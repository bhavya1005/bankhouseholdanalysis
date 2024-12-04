# **Bank Household Analysis Dashboard**

The **Bank Household Analysis Dashboard** is an interactive application designed to analyze household-level financial data, empowering banks and financial institutions with actionable insights. By aggregating and analyzing customer demographics, financial metrics, and household patterns, this dashboard enables tailored financial strategies, accurate risk assessments, and effective customer segmentation.

Key features include:
- **Household Detection**: Group customers based on shared attributes like last names and addresses.
- **State-wise Income Distributions**: Analyze income variations across U.S. states.
- **Regression Models for Income Prediction**: Predict household income using demographic and financial attributes.
- **Classification Models for Creditworthiness Evaluation**: Identify high-creditworthiness households with precision.

### **Built with Streamlit**
This dashboard leverages **Streamlit** for a seamless user experience, integrating:
- **Pandas** for robust data manipulation and processing.
- **Plotly** for dynamic, interactive visualizations.
- **XGBoost** and other machine learning models for predictive analysis.
- **Faker** for generating synthetic data to replicate real-world financial datasets.

### **Project Highlights**
1. **Exploratory Data Analysis (EDA)**: 
   - Interactive visualizations uncover patterns in customer demographics, income, credit scores, and household sizes.
   - Heatmaps, bar charts, and scatter plots illustrate correlations and key trends.

2. **Advanced Feature Engineering**:
   - Aggregated household-level metrics (e.g., total income, average age, credit scores).
   - Created new features like "Income per Member" and "High Creditworthiness."

3. **Machine Learning Models**:
   - **Ridge Regression**: Used for predicting total household income with strong explanatory power (R² = 0.85).
   - **Logistic Regression**: Effectively classifies households into high or low creditworthiness groups with high precision and recall.

4. **Real-world Application**:
   - Mimics real-world challenges by introducing 20% artificial missingness and performing imputation.
   - Provides insights into household financial behaviors to support decision-making in the banking sector.

### **Live Dashboard**
Explore the live app here: [**Bank Household Analysis Dashboard**](https://finalprojectcmse830bhavya.streamlit.app/)

### **Key Use Cases**
- **Customer Segmentation**: Tailor financial products to specific customer groups based on household metrics.
- **Credit Risk Assessment**: Identify low-risk households for loan approvals or premium credit products.
- **Data-driven Strategy Design**: Inform marketing and risk management strategies with household-level insights.

### **How to Navigate the Dashboard**
The dashboard is divided into the following sections:
1. **Introduction**: Overview of the project and goals.
2. **Dataset Overview**: Exploration of the dataset, including summary statistics and missing data handling.
3. **Household Insights**: Analysis of aggregated household-level metrics.
4. **Predictive Analysis**: Regression and classification results with detailed visualizations.
5. **Final Results**: Comprehensive findings, challenges, and actionable recommendations.

---

### **Project Goals**
This project showcases the application of data science and machine learning in the financial sector. It aims to:
- Enable financial institutions to gain deeper insights into their customer base.
- Demonstrate the importance of household-level analysis for informed decision-making.
- Highlight the practical use of synthetic data in replicating real-world challenges.

### **Technologies Used**
- **Python**: Core language for data processing and machine learning.
- **Streamlit**: Framework for creating the interactive dashboard.
- **Pandas**: Data cleaning, transformation, and analysis.
- **Plotly**: Interactive and visually engaging charts.
- **Scikit-learn**: Machine learning models and evaluation metrics.
- **XGBoost**: Advanced predictive modeling.
- **Faker**: Synthetic data generation.

### **Future Enhancements**
- Incorporating real-world datasets for richer analysis.
- Expanding feature engineering with additional attributes like employment type or education level.
- Experimenting with advanced machine learning models (e.g., Random Forest, Neural Networks).
- Enhancing app performance with parallel processing and cloud deployment.

