import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, XGBClassifier, plot_importance
from sklearn.inspection import permutation_importance
import shap

# ---------------------------
# 1. App Configuration
# ---------------------------
st.set_page_config(
    page_title="Predictive Analysis for Bank Households",
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
    <h1 style='text-align: center; font-family: Bebas Neue, sans-serif; color: #003366;'>Exploring Financial Patterns Through Machine Learning Models</h1>
    
""", unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-family: Raleway, sans-serif; color: #003366;">
    Welcome to the analysis page, where we delve into the financial data of households to uncover meaningful insights using predictive models. This page explores how <strong>machine learning</strong> can provide valuable answers to critical questions, helping banks and financial institutions make informed decisions.
</p>
<p style="text-align: left; font-family: Raleway, sans-serif; color: #003366;">
    The analysis is divided into two main sections:
</p>
<ul style="font-family: Raleway, sans-serif; color: #003366;">
    <li><strong>Regression Analysis</strong>: Focused on predicting total household income based on demographic and financial characteristics.</li>
    <li><strong>Classification Analysis</strong>: Assessing household creditworthiness to distinguish between high- and low-risk groups.</li>
</ul>
<p style="font-family: Raleway, sans-serif; color: #003366;">
    Through comprehensive visualizations, performance metrics, and in-depth explanations, we demonstrate the practical applications of models like <strong>Linear Regression</strong>, <strong>Logistic Regression</strong>, and <strong>XGBoost</strong>. Each analysis highlights not only the accuracy of predictions but also the insights derived from key features such as:
</p>
<ul style="font-family: Raleway, sans-serif; color: #003366;">
    <li><strong>Average Age</strong>: How age correlates with income stability and credit risk.</li>
    <li><strong>Household Size</strong>: The role of larger families in pooling financial resources.</li>
    <li><strong>Credit Score</strong>: A critical factor influencing financial reliability.</li>
</ul>
<p style="font-family: Raleway, sans-serif; color: #003366;">
    This page serves as a complete guide to understanding how predictive analytics can transform raw data into actionable insights, tailored for both technical and business audiences. Explore each section to see the models in action, review key findings, and gain a deeper appreciation for the power of data-driven financial analysis.
</p>
""", unsafe_allow_html=True)
# ---------------------------
# 3. Load Dataset and Process Data
# ---------------------------
@st.cache_data
def load_and_prepare_data(filepath):
    """
    Load the dataset and prepare it for analysis.

    Steps:
    1. Group customers into households using last name, address, and state.
    2. Compute household-level statistics like total income, average credit score, and size.
    3. Impute missing values to ensure the dataset is ready for modeling.
    """
    df = pd.read_csv(filepath)

    # Group by household and calculate aggregated values
    household_df = df.groupby(['Last_Name', 'Address', 'State'], as_index=False).agg({
        'Income': 'sum',  # Sum up income for all household members
        'Credit_Score': 'mean',  # Average credit score for the household
        'Age': 'mean',  # Average age of household members
        'First_Name': 'count',  # Count the number of household members
    }).rename(columns={
        'Income': 'Total_Income',
        'Credit_Score': 'Average_Credit_Score',
        'Age': 'Average_Age',
        'First_Name': 'Household_Size',
    })

    # Filter for households with more than 1 member
    household_df = household_df[household_df['Household_Size'] > 1]

    # Feature Engineering
    household_df['Income_Per_Member'] = household_df['Total_Income'] / household_df['Household_Size']
    household_df['High_Creditworthiness'] = (household_df['Average_Credit_Score'] >= 700).astype(int)

    # Impute missing values for numeric columns only
    numeric_cols = household_df.select_dtypes(include=[np.number]).columns
    household_df[numeric_cols] = household_df[numeric_cols].fillna(household_df[numeric_cols].mean())

    return household_df

data_filepath = "customerdataset.csv"  # Replace with your actual dataset file path
household_df = load_and_prepare_data(data_filepath)

# ---------------------------
# 4. Feature Correlation Heatmap
# ---------------------------
st.subheader("Feature Correlation Heatmap")
st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'> 
The correlation heatmap is a powerful visualization tool used to understand relationships between multiple features in a dataset. It helps identify how strongly two variables are related to one another, providing insights into potential dependencies and patterns. For this analysis, the heatmap allows us to explore key household attributes such as income, creditworthiness, household size, and more, uncovering which features influence each other. This is crucial for building effective predictive models and understanding the underlying dynamics in financial data. 
</p>
""", unsafe_allow_html=True)

# Ensure only numeric columns are included in the correlation matrix
numeric_cols = household_df.select_dtypes(include=[np.number]).columns
correlation_matrix = household_df[numeric_cols].corr()

# Plot the correlation heatmap
fig_corr = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="Viridis")
st.plotly_chart(fig_corr)

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'> 
The heatmap above presents the pairwise correlation between various features in the dataset:
<ul>
    <li><strong>Color Coding:</strong> The scale on the right represents the correlation values, ranging from -1 (strong negative correlation) to 1 (strong positive correlation). Yellow indicates a strong positive correlation, while dark purple represents weak or negative correlations.</li>
    <li><strong>Key Insights:</strong>
        <ul>
            <li><strong>Total Income and Household Size:</strong> There is a strong positive correlation (~0.79), indicating that larger households tend to have higher combined incomes due to multiple earning members.</li>
            <li><strong>Income Per Member and Total Income:</strong> A significant positive correlation (~0.57) suggests that higher overall income often results in higher income per individual.</li>
            <li><strong>Average Credit Score and High Creditworthiness:</strong> These features exhibit a clear positive correlation (~0.62), confirming that households with higher average credit scores are more likely to be classified as highly creditworthy.</li>
            <li><strong>Weak or Negative Correlations:</strong> Some features, such as <strong>Household Size</strong> and <strong>Creditworthiness</strong>, have weak or slightly negative correlations, reflecting limited or no direct relationship.</li>
        </ul>
    </li>
</ul>
This heatmap provides a foundation for identifying influential variables, aiding feature selection, and ensuring that our predictive models focus on the most impactful attributes.
</p>
""", unsafe_allow_html=True)

# ---------------------------
# 5. Regression Analysis
# ---------------------------
st.subheader("Regression Analysis: Predicting Total Income")
X_reg = household_df[['Average_Age', 'Household_Size', 'Average_Credit_Score']]
y_reg = household_df['Total_Income']

scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Ridge Regression with Hyperparameter Tuning
ridge_params = {'alpha': [0.1, 1.0, 10.0]}
ridge_grid = GridSearchCV(Ridge(), param_grid=ridge_params, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(X_reg_train, y_reg_train)
best_ridge = ridge_grid.best_estimator_

y_ridge_pred = best_ridge.predict(X_reg_test)
ridge_rmse = np.sqrt(mean_squared_error(y_reg_test, y_ridge_pred))
ridge_r2 = r2_score(y_reg_test, y_ridge_pred)

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'>
Regression analysis is a statistical method used to identify relationships between variables and predict a target variable based on input features. Ridge Regression, a variation of linear regression, incorporates regularization by adding a penalty term to the regression coefficients, making it particularly effective for handling datasets with multicollinearity or correlated features. In this analysis, we use Ridge Regression to predict household income while ensuring the model remains robust and avoids overfitting, even when some features are strongly correlated.
</p>
""", unsafe_allow_html=True)

st.markdown(f"""
**Ridge Regression Performance:**
- **Best Alpha:** {ridge_grid.best_params_['alpha']}
- **Root Mean Squared Error (RMSE):** {ridge_rmse:.2f}  
- **R² Score:** {ridge_r2:.2f}
""")

# Regression visualization
fig_reg = px.scatter(
    x=y_reg_test,
    y=y_ridge_pred,
    labels={'x': 'Actual Income', 'y': 'Predicted Income'},
    title="Ridge Regression Model: Actual vs Predicted Income",
)
fig_reg.add_shape(type="line", x0=0, x1=max(y_reg_test), y0=0, y1=max(y_reg_test), line=dict(dash="dash"))
st.plotly_chart(fig_reg)

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'>
The scatter plot above represents the performance of the Ridge Regression model in predicting total household income. Ridge Regression is particularly effective for handling multicollinearity and prevents overfitting by adding a penalty term to the regression coefficients. This approach is valuable in datasets where features might be correlated, ensuring that the model remains robust and generalizes well.
</p>
""", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'>
The plot provides a comparison between the actual household incomes (x-axis) and the predicted incomes (y-axis) produced by the Ridge Regression model:
<ul>
    <li><strong>Diagonal Line:</strong> The dashed line represents perfect predictions, where actual and predicted values are equal. Points closer to this line indicate higher accuracy of the model's predictions.</li>
    <li><strong>Clustering:</strong> A majority of the data points are tightly clustered along the diagonal, reflecting the model's ability to predict total income with reasonable accuracy. However, some outliers can be observed, indicating instances where the model struggles to match the actual values.</li>
    <li><strong>Performance Metrics:</strong>
        <ul>
            <li><strong>Root Mean Squared Error (RMSE):</strong> 102702.44 – This metric indicates the average deviation of the predicted income from the actual income.</li>
            <li><strong>R² Score:</strong> 0.68 – This suggests that 68% of the variability in household income is explained by the model's features, highlighting a decent but improvable performance.</li>
        </ul>
    </li>
</ul>
This visualization not only validates the model's predictions but also offers insights into areas for potential refinement, such as addressing the outliers and improving feature selection or hyperparameter tuning.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'> 
The model's **Root Mean Squared Error (RMSE)** is **102702.44**, which reflects the average deviation of predicted household incomes from the actual values. While this error might seem high, it is important to consider the following factors:
<ul>
    <li>The variability in household incomes is substantial, with incomes ranging widely across different households.</li>
    <li>The RMSE is measured in absolute terms and does not fully reflect the model's ability to explain trends and patterns in the data.</li>
</ul>
Despite the high RMSE, the model demonstrates a strong ability to explain the variability in income with an **R² score of 0.68**, indicating that it captures 68% of the variance in household income. Furthermore, the feature importance analysis reveals key drivers of income, such as household size and average credit score, providing valuable insights for financial decision-making.
</p>
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'> 
By focusing on actionable insights rather than precise predictions, this model serves as a valuable tool for identifying income-related patterns and prioritizing households for targeted financial strategies. While future iterations can work on reducing the RMSE, the current model effectively supports decision-making processes with meaningful, data-driven insights.
</p>
""", unsafe_allow_html=True)

# ---------------------------
# 6. Classification Analysis
# ---------------------------
st.subheader("Classification Analysis: High Creditworthiness")
st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'>
Classification analysis involves predicting the category or class of a target variable based on a set of features. In this context, we classify households as either "Highly Creditworthy" (1) or "Not Highly Creditworthy" (0) based on attributes like average age, household size, and total income. Evaluating the performance of the classification model is crucial, and metrics like precision, recall, F1-score, and accuracy provide insights into the model's reliability and effectiveness.
</p>
""", unsafe_allow_html=True)

X_clf = household_df[['Average_Age', 'Household_Size', 'Total_Income']]
y_clf = household_df['High_Creditworthiness']

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_clf_balanced, y_clf_balanced = smote.fit_resample(X_clf, y_clf)

X_clf_scaled = scaler.fit_transform(X_clf_balanced)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf_scaled, y_clf_balanced, test_size=0.2, random_state=42)

# Train Logistic Regression
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_clf_train, y_clf_train)

# Evaluate Logistic Regression
y_clf_pred = log_clf.predict(X_clf_test)
clf_report = classification_report(y_clf_test, y_clf_pred, output_dict=True)
st.json(clf_report)

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'>
The classification report above provides detailed metrics for both classes (0 and 1) and overall model performance:
- **Precision**: The proportion of correctly predicted instances among the total predicted for each class. For instance, a precision of ~0.75 for class 0 indicates that 75% of households classified as "Not Highly Creditworthy" were correct.
- **Recall**: The proportion of actual instances correctly predicted. A recall of ~0.72 for class 1 shows that the model identified 72% of all "Highly Creditworthy" households.
- **F1-Score**: The harmonic mean of precision and recall, balancing false positives and false negatives.
- **Overall Accuracy**: ~0.74, showing that 74% of the total predictions are correct.
- **Macro and Weighted Averages**: The macro average calculates metrics equally across classes, while the weighted average accounts for class imbalances.

These metrics highlight that while the model performs well overall, there is room for improvement in balancing precision and recall, particularly for class 1 (Highly Creditworthy).
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'>
The confusion matrix is a fundamental tool in classification analysis, offering a clear overview of how well a model performs in distinguishing between different classes. It displays the number of correct and incorrect predictions for each class, helping identify potential strengths and weaknesses of the model. For this analysis, the confusion matrix evaluates the classification of households as "Highly Creditworthy" (1) or "Not Highly Creditworthy" (0).
</p>
""", unsafe_allow_html=True)

# Confusion matrix
conf_matrix = confusion_matrix(y_clf_test, y_clf_pred)
fig_conf_matrix = px.imshow(
    conf_matrix,
    text_auto=True,
    color_continuous_scale="Blues",
    labels={'x': 'Predicted', 'y': 'Actual'},
    title="Confusion Matrix"
)
st.plotly_chart(fig_conf_matrix)

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'>
The confusion matrix above breaks down the model's predictions as follows:
- **True Positives (Bottom-right cell: 352)**: Households correctly classified as "Highly Creditworthy."
- **True Negatives (Top-left cell: 342)**: Households correctly classified as "Not Highly Creditworthy."
- **False Positives (Top-right cell: 135)**: Households incorrectly classified as "Highly Creditworthy," indicating overestimation of creditworthiness.
- **False Negatives (Bottom-left cell: 112)**: Households incorrectly classified as "Not Highly Creditworthy," suggesting missed opportunities for identifying creditworthy households.

These metrics highlight the balance between precision (minimizing false positives) and recall (minimizing false negatives). While the model performs reasonably well, reducing false negatives could ensure that deserving households are not overlooked, while minimizing false positives helps avoid undue financial risks.
</p>
""", unsafe_allow_html=True)

# ---------------------------
# 7. Feature Importance Using SHAP
# ---------------------------
st.subheader("Feature Importance Using SHAP")

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'> 
SHAP (SHapley Additive exPlanations) is a machine learning interpretability tool that breaks down model predictions by quantifying the contribution of each feature to an individual prediction. It provides transparency and ensures fairness by helping us understand the driving factors behind a model's decisions. For our analysis, SHAP is used to highlight the most influential factors in predicting household creditworthiness, ensuring that the model's decisions align with domain knowledge and business logic.
</p>
""", unsafe_allow_html=True)

explainer = shap.Explainer(log_clf, X_clf_train)
shap_values = explainer(X_clf_train)
shap.summary_plot(shap_values, X_clf_balanced, plot_type="bar", show=False)
st.pyplot(plt.gcf())

st.markdown("""
<p style='text-align: justify; font-family: Raleway, sans-serif; color: #003366;'> 
The bar chart above displays the <strong>average absolute SHAP values</strong> for each feature, providing insights into the relative importance of predictors in the classification model. Here’s what the results show:
<ul>
    <li><strong>Total Income:</strong> This is the most impactful feature, indicating that higher total income significantly improves the likelihood of being classified as highly creditworthy. This aligns with expectations, as income directly affects financial stability.</li>
    <li><strong>Average Age:</strong> The second most significant feature, reflecting that older households tend to be more financially stable and responsible, positively influencing creditworthiness predictions.</li>
    <li><strong>Household Size:</strong> While less impactful compared to income and age, household size plays a role in understanding financial dynamics. Larger households may share resources and responsibilities, subtly affecting credit risk.</li>
</ul>
These results confirm the importance of income, age, and household size in assessing household creditworthiness, providing actionable insights for banks to tailor financial products and risk assessments.
</p>
""", unsafe_allow_html=True)

# GitHub Link
st.sidebar.markdown("""
---
**Created by [Bhavya Chawla](https://github.com/bhavya1005)** 
""")