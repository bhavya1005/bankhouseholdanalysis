import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier, plot_importance

# ---------------------------
# 1. App Configuration
# ---------------------------
st.set_page_config(
    page_title="Analysis and Results",
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
    <h1 class="title">📊 Analysis and Results</h1>
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

    # Impute missing values
    household_df['Average_Age'].fillna(household_df['Average_Age'].mean(), inplace=True)
    household_df['Average_Credit_Score'].fillna(household_df['Average_Credit_Score'].mean(), inplace=True)
    household_df['Total_Income'].fillna(household_df['Total_Income'].mean(), inplace=True)

    return household_df

# Load and prepare the dataset
data_filepath = "customerdataset.csv"  # Replace with your actual dataset file path
household_df = load_and_prepare_data(data_filepath)

# ---------------------------
# 4. Regression Analysis
# ---------------------------
st.subheader("Regression Analysis: Predicting Total Income")
st.markdown("""
Predicting household income is a critical task for banks to understand the financial standing of their customers. 
Income levels help banks make informed decisions about loan eligibility, financial product offerings, and risk management. 
In this analysis, we build a **regression model** to predict the total income of a household based on key demographic 
and financial characteristics, such as the average age of household members, the size of the household, and their average 
credit score.

By training this regression model, we aim to uncover patterns and relationships that can explain household income variability. 
For example, does having a larger household always mean higher income? Or do households with older members exhibit more financial stability? 
The results from this model can provide answers to these questions and offer actionable insights.
""")

# Splitting data for regression
X_reg = household_df[['Average_Age', 'Household_Size', 'Average_Credit_Score']]
y_reg = household_df['Total_Income']
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Training the regression model
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)

# Regression evaluation
y_reg_pred = reg_model.predict(X_reg_test)
reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
reg_r2 = r2_score(y_reg_test, y_reg_pred)

# Display regression metrics
st.markdown(f"""
**Model Performance:**
- **Root Mean Squared Error (RMSE):** {reg_rmse:.2f}  
- **R² Score:** {reg_r2:.2f}  

An RMSE of {reg_rmse:.2f} means that, on average, our model's predictions deviate from the actual income by this amount. 
The R² score of {reg_r2:.2f} indicates that {reg_r2*100:.2f}% of the variability in household income is explained by 
our model. A high R² score suggests that our chosen features (`Average Age`, `Household Size`, and `Average Credit Score`) 
are strongly related to income.
""")

# Regression visualization
fig_reg = px.scatter(
    x=y_reg_test, 
    y=y_reg_pred, 
    labels={'x': 'Actual Income', 'y': 'Predicted Income'},
    title="Regression Model: Actual vs Predicted Income",
)
fig_reg.add_shape(type="line", x0=0, x1=max(y_reg_test), y0=0, y1=max(y_reg_test), line=dict(dash="dash"))
st.plotly_chart(fig_reg)
st.markdown("""
The scatter plot above compares the actual household incomes (horizontal axis) with the predicted incomes 
(vertical axis). Points that fall closer to the diagonal line represent predictions that closely match 
the actual values, indicating model accuracy. Outliers, or points far from the diagonal, highlight areas 
where the model's predictions deviate significantly, offering opportunities for further model refinement.
""")

# ---------------------------
# 5. Classification Analysis
# ---------------------------
st.subheader("Classification Analysis: High Creditworthiness")
st.markdown("""
A household's creditworthiness is a crucial factor for banks in assessing financial risk. Households with high 
credit scores are more likely to repay loans on time and handle financial obligations responsibly. In this analysis, 
we use a **classification model** to predict whether a household is "Highly Creditworthy" (i.e., `Credit Score >= 700`). 

This classification task uses features such as the household's total income, the average age of its members, and 
its size. The results can help banks segment their customers into high- and low-risk groups, enabling tailored 
financial product offerings and risk mitigation strategies.
""")

# Splitting data for classification
X_clf = household_df[['Average_Age', 'Household_Size', 'Total_Income']]
y_clf = household_df['High_Creditworthiness']
X_clf_scaled = scaler.fit_transform(X_clf)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42)

# Training the classification model
clf_model = LogisticRegression()
clf_model.fit(X_clf_train, y_clf_train)

# Classification evaluation
y_clf_pred = clf_model.predict(X_clf_test)
clf_report = classification_report(y_clf_test, y_clf_pred, output_dict=True)

# Display classification metrics
st.markdown("""
**Model Performance Metrics**:
The classification model evaluates its performance using metrics like precision, recall, and F1-score. These metrics 
provide a detailed breakdown of how well the model identifies highly creditworthy households compared to less creditworthy ones.
""")
st.json(clf_report)

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
The confusion matrix above provides a clear view of the model's performance. The diagonal elements represent 
correct classifications, while the off-diagonal elements indicate misclassifications. A high number of correct 
classifications demonstrates the model's reliability, while misclassifications suggest areas for further improvement.
For example:
- **True Positives (Top-left cell):** The number of households correctly classified as highly creditworthy.
- **True Negatives (Bottom-right cell):** The number of households correctly classified as not highly creditworthy.
- **False Positives (Top-right cell):** Households incorrectly classified as highly creditworthy.
- **False Negatives (Bottom-left cell):** Households incorrectly classified as not highly creditworthy.

Understanding these metrics is vital for improving model accuracy and identifying potential biases in the data. Misclassifications, 
such as false negatives, may indicate households that are unfairly excluded from financial opportunities.
""")

# Regression Model with XGBoost
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_reg.fit(X_reg_train, y_reg_train)
y_xgb_pred = xgb_reg.predict(X_reg_test)
xgb_rmse = np.sqrt(mean_squared_error(y_reg_test, y_xgb_pred))
xgb_r2 = r2_score(y_reg_test, y_xgb_pred)
st.markdown(f"**XGBoost Regression RMSE:** {xgb_rmse:.2f}")
st.markdown(f"**XGBoost Regression R²:** {xgb_r2:.2f}")

# Classification Model with XGBoost
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_clf.fit(X_clf_train, y_clf_train)
y_xgb_clf_pred = xgb_clf.predict(X_clf_test)
st.markdown(f"**XGBoost Classification Report:**")
st.json(classification_report(y_clf_test, y_xgb_clf_pred, output_dict=True))

# Feature Importance for Regression
st.markdown("### Feature Importance: Regression")
fig, ax = plt.subplots(figsize=(10, 6))
plot_importance(xgb_reg, ax=ax)
st.pyplot(fig)

# Explanation for Regression Feature Importance
st.markdown("""
The plot above shows the feature importance for the XGBoost regression model. Feature importance indicates how much each feature contributes to the model's predictions. Features with higher importance values have a greater impact on the model's predictions. Understanding feature importance helps in identifying which features are most influential in predicting the target variable (e.g., income).
""")
st.markdown("""
The feature importance plot for the regression model highlights which variables most influence the prediction of household income:
- **Average Age**: Older households may have more established incomes, contributing significantly to predictions.
- **Household Size**: Larger households often pool multiple incomes, making this a crucial feature.
- **Average Credit Score**: A high credit score may correlate with better financial stability and higher income.

Understanding feature importance allows banks to:
1. Focus on the most impactful attributes when assessing household income.
2. Identify opportunities for further feature engineering.
""")

# Feature Importance for Classification
st.markdown("### Feature Importance: Classification")
fig, ax = plt.subplots(figsize=(10, 6))
plot_importance(xgb_clf, ax=ax)
st.pyplot(fig)

# Explanation for Classification Feature Importance
st.markdown("""
The plot above shows the feature importance for the XGBoost classification model. Similar to the regression model, feature importance in the classification model indicates how much each feature contributes to the model's ability to classify households as highly creditworthy or not. Features with higher importance values are more influential in determining the classification outcome. This information can be used to focus on the most impactful features when improving the model or making business decisions.
""")
st.markdown("""
The feature importance plot for the classification model shows the relative contribution of each feature in predicting creditworthiness:
- **Total Income**: Higher-income households tend to have better financial stability, leading to higher credit scores.
- **Average Age**: Older households may have more credit history, positively influencing creditworthiness.
- **Household Size**: Larger households might have shared financial obligations, potentially reducing individual credit risk.

These insights enable banks to develop more targeted strategies for evaluating credit risk and tailoring financial products.
""")


# Hyperparameter Tuning with RandomizedSearchCV

# Define hyperparameter grid for XGBoost Regression
param_grid_reg = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Randomized Search for XGBoost Regression
random_search_reg = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=param_grid_reg,
    n_iter=10,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=1
)
random_search_reg.fit(X_reg_train, y_reg_train)
st.markdown(f"**Best Parameters for XGBoost Regression (Randomized Search):** {random_search_reg.best_params_}")

# Define hyperparameter grid for XGBoost Classification
param_grid_clf = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Randomized Search for XGBoost Classification
random_search_clf = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_distributions=param_grid_clf,
    n_iter=10,  # Number of parameter settings sampled
    scoring='accuracy',
    cv=3,
    random_state=42,
    verbose=1
)
random_search_clf.fit(X_clf_train, y_clf_train)
st.markdown(f"**Best Parameters for XGBoost Classification (Randomized Search):** {random_search_clf.best_params_}")

# ---------------------------
# 6. Key Insights
# ---------------------------
st.subheader("Key Takeaways")
st.markdown("""
### Regression Model:
- The **R² score** of {reg_r2:.2f} demonstrates that key features such as household size, average age, and average credit score 
  are strong predictors of total household income. Banks can use this insight to prioritize high-income households for premium 
  financial products.
- However, some **outliers** indicate unique or unexpected households that the model struggles to predict accurately. These cases 
  could represent special financial circumstances or irregular data patterns.

### Classification Model:
- The **precision** and **recall** scores suggest that the model performs well in identifying highly creditworthy households. 
  These insights can help banks streamline loan approvals and reduce the risk of defaults.
- The **confusion matrix** highlights areas for improvement. For instance, reducing **false positives** (households incorrectly 
  classified as creditworthy) can help banks avoid financial risks.

### Overall Recommendations:
1. **Focus on High-Income Households:** Banks can target households with high predicted incomes for exclusive financial products, 
   such as investment plans or premium credit cards.
2. **Improve Creditworthiness Predictions:** Addressing false positives in the classification model can refine risk management strategies.
3. **Explore Outliers:** Investigate households that deviate from expected patterns to uncover untapped opportunities or detect anomalies.
""")