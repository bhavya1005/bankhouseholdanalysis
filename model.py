import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_filepath = "customerdataset.csv"  # Replace with your actual file path
raw_df = pd.read_csv(data_filepath)

# Prepare household data
def prepare_household_data(df):
    # Aggregate household-level data
    household_df = df.groupby(['Last_Name', 'Address', 'State'], as_index=False).agg({
        'Income': 'sum',  # Total household income
        'Credit_Score': 'mean',  # Average credit score
        'Age': 'mean',  # Average age
        'First_Name': 'count',  # Household size
    }).rename(columns={
        'Income': 'Total_Income',
        'Credit_Score': 'Average_Credit_Score',
        'Age': 'Average_Age',
        'First_Name': 'Household_Size',
    })
    # Filter households with more than one member
    household_df = household_df[household_df['Household_Size'] > 1]
    return household_df

household_df = prepare_household_data(raw_df)

# Feature Engineering
household_df['Income_Per_Member'] = household_df['Total_Income'] / household_df['Household_Size']
household_df['High_Creditworthiness'] = (household_df['Average_Credit_Score'] >= 700).astype(int)

# Splitting Features and Targets
X_reg = household_df[['Average_Age', 'Household_Size', 'Average_Credit_Score']]
y_reg = household_df['Total_Income']

X_clf = household_df[['Average_Age', 'Household_Size', 'Total_Income']]
y_clf = household_df['High_Creditworthiness']

# Standardize Features
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)
X_clf_scaled = scaler.fit_transform(X_clf)

# Split into Training and Testing Sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42)

# Regression Model
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

# Evaluate Regression Model
reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
reg_r2 = r2_score(y_reg_test, y_reg_pred)
print(f"Regression RMSE: {reg_rmse}")
print(f"Regression R²: {reg_r2}")

# Classification Model
clf_model = LogisticRegression()
clf_model.fit(X_clf_train, y_clf_train)
y_clf_pred = clf_model.predict(X_clf_test)

# Evaluate Classification Model
print("Classification Report:")
print(classification_report(y_clf_test, y_clf_pred))

conf_matrix = confusion_matrix(y_clf_test, y_clf_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
