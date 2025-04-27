# credit-card-fraud-detector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score)
from imblearn.over_sampling import SMOTE

# Step 3: Generate realistic synthetic financial data
X, y = make_classification(
    n_samples=50000,
    n_features=10,
    n_informative=7,
    n_redundant=3,
    n_classes=2,
    weights=[0.99, 0.01],  # Class imbalance
    random_state=42
)

# Create realistic feature names
features = [
    'Transaction_Amount', 
    'Time_Since_Last_Transaction',
    'Purchase_Frequency',
    'Avg_Transaction_Value',
    'Card_Usage_Pattern',
    'Geo_Location_Risk',
    'Device_Trust_Score',
    'IP_Address_Risk',
    'User_Behavior_Score',
    'Time_Of_Day'
]

# Generate realistic dates and categories
timestamps = pd.date_range(start='2022-01-01', end='2023-01-01', periods=50000)
merchant_categories = np.random.choice(['Retail', 'Travel', 'Entertainment', 
                                      'Digital_Services', 'Fuel'], 50000)

df = pd.DataFrame(X, columns=features)
df['Timestamp'] = timestamps
df['Merchant_Category'] = merchant_categories
df['is_fraud'] = y

# Step 4: Enhanced visualizations for fraud detection
plt.figure(figsize=(20, 15))

# 1. Fraud Class Distribution
plt.subplot(3, 3, 1)
sns.countplot(x='is_fraud', data=df)
plt.title('Fraud Class Distribution')

# 2. Transaction Amount vs Fraud
plt.subplot(3, 3, 2)
sns.boxplot(x='is_fraud', y='Transaction_Amount', data=df)
plt.title('Transaction Amount by Fraud Status')

# 3. Time Pattern Analysis
plt.subplot(3, 3, 3)
sns.scatterplot(x='Time_Of_Day', y='Transaction_Amount', hue='is_fraud', 
                data=df.sample(1000), alpha=0.6)
plt.title('Time vs Amount Fraud Pattern')

# 4. Correlation Heatmap
plt.subplot(3, 3, 4)
sns.heatmap(df[features].corr(), annot=False, cmap='coolwarm', mask=np.triu(np.ones_like(df[features].corr())))
plt.title('Feature Correlation Matrix')

# 5. Merchant Category Risk
plt.subplot(3, 3, 5)
df.groupby('Merchant_Category')['is_fraud'].mean().sort_values().plot(kind='bar')
plt.title('Fraud Probability by Merchant Category')

# 6. Transaction Frequency Over Time
plt.subplot(3, 3, 6)
df.set_index('Timestamp')['Transaction_Amount'].resample('W').count().plot()
plt.title('Weekly Transaction Frequency')

# 7. Anomaly Detection (Isolation Forest)
plt.subplot(3, 3, 7)
iso = IsolationForest(contamination=0.01)
df['anomaly_score'] = iso.fit_predict(df[features])
sns.boxplot(x='is_fraud', y='anomaly_score', data=df)
plt.title('Anomaly Scores vs Fraud Status')

# 8. Feature Importance Preview
plt.subplot(3, 3, 8)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(df[features], df['is_fraud'])
sns.barplot(x=rf.feature_importances_, y=features)
plt.title('Feature Importance (Random Forest)')

plt.tight_layout()
plt.show()

# Step 5: Advanced preprocessing
# Feature engineering
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

X = df.drop(['is_fraud', 'Timestamp'], axis=1)
X = pd.get_dummies(X, columns=['Merchant_Category'])

# Handle class imbalance
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_res, y_res = smote.fit_resample(X, df['is_fraud'])

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Initialize models with fraud detection parameters
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced_subsample'),
    'XGBoost': XGBClassifier(scale_pos_weight=99),
    'LightGBM': LGBMClassifier(class_weight='balanced'),
    'Isolation Forest': IsolationForest(contamination=0.01)
}

# Step 7: Enhanced model evaluation
for name, model in models.items():
    if name == 'Isolation Forest':
        y_pred = model.fit_predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert anomaly scores
        print(f'\n{name} Detection Results:')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'\n{name} Performance:')
        print(f'ROC AUC: {roc_auc_score(y_test, y_pred):.4f}')
    
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

# Step 8: Feature importance analysis
plt.figure(figsize=(12, 8))
fi = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=fi.head(10))
plt.title('Top 10 Fraud Indicators')
plt.show()
