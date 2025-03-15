import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib



# @st.cache_data
def load_data():
    path = 'https://raw.githubusercontent.com/archanags001/CustomerChurnPredictor/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    data = pd.read_csv(path)
    return data
def process_senior_citizen(df):
    try:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    except Exception as e:
        st.error(e)
    return df


# Process TotalCharges column
def process_total_charges(df):
    try:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
    except Exception as e:
        st.error(e)
    return df
# Handles missing values in a DataFrame.
def na_value_handling(df):
    try:
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Impute numerical columns with mean
        numerical_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

        # Impute categorical columns with most frequent value
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

        # Specific handling for 'SeniorCitizen' column
        if 'SeniorCitizen' in df.columns:
            # Assuming SeniorCitizen is numerical (binary 0 or 1)
            df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0)  # Filling NaN with 0 for SeniorCitizen

    except Exception as e:
        st.error(f"An error occurred during missing value handling: {e}")
    return df

# Encode categorical variables
def encode_categorical(df):
    try:
        if 'customerID' in df.columns:
            return pd.get_dummies(df.drop(columns=['customerID']), drop_first=True)
        else:
            return pd.get_dummies(df, drop_first=True)
    except Exception as e:
        st.error(e)
    return df



# Combine all transformations using pipe

def process_data(df):
    df_processed = (df
                    .pipe(process_senior_citizen)
                    .pipe(process_total_charges)
                    .pipe(na_value_handling)
                    .pipe(encode_categorical)
                    )

    print(df_processed.head())

    # Define features and target variable
    X = df_processed.drop(columns=['Churn_Yes'])
    y = df_processed['Churn_Yes']

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_sampled, y_sampled = smote.fit_resample(X, y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42,
                                                        stratify=y_sampled)

    return X_train, X_test, y_train, y_test, X_test.columns
def train_models(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    parameters = [
        {'classifier__penalty': ['l1', 'l2', 'elasticnet'], 'classifier__solver': ['liblinear', 'saga']},
        {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    joblib.dump(grid_search.best_estimator_, "customer_churn_model.pkl")
    return grid_search.best_estimator_

def evaluate_models(model, X_test, y_test):
    X_test_scaled = model.named_steps['scaler'].transform(X_test)
    y_pred = model.named_steps['classifier'].predict(X_test_scaled)
    y_prob = model.named_steps['classifier'].predict_proba(X_test_scaled)[:, 1]
    st.write(classification_report(y_test, y_pred))
    st.write(f'AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}')


if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test, feature_names = process_data(data)
    log_reg = train_models(X_train, y_train, X_test, y_test)
    evaluate_models(log_reg, X_test, y_test)



