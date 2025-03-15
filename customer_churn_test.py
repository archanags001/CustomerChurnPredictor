from customer_churn_training import *
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import requests
from io import BytesIO


def display_feature_insights(model, X_test, feature_names):
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("<h3 class='centered-header'>Feature Impact on Customer Churn Predictions</h3>", unsafe_allow_html=True)
        X_test_scaled = model.named_steps['scaler'].transform(X_test)
        explainer = shap.Explainer(model.named_steps['classifier'], X_test_scaled, feature_names=feature_names)
        shap_values = explainer(X_test_scaled)
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,max_display=13, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    with col_b:
        st.markdown("<h3 class='centered-header'>Feature for Dependence Plot</h3>", unsafe_allow_html=True)
        feature_index = st.selectbox("Select Feature for Dependence Plot", feature_names)
        st.write("")
        st.write("")
        feature_col_index = feature_names.tolist().index(feature_index)
        shap.dependence_plot(feature_col_index, shap_values.values, X_test_scaled, feature_names=feature_names, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

# Plots feature importance for a Logistic Regression model using Plotly.
def plot_feature_importance_plotly(model, feature_names):
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.named_steps['classifier'].coef_[0]
    })
    importance = importance.sort_values(by='Importance', ascending=False)

    fig = px.bar(
        importance,
        x='Feature',
        y='Importance',
        title="Feature Importance (Logistic Regression)",
        labels={'Importance': 'Coefficient Value'},
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

if __name__ == "__main__":
    st.set_page_config(layout='wide')
    st.markdown(
        """
        <style>
        .centered-header {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='text-align: center;'>Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
    # st.title("Churn Prediction Dashboard")
    data = load_data()
    X_train, X_test, y_train, y_test, feature_names = process_data(data)
    # url = "https://github.com/archanags001/CustomerChurnPredictor/blob/main/customer_churn_model.pkl"
    # log_reg = pickle.load(open('customer_churn_model.pkl', 'rb'))
    log_reg = joblib.load('customer_churn_model.pkl')

    col1, col2 = st.columns([1,2])  # Left, Center, Right

    with col1:
        st.subheader("📊 Model Performance")
        with st.container():
            evaluate_models(log_reg, X_test, y_test)

            plot_feature_importance_plotly(log_reg, X_train.columns)

    with col2:
        st.subheader("🔍 Select Test Data Option")
        option = st.radio(
            "Choose an option:",
            ["Edit/Add Data Manually", "Upload Data File", "Use Existing Data"]
        )

        if option == "Upload Data File":
            # Allow the user to upload CSV or XLSX files
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

            # Process the uploaded file if a file is uploaded
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            test_data = pd.read_csv(uploaded_file)
                        except Exception as e:
                            st.error(str(e))
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        try:
                            test_data = pd.read_excel(uploaded_file)

                        except Exception as e:
                            st.error(str(e))
                    # Ensure test_data has the same columns as data used before training
                    try:
                        col_names = list(data.columns)
                        col_names.remove('Churn')
                        test_data = test_data[col_names]
                    except Exception as e:
                        st.error(e)
                        st.info('Take a moment to inspect the test data features.')

                    if len(test_data) < 1:
                        st.warning("Test data is empty! Please add at least one row before running predictions.")
                    else:
                        st.dataframe(test_data)
                        st.session_state.test_data = test_data

                except Exception as e:
                    st.error(f"Error reading file: {e}")

        elif option == "Edit/Add Data Manually":

            drop_down_columns = ['gender','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                   'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
            test_data = pd.DataFrame()
            churn_0 = data[data['Churn'] == 'No']
            churn_1 = data[data['Churn'] == 'Yes']
            sample_data = pd.concat([churn_0[:4], churn_1[:4]], ignore_index=True)

            # Convert categorical columns to categorical type
            for col in data.columns:
                if (col in drop_down_columns) and (col not in ['customerID','Churn']) :
                    test_data[col] = sample_data[col].astype(pd.CategoricalDtype(data[col].unique()))
                elif (col not in ['customerID','Churn']) :
                    test_data[col] = sample_data[col]

            test_data['SeniorCitizen'] = test_data['SeniorCitizen'].map({'No':0, 'Yes':1})


            # Create a form to control updates
            with st.form(key="edit_form"):
                edited_data = st.data_editor(test_data, num_rows='dynamic', use_container_width=True)
                # edited_data = st.data_editor(st.session_state.test_data, num_rows='dynamic', use_container_width=True,
                #                              key="editor")

                # Submit button inside the form
                submit_button = st.form_submit_button("Submit Changes")

            # Only update session state when the form is submitted
            if submit_button:
                if len(edited_data) < 1:
                    st.warning("Test data is empty! Please add at least one row before running predictions.")
                else:
                    st.session_state.test_data = edited_data
                    st.success("Test data updated successfully!")

        else:
            test_data = pd.read_csv('https://raw.githubusercontent.com/archanags001/CustomerChurnPredictor/refs/heads/main/Test_data_no_churn.csv')

            if st.button("Submit Test Data"):
                st.dataframe(test_data)
                st.session_state.test_data = test_data

    # Retrieve df from session state
    if 'test_data' in st.session_state:
        test_data = st.session_state.test_data
        # copy_test_data = test_data.copy()
        test_data_processed = (test_data
                        .pipe(process_senior_citizen)
                        .pipe(process_total_charges)
                        .pipe(encode_categorical)
                        )
        test_data_processed = test_data_processed.reindex(columns=feature_names, fill_value=0)
        # Predict with the trained model
        X_test_new = log_reg.named_steps['scaler'].transform(test_data_processed)
        y_pred_prob = log_reg.named_steps['classifier'].predict_proba(X_test_new)[:,
                      1]  # Prediction probabilities (probability of class 1)
        y_pred = (y_pred_prob >= 0.5).astype(int)  # Convert probabilities to binary predictions
        y_pred = (y_pred >= 0.5).astype(int)

        # Add prediction probabilities and churn predictions to the original test_data
        test_data['Churn_Probability'] = y_pred_prob
        test_data['Churn_Prediction'] = y_pred


        # Function to apply row-based coloring (red for churn, green for no churn)
        def highlight_churn(row):
            color = "background-color: red; color: white;" if row[
                                                                  "Churn_Prediction"] == 1 else "background-color: lightgreen; color: black;"
            return [color] * len(row)


        # Apply the styling to the original test_data
        styled_df = test_data.style.apply(highlight_churn, axis=1)

        st.markdown("<h3 class='centered-header'> New Data with Predictions and Probabilities</h3>", unsafe_allow_html=True)

        st.dataframe(styled_df, use_container_width=True)

        display_feature_insights(log_reg, test_data_processed, feature_names)


        col_p, col_t = st.columns([1,1])
        def segment_customer(row):
            if row['MonthlyCharges'] > 80 and row['tenure'] < 12:  # Adjust thresholds
                return "High Risk"
            elif row['MonthlyCharges'] > 40 and row['tenure'] < 36:
                return "Medium Risk"
            else:
                return "Low Risk"

        def color_rows(row):
            if row['Risk Segment'] == 'High Risk':
                return ['background-color: red'] * len(row)
            elif row['Risk Segment'] == 'Medium Risk':
                return ['background-color: orange'] * len(row)
            elif row['Risk Segment'] == 'Low Risk':
                return ['background-color: green'] * len(row)
            else:
                return [''] * len(row)
        with col_p:
            st.markdown("<h3 class='centered-header'>Customer Segmentation Based on Tenure & MonthlyCharges</h3>", unsafe_allow_html=True)
            test_data['Risk Segment'] = test_data.apply(segment_customer, axis=1)

            segment_counts = test_data['Risk Segment'].value_counts().reset_index()
            segment_counts.columns = ['Risk Segment', 'Count']

            fig_segment = px.bar(segment_counts, x='Risk Segment', y='Count')
            st.plotly_chart(fig_segment)

        with col_t:
            styled_table = test_data.style.apply(color_rows, axis=1)
            st.markdown("<h3 class='centered-header'>Customer Data with Risk Segment Based on Tenure & MonthlyCharges</h3>", unsafe_allow_html=True)

            st.dataframe(styled_table)
