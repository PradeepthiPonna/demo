import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Function to add a background image and additional styling
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #cceeff, #ffccff);
            background-size: cover;
        }

        /* Customize the title font */
        .stApp h1 {
            font-family: 'Arial', sans-serif;
            color: #3b3b3b;
            font-weight: bold;
        }

        /* Customize dataframe display */
        .dataframe {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
        }

        /* Customize sidebar if used */
        .css-1d391kg {
            background-color: #e6f7ff;
            padding: 20px;
        }

        /* Customizing sliders */
        .stSlider label {
            color: #3b3b3b;
        }

        /* Modify table appearance */
        table {
            background-color: #fff;
            border-radius: 10px;
            padding: 10px;
        }

        /* Customize the buttons */
        button {
            background-color: #4CAF50;
            color: white;
        }

        /* Customize headers */
        .stApp h2, .stApp h3 {
            color: #3b3b3b;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# Call the custom CSS function
add_custom_css()

# Title of the app
st.title("Sales Prediction App using Random Forest")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV data for sales prediction.", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Split data into features and target
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Pair plot to visualize relationships
    st.write("### Pair Plot")
    fig = sns.pairplot(data)
    st.pyplot(fig)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning options
    st.write("### Hyperparameter Tuning")
    n_estimators = st.slider("Number of Trees (n_estimators)", min_value=1, max_value=100, value=10)
    max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)

    # Initialize and train the Random Forest model with user-defined parameters
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Convert predictions to a Pandas Series for summary statistics
    predictions_series = pd.Series(predictions)

    # Display summary statistics for predictions
    st.write("### Summary Statistics for Predictions")
    st.write(predictions_series.describe())

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)  # RMSE calculation
    r2 = r2_score(y_test, predictions)

    # Display metrics
    st.write("### Model Performance Metrics")
    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.2f}")
    st.write(f"**R-squared**: {r2:.2f}")

    # Feature importance visualization
    st.write("### Feature Importance")
    importance = model.feature_importances_
    feature_names = X.columns

    # Plot feature importance
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=feature_names, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # Residuals distribution
    st.write("### Residuals Analysis")
    residuals = y_test - predictions
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    ax.axvline(0, color='red', linestyle='--')  # Line at zero for reference
    st.pyplot(fig)

    # Residuals vs Fitted Plot
    st.write("### Residuals vs Fitted Plot")
    fig, ax = plt.subplots()
    ax.scatter(predictions, residuals, color='blue', alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Fitted Values (Predicted Sales)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Plot")
    st.pyplot(fig)

    # Distribution Plot for each feature vs Sales
    st.write("### Feature Distributions vs Sales")
    for feature in X.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=feature, y='Sales', ax=ax)
        ax.set_title(f"{feature} vs Sales")
        st.pyplot(fig)

    # Cumulative Feature Importance Plot
    st.write("### Cumulative Feature Importance")
    sorted_importance = sorted(importance, reverse=True)
    cumulative_importance = np.cumsum(sorted_importance)
    fig, ax = plt.subplots()
    ax.plot(cumulative_importance, marker='o')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_title("Cumulative Feature Importance")
    st.pyplot(fig)

    # Actual vs Predicted
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    st.write("### Actual vs Predicted Sales")
    st.dataframe(results_df)

    # Sales Prediction Table
    st.write("### Sales Prediction Table")
    st.dataframe(results_df)

    # Distribution Plot for Actual vs Predicted
    st.write("### Actual vs Predicted Distribution Plot")
    fig, ax = plt.subplots()
    sns.kdeplot(y_test, label='Actual', ax=ax, color='blue', fill=True)
    sns.kdeplot(predictions, label='Predicted', ax=ax, color='green', fill=True)
    ax.set_title("Actual vs Predicted Distribution")
    ax.legend()
    st.pyplot(fig)

    # Scatter plot for Actual vs Predicted Sales
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, color='blue', label='Predicted', alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    ax.legend()
    st.pyplot(fig)
