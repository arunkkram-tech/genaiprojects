import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class CarPricePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = LinearRegression()
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.feature_columns = None
        self.numerical_cols = None
        self.categorical_cols = None

    def train(self, data_file):
        # Load data
        df = pd.read_excel(data_file)
        
        # Store column information
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.numerical_cols.remove('price')  # Remove target variable
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Display basic info
        st.write("Data Shape:", df.shape)
        st.write("\nMissing Values:")
        st.write(df.isnull().sum())
        
        # Display descriptive statistics
        st.write("\nDescriptive Statistics:")
        st.write(df.describe())
        
        # Visualizations
        self.plot_eda(df)
        
        # Handle missing values
        for col in self.numerical_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        
        for col in self.categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Create dummy variables
        df_encoded = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
        
        # Store feature columns
        self.feature_columns = [col for col in df_encoded.columns if col != 'price']
        
        # Prepare features and target
        X = df_encoded[self.feature_columns]
        y = df_encoded['price']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train models
        self.lr_model.fit(X_train, y_train)
        self.dt_model.fit(X_train, y_train)
        
        # Calculate and display model performance
        lr_pred = self.lr_model.predict(X_test)
        dt_pred = self.dt_model.predict(X_test)
        
        st.write("\nModel Performance:")
        st.write("Linear Regression R² Score:", round(self.lr_model.score(X_test, y_test), 3))
        st.write("Decision Tree R² Score:", round(self.dt_model.score(X_test, y_test), 3))

    def plot_eda(self, df):
        # Create visualizations
        st.write("\nExploratory Data Analysis Visualizations:")
        
        # Price distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='price', kde=True, ax=ax)
        plt.title('Price Distribution')
        st.pyplot(fig)
        
        # Correlation heatmap
        numerical_data = df[self.numerical_cols + ['price']]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap')
        st.pyplot(fig)

    def predict(self, input_data):
        # Create DataFrame with user input
        input_df = pd.DataFrame([input_data])
        
        # Create dummy variables
        input_encoded = pd.get_dummies(input_df, columns=self.categorical_cols)
        
        # Align columns with training data
        for col in self.feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[self.feature_columns]
        
        # Scale features
        input_scaled = self.scaler.transform(input_encoded)
        
        # Make predictions
        lr_pred = self.lr_model.predict(input_scaled)[0]
        dt_pred = self.dt_model.predict(input_scaled)[0]
        
        return lr_pred, dt_pred

def main():
    st.title("Car Price Prediction App")
    
    # Initialize or get the model from session state
    if 'model' not in st.session_state:
        st.session_state.model = CarPricePredictor()
        st.session_state.trained = False
    
    # File uploader for training data
    data_file = st.file_uploader("Upload your car dataset (Excel file)", type=['xlsx'])
    
    if data_file is not None and st.button("Train Model"):
        st.write("Training model...")
        st.session_state.model.train(data_file)
        st.session_state.trained = True
        st.success("Model trained successfully!")
    
    if st.session_state.trained:
        st.header("Enter Car Details for Prediction")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        # Dictionary to store user inputs
        user_input = {}
        
        # Numerical inputs
        with col1:
            st.subheader("Numerical Features")
            for col in st.session_state.model.numerical_cols:
                user_input[col] = st.number_input(f"Enter {col}", value=0.0)
        
        # Categorical inputs
        with col2:
            st.subheader("Categorical Features")
            for col in st.session_state.model.categorical_cols:
                # You'll need to replace these options with actual categories from your data
                options = ['option1', 'option2', 'option3']  # Replace with your actual options
                user_input[col] = st.selectbox(f"Select {col}", options)
        
        if st.button("Predict Price"):
            lr_pred, dt_pred = st.session_state.model.predict(user_input)
            
            # Display predictions
            st.header("Predicted Prices")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Linear Regression Prediction", f"${lr_pred:,.2f}")
            
            with col2:
                st.metric("Decision Tree Prediction", f"${dt_pred:,.2f}")

if __name__ == "__main__":
    main()
