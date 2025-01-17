# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class CarPriceModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = LinearRegression()
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.numerical_columns = None
        self.categorical_columns = None
        self.dummy_columns = None
        
    def load_and_process_data(self, file_path):
        """Load and preprocess the data"""
        # Load data
        df = pd.read_excel(file_path)
        
        # Store column information
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numerical_columns.remove('Price')  # Remove target variable
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values
        for col in self.numerical_columns:
            df[col].fillna(df[col].mean(), inplace=True)
        for col in self.categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Create dummy variables
        df_encoded = pd.get_dummies(df, columns=self.categorical_columns, drop_first=True)
        self.dummy_columns = [col for col in df_encoded.columns 
                            if col not in self.numerical_columns + ['Price']]
        
        # Prepare features and target
        X = df_encoded.drop('Price', axis=1)
        y = df_encoded['Price']
        
        return X, y
    
    def train_models(self, X, y):
        """Train both linear regression and decision tree models"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train models
        self.lr_model.fit(X_train, y_train)
        self.dt_model.fit(X_train, y_train)
        
        # Get predictions
        lr_pred = self.lr_model.predict(X_test)
        dt_pred = self.dt_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Linear Regression': {
                'R2': r2_score(y_test, lr_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred))
            },
            'Decision Tree': {
                'R2': r2_score(y_test, dt_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred))
            }
        }
        
        return metrics
    
    def save_models(self, filename):
        """Save the trained models and preprocessing information"""
        model_data = {
            'scaler': self.scaler,
            'lr_model': self.lr_model,
            'dt_model': self.dt_model,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'dummy_columns': self.dummy_columns
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

def load_model(filename):
    """Load the trained model and preprocessing information"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def prepare_single_prediction(input_data, model_data):
    """Prepare a single input for prediction"""
    # Create a DataFrame with numerical columns
    df = pd.DataFrame([input_data])
    
    # Create dummy variables
    for col in model_data['categorical_columns']:
        # Get the selected value for this categorical column
        selected_value = input_data[col]
        
        # Create dummy column names that would match the training data
        possible_dummies = [f"{col}_{selected_value}"]
        
        # Set all dummy columns to 0 initially
        for dummy_col in model_data['dummy_columns']:
            if dummy_col.startswith(col + '_'):
                df[dummy_col] = 0
        
        # Set the selected dummy to 1
        for dummy_col in possible_dummies:
            if dummy_col in model_data['dummy_columns']:
                df[dummy_col] = 1
    
    # Ensure all columns match the training data
    missing_cols = set(model_data['dummy_columns']) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
        
    # Reorder columns to match training data
    final_columns = model_data['numerical_columns'] + model_data['dummy_columns']
    df = df[final_columns]
    
    # Scale the features
    df_scaled = model_data['scaler'].transform(df)
    
    return df_scaled

def main():
    st.title("Car Price Prediction App")
    
    # Load the trained model and preprocessing information
    try:
        model_data = load_model('car_price_model.pkl')
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Create input form
    st.header("Enter Car Details")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    # Numerical inputs
    with col1:
        st.subheader("Numerical Features")
        for col in model_data['numerical_columns']:
            input_data[col] = st.number_input(f"Enter {col}", value=0.0)
    
    # Categorical inputs
    with col2:
        st.subheader("Categorical Features")
        for col in model_data['categorical_columns']:
            # Extract unique values from dummy columns
            unique_values = set()
            for dummy_col in model_data['dummy_columns']:
                if dummy_col.startswith(col + '_'):
                    value = dummy_col.replace(col + '_', '')
                    unique_values.add(value)
            input_data[col] = st.selectbox(f"Select {col}", list(unique_values))
    
    # Make prediction button
    if st.button("Predict Price"):
        # Prepare input data
        X_pred = prepare_single_prediction(input_data, model_data)
        
        # Make predictions
        lr_pred = model_data['lr_model'].predict(X_pred)[0]
        dt_pred = model_data['dt_model'].predict(X_pred)[0]
        
        # Display predictions
        st.header("Predicted Prices")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Linear Regression Prediction", f"${lr_pred:,.2f}")
        
        with col2:
            st.metric("Decision Tree Prediction", f"${dt_pred:,.2f}")
        
        # Feature importance plot for Linear Regression
        if st.checkbox("Show Feature Importance"):
            coefficients = pd.DataFrame({
                'Feature': model_data['numerical_columns'] + model_data['dummy_columns'],
                'Importance': abs(model_data['lr_model'].coef_)
            })
            coefficients = coefficients.sort_values('Importance', ascending=False)
            
            fig = px.bar(coefficients.head(10), 
                        x='Importance', 
                        y='Feature',
                        title='Top 10 Most Important Features')
            st.plotly_chart(fig)

if __name__ == "__main__":
    # Training script
    if st.sidebar.button("Train New Model"):
        st.sidebar.write("Training new model...")
        model = CarPriceModel()
        X, y = model.load_and_process_data('car_data.xlsx')  # Replace with your data file
        metrics = model.train_models(X, y)
        model.save_models('car_price_model.pkl')
        st.sidebar.write("Model training completed!")
        st.sidebar.write("Model Metrics:", metrics)
    
    # Run the main app
    main()
