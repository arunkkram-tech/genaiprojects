import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

class CarPricePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = LinearRegression()
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.categorical_cols = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
        self.numerical_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
        self.target = 'Price'
        self.feature_columns = None

    def preprocess_data(self, df, is_training=True):
        # Create a copy
        processed_df = df.copy()
        
        # Handle missing values
        for col in self.numerical_cols:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            
        for col in self.categorical_cols:
            processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
        
        # Create dummy variables
        processed_df = pd.get_dummies(processed_df, columns=self.categorical_cols, drop_first=True)
        
        if is_training:
            self.feature_columns = [col for col in processed_df.columns if col != self.target]
            return processed_df
        else:
            # Ensure all training columns exist in prediction data
            for col in self.feature_columns:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            return processed_df[self.feature_columns]

    def train(self, df):
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Split features and target
        X = processed_df[self.feature_columns]
        y = processed_df[self.target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train models
        self.lr_model.fit(X_train, y_train)
        self.dt_model.fit(X_train, y_train)
        
        # Get predictions
        lr_pred = self.lr_model.predict(X_test)
        dt_pred = self.dt_model.predict(X_test)
        
        return {
            'lr_r2': r2_score(y_test, lr_pred),
            'dt_r2': r2_score(y_test, dt_pred),
            'lr_rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'dt_rmse': np.sqrt(mean_squared_error(y_test, dt_pred))
        }

    def predict(self, df):
        # Preprocess prediction data
        processed_df = self.preprocess_data(df, is_training=False)
        
        # Scale features
        X_scaled = self.scaler.transform(processed_df)
        
        # Make predictions
        lr_pred = self.lr_model.predict(X_scaled)
        dt_pred = self.dt_model.predict(X_scaled)
        
        return lr_pred, dt_pred

def plot_eda(df):
    st.subheader("Exploratory Data Analysis")
    
    # Numerical columns analysis
    numerical_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']
    
    # Correlation heatmap
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)
    plt.close()
    
    # Price distribution
    st.write("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Price', kde=True)
    plt.title('Price Distribution')
    st.pyplot(fig)
    plt.close()
    
    # Categorical analysis
    st.write("Average Price by Categories")
    categorical_cols = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(categorical_cols):
        sns.boxplot(data=df, x=col, y='Price', ax=axes[idx])
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
        axes[idx].set_title(f'Price by {col}')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def main():
    st.title("Car Price Prediction App")
    
    # Initialize model in session state
    if 'model' not in st.session_state:
        st.session_state.model = CarPricePredictor()
        st.session_state.trained = False
    
    # File upload section
    st.header("1. Model Training")
    training_file = st.file_uploader("Upload Training Data (XLSX)", type=['xlsx'], key='train')

    
    
    if training_file is not None:
        try:
            training_data = pd.read_excel(training_file)
            st.write("Training Data Preview:")
            st.write(training_data.head())
            
            if st.button("Train Model"):
                st.write("Training model...")
                
                # Perform EDA
                plot_eda(training_data)
                
                # Train model
                metrics = st.session_state.model.train(training_data)
                
                st.write("Model Performance Metrics:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Linear Regression R² Score", f"{metrics['lr_r2']:.3f}")
                    st.metric("Linear Regression RMSE", f"{metrics['lr_rmse']:.2f}")
                with col2:
                    st.metric("Decision Tree R² Score", f"{metrics['dt_r2']:.3f}")
                    st.metric("Decision Tree RMSE", f"{metrics['dt_rmse']:.2f}")
                
                st.session_state.trained = True
                st.success("Model trained successfully!")
        
        except Exception as e:            
            st.error(f"Error: {str(e)}")
            #error_trace = traceback.format_exc()
            #st.error(f"Error: {error_trace}")
    
    # Prediction section
    if st.session_state.trained:
        st.header("2. Price Prediction")
        prediction_file = st.file_uploader("Upload Prediction Data (XLSX)", type=['xlsx'], key='predict')
        
        if prediction_file is not None:
            try:
                prediction_data = pd.read_excel(prediction_file)
                st.write("Prediction Data Preview:")
                st.write(prediction_data.head())
                
                if st.button("Generate Predictions"):
                    lr_pred, dt_pred = st.session_state.model.predict(prediction_data)
                    
                    # Add predictions to the dataframe
                    results = prediction_data.copy()
                    results['Linear_Regression_Prediction'] = lr_pred
                    results['Decision_Tree_Prediction'] = dt_pred
                    
                    st.write("Prediction Results:")
                    st.write(results)
                    
                    # Download button for predictions
                    st.download_button(
                        label="Download Predictions",
                        data=results.to_csv(index=False).encode('utf-8'),
                        file_name="car_price_predictions.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                #error_trace = traceback.format_exc()
                #st.error(f"Error: {error_trace}")
    else:
        st.warning("Please train the model first before making predictions.")

if __name__ == "__main__":
    main()
