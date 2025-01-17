import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """
    Load data from Excel file
    """
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    """
    Perform exploratory data analysis
    """
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Correlation heatmap for numerical columns
    plt.subplot(2, 2, 1)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numerical_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    
    # Price distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='Price', kde=True)
    plt.title('Price Distribution')
    
    # Box plot for price by categorical variable (assuming 'brand' exists)
    if 'brand' in df.columns:
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x='Brand', y='Price')
        plt.xticks(rotation=45)
        plt.title('Price Distribution by Brand')
    
    # Scatter plot of price vs numerical variable
    num_cols = df.select_dtypes(include=[np.number]).columns
    price_col = num_cols[num_cols != 'Price'][0]  # First numerical column that's not price
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x=price_col, y='Price')
    plt.title(f'Price vs {price_col}')
    
    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    """
    Preprocess the data including handling missing values,
    encoding categorical variables, and scaling numerical features
    """
    print("\n=== Data Preprocessing ===")
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Handle missing values
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            print(f"Filling missing values in {col} with mean")
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            print(f"Filling missing values in {col} with mode")
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Create dummy variables for categorical columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Separate features and target
    X = df_processed.drop('Price', axis=1)
    y = df_processed['Price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

def build_models(X, y):
    """
    Build and evaluate multiple regression models
    """
    print("\n=== Model Building and Evaluation ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"R2 Score: {r2:.3f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=4)
        print(f"Cross-validation scores (5-fold):")
        print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{name}: Actual vs Predicted')
        plt.show()
        
        # For Linear Regression, show feature importance
        if name == 'Linear Regression':
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_
            })
            feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
            feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='Coefficient', y='Feature')
            plt.title('Top 10 Feature Importance (Linear Regression)')
            plt.show()

def main():
    # Specify your Excel file path
    file_path = '/DATA_SCIENCE/OLX Car Sales_sample.xlsx'  # Replace with your actual file path
    
    # Load the data
    df = load_data(file_path)
    if df is None:
        return
    
    # Perform EDA
    explore_data(df)
    
    # Preprocess the data
    X_scaled, y = preprocess_data(df)
    
    # Build and evaluate models
    build_models(X_scaled, y)

if __name__ == "__main__":
    main()