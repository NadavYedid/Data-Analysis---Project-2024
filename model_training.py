import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error as mse
import pickle
from scipy import sparse

# Import prepare_data function from your custom module
from car_data_prep import prepare_data

def main():
    # Load your dataset
    try:
        df = pd.read_csv('dataset.csv')
        print("Dataset loaded successfully.")
        
        # Prepare the data using custom preprocessing function
        processed_df = prepare_data(df)
        print("Data preparation completed.")
        print("Processed DataFrame columns:", processed_df.columns)
        
        # Splitting the data into features (X) and target variable (y)
        X = processed_df.drop('Price', axis=1)  # Features
        y = processed_df['Price']  # Target variable
        
        # Save feature names for future use
        feature_names = X.columns.tolist()
        with open("feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        
        # Initializing and fitting the ElasticNet model
        alpha = 0.01
        l1_ratio = 0.9
        elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        elastic_net.fit(X, y)

        # Save the trained model using pickle
        with open("trained_model.pkl", "wb") as model_file:
            pickle.dump(elastic_net, model_file)
        print("Trained model saved as trained_model.pkl")
        
    except FileNotFoundError:
        print("Error: dataset.csv not found.")
    except Exception as e:
        print(f"Error during data processing: {e}")

if __name__ == "__main__":
    main()
