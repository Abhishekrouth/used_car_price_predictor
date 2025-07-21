import pandas as pd
import json
import joblib
import os

def analyze_brand_models():
    try:
        # Load the original dataset
        df = pd.read_csv("used_car_dataset.csv")
        
        # Create brand-model mapping
        brand_models = {}
        
        # Group by brand and get unique models for each brand
        for brand in df['Brand'].unique():
            if pd.notna(brand):  # Skip NaN values
                brand_models[brand] = sorted(df[df['Brand'] == brand]['model'].unique().tolist())
        
        print("=== BRAND-MODEL MAPPING ===")
        for brand, models in sorted(brand_models.items()):
            print(f"\n{brand}:")
            for model in models:
                print(f"  - {model}")
        
        # Save this mapping for the Flask app
        os.makedirs('model', exist_ok=True)
        with open('model/brand_models.json', 'w') as f:
            json.dump(brand_models, f, indent=2)
        
        print(f"\n=== SUMMARY ===")
        print(f"Total brands: {len(brand_models)}")
        print(f"Brand-model mapping saved to 'model/brand_models.json'")
        
        return brand_models
        
    except FileNotFoundError:
        print("Error: used_car_dataset.csv not found")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    analyze_brand_models()