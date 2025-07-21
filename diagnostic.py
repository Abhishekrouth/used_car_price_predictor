import joblib
import pandas as pd

try:
    # Load the encoders
    encoders = joblib.load("model/encoders.pkl")
    
    print("=== ENCODER CLASSES ===")
    for encoder_name, encoder in encoders.items():
        print(f"\n{encoder_name}:")
        print(f"Classes: {list(encoder.classes_)}")
        print(f"Number of classes: {len(encoder.classes_)}")
    
    print("\n" + "="*50)
    
    # Also check the original dataset
    print("\n=== ORIGINAL DATASET ANALYSIS ===")
    try:
        df = pd.read_csv("used_car_dataset.csv")
        
        for col in ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']:
            if col in df.columns:
                unique_vals = df[col].unique()
                print(f"\n{col} (from CSV):")
                print(f"Unique values: {sorted([str(x) for x in unique_vals if pd.notna(x)])}")
                print(f"Count: {len(unique_vals)}")
            else:
                print(f"\n{col}: Column not found in CSV")
                
    except FileNotFoundError:
        print("used_car_dataset.csv not found")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        
except FileNotFoundError:
    print("model/encoders.pkl not found. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading encoders: {e}")