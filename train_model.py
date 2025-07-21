import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os


df = pd.read_csv("used_car_dataset.csv")
df.drop(columns=['PostedDate', 'AdditionInfo'], inplace=True)
df.dropna(inplace=True)

df['kmDriven'] = (
    df['kmDriven']
    .astype(str)
    .str.replace(' km', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
    .astype(int)
)

df['AskPrice'] = (
    df['AskPrice']
    .astype(str)
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.replace(' ', '', regex=False) 
    .astype(int)
)

print("Unique values in dataset:")
for col in ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']:
    print(f"{col}: {sorted(df[col].unique())}")
    print(f"Count: {len(df[col].unique())}\n")

label_encoders = {}
for col in ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
    print(f"{col} encoding:")
    for i, class_name in enumerate(le.classes_):
        print(f"  {class_name} -> {i}")
    print()

X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Model Performance:")
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

os.makedirs('model', exist_ok=True)

joblib.dump(model, 'model/model.pkl')
joblib.dump(label_encoders, 'model/encoders.pkl')

classes_info = {}
for col, encoder in label_encoders.items():
    classes_info[col] = encoder.classes_.tolist()

joblib.dump(classes_info, 'model/classes_info.pkl')

print("Model training complete and saved to 'model/' directory.")
print("\nAvailable classes saved for reference.")