import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("crop_yield.csv")

# Encode categorical columns
label_encoders = {}
categorical_cols = ["Crop", "Season", "State"]
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
X = data[["Crop", "Crop_Year", "Season", "State", "Area", "Production",
          "Annual_Rainfall", "Fertilizer", "Pesticide"]]
y = data["Yield"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "trained_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model training complete. Files saved: trained_model.pkl, label_encoders.pkl")
