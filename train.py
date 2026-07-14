import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load only first 50000 records (faster training)
data = pd.read_csv("datasets/fraudTrain.csv", nrows=50000)

print("Dataset Shape:", data.shape)

# Select useful columns
data = data[["category", "amt", "gender", "is_fraud"]]

# Encode categorical columns
le_category = LabelEncoder()
le_gender = LabelEncoder()

data["category"] = le_category.fit_transform(data["category"])
data["gender"] = le_gender.fit_transform(data["gender"])

# Features and Target
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"Accuracy : {accuracy*100:.2f}%")

# Save Files
joblib.dump(model, "model.pkl")
joblib.dump(le_category, "category_encoder.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")

print("✅ model.pkl saved")
print("✅ category_encoder.pkl saved")
print("✅ gender_encoder.pkl saved")