import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("cardekho.csv")

# Convert numeric columns
numeric_cols = [
    "year",
    "selling_price",
    "km_driven",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values
df = df.dropna()

# Columns used by the model
features = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats"
]

target = "selling_price"

# Create encoders
encoders = {}

categorical_cols = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

# Encode categorical columns
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    encoders[col] = encoder

# Prepare X and y
X = df[features]
y = df[target]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save trained model
joblib.dump(model, "carpredict_model.pkl")

# Save encoders
joblib.dump(encoders, "encoders.pkl")

print("Model trained successfully!")
print("carpredict_model.pkl created!")
print("encoders.pkl created!")
