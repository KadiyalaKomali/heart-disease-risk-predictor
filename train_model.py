import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("heart.csv")

# Target
y = df["HeartDisease"]

# Label Encoding for categorical columns
le_sex = LabelEncoder()
le_cp = LabelEncoder()
le_ecg = LabelEncoder()
le_angina = LabelEncoder()
le_slope = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["ChestPainType"] = le_cp.fit_transform(df["ChestPainType"])
df["RestingECG"] = le_ecg.fit_transform(df["RestingECG"])
df["ExerciseAngina"] = le_angina.fit_transform(df["ExerciseAngina"])
df["ST_Slope"] = le_slope.fit_transform(df["ST_Slope"])

X = df.drop("HeartDisease", axis=1)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("le_sex.pkl", "wb") as f:
    pickle.dump(le_sex, f)

with open("le_cp.pkl", "wb") as f:
    pickle.dump(le_cp, f)

with open("le_ecg.pkl", "wb") as f:
    pickle.dump(le_ecg, f)

with open("le_angina.pkl", "wb") as f:
    pickle.dump(le_angina, f)

with open("le_slope.pkl", "wb") as f:
    pickle.dump(le_slope, f)

print("✅ Model and encoders saved successfully.")
