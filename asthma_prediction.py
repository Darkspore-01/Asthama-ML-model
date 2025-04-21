import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("C:\\Users\\HP\\Downloads\\Asthma ML model\\asthma_disease_data.csv")

# Drop unnecessary columns
X = df.drop(['Diagnosis', 'PatientID', 'DoctorInCharge'], axis=1)
y = df['Diagnosis']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model and feature names
joblib.dump(model, "asthma_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Optional: test with an example input
symptoms = [1, 0, 2, 1, 0, 0, 1, 3, 2, 0, 0, 1, 2, 1, 3, 0, 1, 0, 2, 1, 3, 0, 1, 2, 1, 0]  # Example only
prediction = model.predict([symptoms])[0]
print("\nPredicted Diagnosis (Example Input):", prediction)
