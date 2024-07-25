import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving and loading the model

# Load the data from Excel
data = pd.read_excel('landmarks_data.xlsx')

# Prepare features and labels
X = data[['Left_Hip_X', 'Left_Hip_Y', 'Left_Hip_Z', 'Right_Hip_X', 'Right_Hip_Y', 'Right_Hip_Z', 'Left_Shoulder_X', 'Left_Shoulder_Y', 'Left_Shoulder_Z', 'Right_Shoulder_X', 'Right_Shoulder_Y', 'Right_Shoulder_Z', 'Nose_X', 'Nose_Y', 'Nose_Z']]
y = data['Class']

# Handle missing values if any
X.fillna(method='ffill', inplace=True)
X.fillna(method='bfill', inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
