import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
joblib.dump(
    model,
    "ml/saved_model/model.joblib"
)

print("\nModel Saved Successfully")
loaded_model = joblib.load(
    "ml/saved_model/model.joblib"
)

print("\nModel Loaded Successfully")
sample_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = loaded_model.predict(sample_flower)

predicted_species = iris.target_names[prediction[0]]

print("\nPrediction Result:")
print("Predicted Species:", predicted_species)