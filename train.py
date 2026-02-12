import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Create simple dataset
data = {
    "study_hours": [2, 4, 6, 8, 1, 3, 7, 5, 9, 2],
    "attendance": [60, 80, 90, 95, 50, 70, 85, 75, 98, 65],
    "passed": [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# 2. Split features & target
X = df[["study_hours", "attendance"]]
y = df["passed"]

# 3. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
