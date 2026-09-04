import numpy as np

scores = np.array([72, 85, 91, 68, 77, 95, 83])

print("Mean:", scores.mean())
print("Highest:", scores.max())
print("Lowest:", scores.min())
print("Scores of at least 80:", scores[scores>=80])
print("Number of scores at least 80:", len(scores[scores>=80])) #scores[scores>=80].sum()


import pandas as pd

students = pd.DataFrame({
    "Name": ["Alex", "Beth", "Chris", "Diana", "Ethan"],
    "Age": [18, 20, 19, 21, 18],
    "Score": [72, 85, 91, 68, 95],
    "Passed": [True, True, True, False, True]
})

print("Average score:", students["Score"].mean())
print("Students aged 19 or older:")
print(students[students["Age"] >= 19])

print("Names of students who scored at least 80:")
print(students[students["Score"] >= 80]["Name"])
print(students.loc[students["Score"] >= 80, "Name"])


passengers = pd.DataFrame({
    "Name": ["Alex", "Beth", "Chris", "Diana", "Ethan"],
    "Age": [18, 20, np.nan, 21, 18],
    "Fare": [25, 40, 35, np.nan, 50],
    "Survived": [0, 1, 1, 0, 1]
})

print(passengers.head())
print(passengers.isna().sum())

median_fare = passengers["Fare"].median()

passengers["Fare"] = passengers["Fare"].fillna(median_fare)

median_age = passengers["Age"].median()
passengers["Age"] = passengers["Age"].fillna(median_age)

print(passengers)
print("Missing values remaining:")
print(passengers.isna().sum())
print(median_fare)

import matplotlib.pyplot as plt

plt.scatter(passengers["Age"], passengers["Fare"])

plt.xlabel("Passenger Age")
plt.ylabel("Passenger Fare")
plt.title("Passenger Age vs Fare")



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

print(X.head())
print(y.head())

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=42,
    stratify=y_remaining
)

print("Training examples:", len(X_train))
print("Validation examples:", len(X_validation))
print("Test examples:", len(X_test))

print("Training feature shape:", X_train.shape)
print("Training target shape:", y_train.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

validation_predictions = model.predict(X_validation)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print("Validation predictions:", validation_predictions)
print("Correct answers:", y_validation.to_numpy())
print("Validation accuracy:", validation_accuracy)

training_predictions = model.predict(X_train)

training_accuracy = accuracy_score(
    y_train,
    training_predictions
)

print(f"Training accuracy: {training_accuracy:.2%}")
print(f"Validation accuracy: {validation_accuracy:.2%}")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

matrix = confusion_matrix(
    y_validation,
    validation_predictions
)

print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=iris.target_names
)

display.plot(cmap="Blues")
plt.title("Validation Confusion Matrix")
plt.show()

from sklearn.metrics import classification_report

print(
    classification_report(
        y_validation,
        validation_predictions,
        target_names=iris.target_names
    )
)