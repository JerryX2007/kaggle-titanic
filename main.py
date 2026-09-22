import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


import warnings

# Suppress a known false NumPy/Apple M4 matmul warning.
warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning
)

def engineer_features(data):
    data = data.copy()

    data["FamilySize"] = (
        data["SibSp"]
        + data["Parch"]
        + 1
    )

    data["FamilyGroup"] = pd.cut(
        data["FamilySize"],
        bins=[0, 1, 4, float("inf")],
        labels=["Alone", "Small", "Large"]
    )

    data["Title"] = (
        data["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
    )

    data["Title"] = data["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    common_titles = ["Mr", "Miss", "Mrs", "Master"]

    data["Title"] = data["Title"].where(
        data["Title"].isin(common_titles),
        "Rare"
    )

    return data


train_data = pd.read_csv("train.csv")
train_data = engineer_features(train_data)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilyGroup",
    "Title"
]

numeric_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
    "FamilyGroup",
    "Title"
]

X = train_data[features]
y = train_data["Survived"]

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training examples:", len(X_train))
print("Validation examples:", len(X_validation))

numeric_processor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_processor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_processor, numeric_features),
    ("categorical", categorical_processor, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            solver="liblinear",
            max_iter=1000
        )
    )
])

model.fit(X_train, y_train)

validation_predictions = model.predict(X_validation)

model_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

always_zero_predictions = [0] * len(y_validation)

simple_baseline_accuracy = accuracy_score(
    y_validation,
    always_zero_predictions
)

print(f"Always-zero accuracy: {simple_baseline_accuracy:.2%}")
print(f"Logistic regression accuracy: {model_accuracy:.2%}")


print("\nConfusion matrix:")
print(confusion_matrix(y_validation, validation_predictions))

print("\nClassification report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        target_names=["Did not survive", "Survived"]
    )
)

cross_validator = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cross_validation_results = cross_validate(
    model,
    X,
    y,
    cv=cross_validator,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=True
)

print("\nValidation accuracy for each fold:")
print(cross_validation_results["test_accuracy"])

print(
    "Mean training accuracy:",
    cross_validation_results["train_accuracy"].mean()
)

print(
    "Mean validation accuracy:",
    cross_validation_results["test_accuracy"].mean()
)

print(
    "Mean survivor precision:",
    cross_validation_results["test_precision"].mean()
)

print(
    "Mean survivor recall:",
    cross_validation_results["test_recall"].mean()
)

print(
    "Mean survivor F1:",
    cross_validation_results["test_f1"].mean()
)

print(
    "Validation accuracy standard deviation:",
    cross_validation_results["test_accuracy"].std()
)

baseline_features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

baseline_categorical_features = [
    "Pclass",
    "Sex",
    "Embarked"
]

baseline_preprocessor = ColumnTransformer([
    ("numeric", numeric_processor, numeric_features),
    (
        "categorical",
        categorical_processor,
        baseline_categorical_features
    )
])

baseline_model = Pipeline([
    ("preprocessor", baseline_preprocessor),
    (
        "classifier",
        LogisticRegression(
            solver="liblinear",
            max_iter=1000
        )
    )
])

baseline_results = cross_validate(
    baseline_model,
    train_data[baseline_features],
    y,
    cv=cross_validator,
    scoring=["accuracy", "precision", "recall", "f1"]
)

print("\nBaseline cross-validation:")
print(
    "Accuracy:",
    baseline_results["test_accuracy"].mean()
)
print(
    "Survivor recall:",
    baseline_results["test_recall"].mean()
)
print(
    "Survivor F1:",
    baseline_results["test_f1"].mean()
)

print("\nEngineered cross-validation:")
print(
    "Accuracy:",
    cross_validation_results["test_accuracy"].mean()
)
print(
    "Survivor recall:",
    cross_validation_results["test_recall"].mean()
)
print(
    "Survivor F1:",
    cross_validation_results["test_f1"].mean()
)

parameter_search = GridSearchCV(
    estimator=model,
    param_grid={
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    },
    scoring={
        "accuracy": "accuracy",
        "recall": "recall",
        "f1": "f1"
    },
    refit="accuracy",
    cv=cross_validator,
    return_train_score=True
)

parameter_search.fit(X, y)

regularization_results = pd.DataFrame(
    parameter_search.cv_results_
)

print(
    regularization_results[[
        "param_classifier__C",
        "mean_train_accuracy",
        "mean_test_accuracy",
        "mean_test_recall",
        "mean_test_f1"
    ]]
)

print("\nBest C:")
print(parameter_search.best_params_)

print("\nBest cross-validation accuracy:")
print(parameter_search.best_score_)
print(
    regularization_results[[
        "param_classifier__C",
        "mean_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_recall",
        "mean_test_f1"
    ]].to_string(index=False)
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            C=10,
            solver="liblinear",
            max_iter=1000
        )
    )
])



print("\nPassenger title counts:")
print(train_data["Title"].value_counts())

print("\nGrouped title statistics:")
print(
    train_data.groupby("Title")["Survived"].agg(
        ["count", "mean"]
    )
)

X = train_data[features]

title_results = cross_validate(
    model,
    X,
    y,
    cv=cross_validator,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=True
)

print("\nModel with Title:")
print("Training accuracy:", title_results["train_accuracy"].mean())
print("Validation accuracy:", title_results["test_accuracy"].mean())
print("Survivor precision:", title_results["test_precision"].mean())
print("Survivor recall:", title_results["test_recall"].mean())
print("Survivor F1:", title_results["test_f1"].mean())
print(
    "Accuracy standard deviation:",
    title_results["test_accuracy"].std()
)

title_parameter_search = GridSearchCV(
    estimator=model,
    param_grid={
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    },
    scoring={
        "accuracy": "accuracy",
        "recall": "recall",
        "f1": "f1"
    },
    refit="accuracy",
    cv=cross_validator,
    return_train_score=True
)

title_parameter_search.fit(X, y)

title_tuning_results = pd.DataFrame(
    title_parameter_search.cv_results_
)

print(
    title_tuning_results[[
        "param_classifier__C",
        "mean_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_recall",
        "mean_test_f1"
    ]].to_string(index=False)
)

print("\nBest C with Title:")
print(title_parameter_search.best_params_)

print("\nBest accuracy with Title:")
print(title_parameter_search.best_score_)

print(title_parameter_search.best_score_)
test_data = pd.read_csv("test.csv")
test_data = engineer_features(test_data)

X_test = test_data[features]

final_model = title_parameter_search.best_estimator_

test_predictions = final_model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions.astype(int)
})

assert submission.shape == (418, 2)
assert submission["PassengerId"].equals(
    test_data["PassengerId"]
)
assert set(submission["Survived"].unique()).issubset({0, 1})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSubmission preview:")
print(submission.head())

print("\nSubmission shape:")
print(submission.shape)

print("\nPrediction counts:")
print(submission["Survived"].value_counts())

forest_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )
    )
])

forest_results = cross_validate(
    forest_model,
    X,
    y,
    cv=cross_validator,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=True
)

forest_parameter_search = GridSearchCV(
    estimator=forest_model,
    param_grid={
        "classifier__max_depth": [3, 5, 7, None],
        "classifier__min_samples_leaf": [1, 3, 5, 10]
    },
    scoring={
        "accuracy": "accuracy",
        "recall": "recall",
        "f1": "f1"
    },
    refit="accuracy",
    cv=cross_validator,
    return_train_score=True
)

forest_parameter_search.fit(X, y)

forest_tuning_results = pd.DataFrame(
    forest_parameter_search.cv_results_
)

top_forest_results = forest_tuning_results.sort_values(
    "mean_test_accuracy",
    ascending=False
).head(10)

print("\nTop random forest settings:")
print(
    top_forest_results[[
        "param_classifier__max_depth",
        "param_classifier__min_samples_leaf",
        "mean_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_recall",
        "mean_test_f1"
    ]].to_string(index=False)
)

print("\nBest random forest parameters:")
print(forest_parameter_search.best_params_)

print("\nBest random forest accuracy:")
print(forest_parameter_search.best_score_)

print("\nRandom forest results:")
print(
    "Training accuracy:",
    forest_results["train_accuracy"].mean()
)
print(
    "Validation accuracy:",
    forest_results["test_accuracy"].mean()
)
print(
    "Survivor precision:",
    forest_results["test_precision"].mean()
)
print(
    "Survivor recall:",
    forest_results["test_recall"].mean()
)
print(
    "Survivor F1:",
    forest_results["test_f1"].mean()
)
print(
    "Accuracy standard deviation:",
    forest_results["test_accuracy"].std()
)

forest_parameter_search = GridSearchCV(
    estimator=forest_model,
    param_grid={
        "classifier__max_depth": [3, 5, 7, None],
        "classifier__min_samples_leaf": [1, 3, 5, 10]
    },
    scoring={
        "accuracy": "accuracy",
        "recall": "recall",
        "f1": "f1"
    },
    refit="accuracy",
    cv=cross_validator,
    return_train_score=True
)

forest_parameter_search.fit(X, y)

forest_tuning_results = pd.DataFrame(
    forest_parameter_search.cv_results_
)

top_forest_results = forest_tuning_results.sort_values(
    "mean_test_accuracy",
    ascending=False
).head(10)

print("\nTop random forest settings:")
print(
    top_forest_results[[
        "param_classifier__max_depth",
        "param_classifier__min_samples_leaf",
        "mean_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_recall",
        "mean_test_f1"
    ]].to_string(index=False)
)

print("\nBest random forest parameters:")
print(forest_parameter_search.best_params_)

print("\nBest random forest accuracy:")
print(forest_parameter_search.best_score_)

final_forest_model = (
    forest_parameter_search.best_estimator_
)

forest_test_predictions = final_forest_model.predict(
    X_test
)

forest_submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": forest_test_predictions.astype(int)
})

assert forest_submission.shape == (418, 2)
assert set(
    forest_submission["Survived"].unique()
).issubset({0, 1})

forest_submission.to_csv(
    "submission_random_forest.csv",
    index=False
)

print("\nRandom forest submission preview:")
print(forest_submission.head())

print("\nPrediction counts:")
print(forest_submission["Survived"].value_counts())

print(
    "\nPredictions different from logistic regression:",
    (forest_test_predictions != test_predictions).sum()
)