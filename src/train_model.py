import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/student_exam_performance.csv")

print("Dataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# Feature Selection

X = df[
    [
        'Age',
        'Study_Hours_Per_Week',
        'Attendance_Percentage',
        'Previous_Score',
        'Sleep_Hours'
    ]
]

y = df['Final_Exam_Score']


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Model Comparison

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_score = -999

print("\n========== MODEL COMPARISON ==========")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    print(f"{name}: R² Score = {round(r2,4)}")

    if r2 > best_score:

        best_score = r2

        best_model = model

# Best Model Evaluation

print("\n========== BEST MODEL ==========")

predictions = best_model.predict(X_test)

for i in range(min(5, len(y_test))):

    print(
        f"Actual: {y_test.iloc[i]} | Predicted: {predictions[i]:.2f}"
    )

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Evaluation")

print("Mean Absolute Error (MAE):", round(mae, 2))

print("R² Score:", round(r2, 4))

# Feature Importance

if hasattr(best_model, "feature_importances_"):

    print("\nFeature Importance")

    importance = best_model.feature_importances_

    for feature, score in zip(
        X.columns,
        importance
    ):

        print(
            f"{feature}: {round(score,4)}"
        )

with open("model.pkl", "wb") as file:

    pickle.dump(best_model, file)

print("\nBest Model Saved Successfully!")

print(
    f"\nBest Model R² Score = {round(best_score,4)}"
)