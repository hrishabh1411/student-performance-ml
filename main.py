import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

print("Loading dataset...")

df = pd.read_csv("data/student_data.csv")

# Features
X = df[["Hours_Studied", "Attendance", "Previous_Score"]]
y = df["Exam_Score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Dataset Info
# -------------------------
print("\nDATASET SIZE:", len(df))
print("TRAIN SIZE:", len(X_train))
print("TEST SIZE:", len(X_test))

# -------------------------
# Model 1: Linear Regression
# -------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)

# -------------------------
# Model 2: Random Forest
# -------------------------
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)

# -------------------------
# Model Comparison
# -------------------------
print("\nMODEL COMPARISON")
print("Linear Regression MAE:", lr_mae)
print("Random Forest MAE:", rf_mae)

# -------------------------
# MAE Bar Chart
# -------------------------
models = ["Linear Regression", "Random Forest"]
mae_scores = [lr_mae, rf_mae]

plt.figure()
plt.bar(models, mae_scores)
plt.title("Model Performance Comparison (Lower is Better)")
plt.ylabel("Mean Absolute Error")
plt.show()

# -------------------------
# Choose Best Model
# -------------------------
if rf_mae < lr_mae:
    best_model = rf_model
    print("\nRandom Forest selected as BEST model")
else:
    best_model = lr_model
    print("\nLinear Regression selected as BEST model")

# -------------------------
# Custom Prediction
# -------------------------
input_data = pd.DataFrame([[6, 80, 75]],
                          columns=["Hours_Studied", "Attendance", "Previous_Score"])

print("\nPrediction for custom student:")
print(best_model.predict(input_data))

# -------------------------
# Actual vs Predicted Plot
# -------------------------
plt.figure()
plt.scatter(y_test, lr_pred, label="Linear Regression")
plt.scatter(y_test, rf_pred, label="Random Forest")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Scores")
plt.legend()
plt.show()

# -------------------------
#   Random Forest only
# -------------------------
if hasattr(rf_model, "feature_importances_"):
    importance = rf_model.feature_importances_
    features = X.columns

    plt.figure()
    plt.bar(features, importance)
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

# -------------------------
# Save Best Model
# -------------------------
joblib.dump(best_model, "models/student_model.pkl")

print("\nModel saved successfully in models/student_model.pkl")