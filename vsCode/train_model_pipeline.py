from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def run_model_pipeline(df):
    print("Preparing features and target variable...")

    X = df.drop(columns=["price", "host_id", "neighbourhood"])
    y = df["price"]

    # Compare model performance before training
    compare_models(X, y)

    # Train best model (Random Forest for now)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining final model: Random Forest Regressor...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Final model RMSE on test data: ${rmse:.2f}")

    # Visualizations
    plot_predicted_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_feature_importance(model, X.columns)

    return model


def compare_models(X, y):
    print("\nComparing model performance (5-fold cross-validation)...")

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Support Vector Regressor": SVR(C=1.0, epsilon=0.2)
    }

    for name, model in models.items():
        neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
        rmse_scores = (-neg_mse_scores) ** 0.5
        print(f"{name:25} RMSE: ${rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")


def plot_predicted_vs_actual(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs. Actual Airbnb Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
