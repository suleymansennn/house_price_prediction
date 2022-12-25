import time

import joblib
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

import model_params
from helpers import *


def hyperparameter_optimization(X, y, cv=10):
    print("Hyperparameter Optimization Starting...")
    start_time = time.perf_counter()
    selected_features = feature_selection(X, y)
    X = X[selected_features]
    best_models = {}
    for name, model, params in model_params.regressors:
        print(f"{name.center(70, '#')}")
        rmse = np.mean(
            np.sqrt(
                -cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"Rmse(Before): {rmse}")
        r2 = np.mean(cross_val_score(model, X, y, cv=10))
        print(f"R2 Score(Before): {r2}")
        cv_best = GridSearchCV(model, params, cv=10, n_jobs=-1, verbose=False).fit(X, y)
        final_model = model.set_params(**cv_best.best_params_)
        r2_final = np.mean(cross_val_score(final_model, X, y, cv=10))
        rmse_final = np.mean(
            np.sqrt(
                -cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"Rmse(After): {rmse_final}")
        print(f"R2 Score(After): {r2}")
        print(f"{name} best params: {cv_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Processing time: {elapsed_time:.6f} seconds")
    return best_models, selected_features


def voting_regressor(weights, X, y, best_models, cv=10):
    start_time = time.perf_counter()
    print("Voting Regressor".center(70, "#"))
    voting_model = VotingRegressor(estimators=[
        ("LightGBM", best_models["LightGBM"].fit(X, y)),
        ("XGBoost", best_models["XGBoost"].fit(X, y)),
        ("GBM", best_models["GBM"].fit(X, y)),
        ("Extra Tree", best_models["Extra Tree"])],
        weights=weights).fit(X, y)
    r2_voting = np.mean(cross_val_score(voting_model, X, y, cv=cv))
    rmse_voting = np.mean(
        np.sqrt(-cross_val_score(voting_model, X, y, cv=cv, scoring="neg_mean_squared_error")))
    print(f"Voting Regressor R2 Score: {r2_voting}")
    print(f"Voting Regressor RMSE: {rmse_voting}")
    joblib.dump(voting_model, "cart_final.pkl")
    print("Voting Model Saved...")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Processing time: {elapsed_time:.6f} seconds")
    return voting_model


def predict(predictor, model):
    start_time = time.perf_counter()
    y_pred = model.predict(predictor)
    y_pred = np.exp(y_pred)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Processing time: {elapsed_time:.6f} seconds")
    return y_pred


def submission(y_pred, submission_path):
    start_time = time.perf_counter()
    submission = pd.read_csv(submission_path)
    submission["SalePrice"] = y_pred
    submission.to_csv("submission.csv", index=False)
    print("Submission file Saved...")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Processing time: {elapsed_time:.6f} seconds")
