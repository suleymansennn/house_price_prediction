from train_models import *
from feature_engineering import *



def main(debug=True):
    print("Pipeline Started...")
    start_time = time.perf_counter()
    X, y, test = feature_engineering("datasets/train.csv", "datasets/test.csv")
    print("Model Training Started...")
    best_models, selected_features = hyperparameter_optimization(X, y, cv=10)
    voting_model = voting_regressor([1, 1, 1, 0.7], X[selected_features], y, best_models, cv=10)
    print("Predicting Results...")
    y_pred = predict(test[selected_features], voting_model)
    print("Submission...")
    submission(y_pred, "datasets/sample_submission.csv")
    print("Feature Importance...")
    compute_feature_importance(voting_model, [1, 1, 1, 0.7], X[selected_features], len_features=30)
    print("Tree Graph...")
    tree_graph(best_models["LightGBM"], 0)
    print("Regression Plot")
    y_pred = predict(X[selected_features], voting_model)
    plot_reg(y, y_pred)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total Processing time: {elapsed_time:.6f} seconds")


if __name__ == "__main__":
    main()
