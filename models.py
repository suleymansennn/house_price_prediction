from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib
from feature_engineering import *

warnings.filterwarnings("ignore")

matplotlib.use("Qt5Agg")
cat_feat_colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]

X = pd.read_csv("prepared_data/X_data.csv")
y = pd.read_csv("prepared_data/y_data.csv")
test = pd.read_csv("prepared_data/pre_test.csv")
y = np.log1p(y).to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)


def reg_models(base_model, X_train, y_train, X_test, y_test):
    rmse_list = []
    r2_list = []
    model_list = []
    for model in base_model:
        y_pred = model.fit(X_train, y_train).predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        rmse_list.append(rmse)
        r2_list.append(r2)
        model_list.append(str(model).split("(")[0][0:15])

    rmse_df = pd.DataFrame({"Scores": rmse_list,
                            "Model": model_list})
    r2_df = pd.DataFrame({"Scores": r2_list,
                          "Model": model_list})

    plt.subplot(1, 2, 1)
    plt.title("RMSE")
    ax = sns.barplot(data=rmse_df, x="Scores", y="Model", hue="Model", palette=cat_feat_colors)
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend(loc='upper left', title='Models')
    plt.subplot(1, 2, 2)
    plt.title("R2 Scores")
    ax = sns.barplot(data=r2_df, x="Scores", y="Model", hue="Model", palette=cat_feat_colors)
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend(loc='upper left', title='Models')
    print(rmse_df)
    print(r2_df)


base_models = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    XGBRegressor(),
    LGBMRegressor(),
    CatBoostRegressor(verbose=False)
]
"""
RMSE:
     Scores            Model
0  0.338453  LinearRegressio
1  0.576818  KNeighborsRegre
2  0.394912  DecisionTreeReg
3  0.318330  RandomForestReg
4  0.305120  GradientBoostin
5  0.323000     XGBRegressor
6  0.335787    LGBMRegressor
7  0.280412  <catboost.core.
"""
"""
R2:
     Scores            Model
0  0.884552  LinearRegressio
1  0.664674  KNeighborsRegre
2  0.842823  DecisionTreeReg
3  0.897872  RandomForestReg
4  0.906173  GradientBoostin
5  0.894854     XGBRegressor
6  0.886364    LGBMRegressor
7  0.920753  <catboost.core.
"""
reg_models(base_models, X_train, y_train, X_test, y_test)

selected_feat = feature_selection(X, y)
len(selected_feat)  # 72

reg_models(base_models, X_train[selected_feat], y_train, X_test[selected_feat], y_test)
"""
RMSE:
     Scores            Model
0  0.316764  LinearRegressio
1  0.473323  KNeighborsRegre
2  0.438922  DecisionTreeReg
3  0.336551  RandomForestReg
4  0.303149  GradientBoostin
5  0.310346     XGBRegressor
6  0.337694    LGBMRegressor
7  0.287464  <catboost.core.
"""
"""
R2:
     Scores            Model
0  0.898874  LinearRegressio
1  0.774209  KNeighborsRegre
2  0.805838  DecisionTreeReg
3  0.885846  RandomForestReg
4  0.907381  GradientBoostin
5  0.902931     XGBRegressor
6  0.885069    LGBMRegressor
7  0.916717  <catboost.core.
"""
lgbm_model = LGBMRegressor()
lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
               "n_estimators": [100, 1000, 1200, 1500, 5000],
               "max_depth": [-1, 6, 8, 10, 15],
               "colsample_bytree": [1, 0.8, 0.5, 0.4]}

np.exp(np.log1p(165))
lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=False).fit(X[selected_feat], y)
# {'colsample_bytree': 0.4, 'learning_rate': 0.01,'max_depth': 6,'n_estimators': 1500}
lgbm_final = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X[selected_feat], y)
np.mean(np.sqrt(-cross_val_score(lgbm_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.12

y_pred = lgbm_final.predict(X_test[selected_feat])
np.sqrt(mean_squared_error(y_test, y_pred))  # 0.06
r2_score(y_test, y_pred)  # 0.97

xgboost_params = {"learning_rate": [0.1, 0.01, 0.5],
                  "max_depth": [5, 8, 15, 20],
                  "n_estimators": [100, 200, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgb_model = XGBRegressor()
xgb_cv_model = GridSearchCV(xgb_model, xgboost_params, cv=10, n_jobs=-1, verbose=False).fit(X[selected_feat], y)
# {'colsample_bytree': 0.5, learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
xgb_final = XGBRegressor(**xgb_cv_model.best_params_).fit(X[selected_feat], y)
np.mean(np.sqrt(-cross_val_score(xgb_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.12

cart_params = {"max_depth": range(1, 20),
               "min_samples_split": range(2, 30)}

cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params, cv=10).fit(X[selected_feat], y)
# {'max_depth': 16, 'min_samples_split': 27}
cart_final = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X[selected_feat], y)
np.mean(np.sqrt(-cross_val_score(cart_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.18

rf_params = {"max_depth": [3, 5, 8, 10, 15, None],
             "max_features": [5, 10, 15, 20, 50, 100],
             "n_estimators": [200, 500, 1000],
             "min_samples_split": [2, 5, 10, 20, 30, 50]}
rf_model = RandomForestRegressor()
rf_model_cv = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=False).fit(X[selected_feat], y)
# {'max_depth': None, 'max_features': 15, 'min_samples_split': 2, 'n_estimators': 500}
rf_final = RandomForestRegressor(**rf_model_cv.best_params_).fit(X[selected_feat], y)
np.mean(np.sqrt(-cross_val_score(rf_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.13

gbm_params = {"learning_rate": [0.1, 0.01, 0.05],
              "max_depth": [3, 5, 8, 10, 20, 30],
              "n_estimators": [200, 500, 1000, 1500],
              "subsample": [1, 0.4, 0.5, 0.7]}
gbm_model = GradientBoostingRegressor()
gbm_model_cv = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=False).fit(X[selected_feat], y)
# {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1500, 'subsample': 0.5}
gbm_final = GradientBoostingRegressor(**gbm_model_cv.best_params_).fit(X[selected_feat], y)
np.mean(np.sqrt(-cross_val_score(gbm_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.12

extra_tree_model = ExtraTreesRegressor()

extra_tree_params = {"max_depth": [3, 5, 8, 10, 15, None],
                     "max_features": [5, 10, 15, 20, 50, 100],
                     "n_estimators": [200, 500, 1000],
                     "min_samples_split": [2, 5, 10, 20, 30, 50]}

extra_tree_cv = GridSearchCV(extra_tree_model, extra_tree_params, cv=10, n_jobs=-1, verbose=False).fit(X[selected_feat],
                                                                                                       y)
# {'max_depth': 15, 'max_features': 50, 'min_samples_split': 5, 'n_estimators': 500}
extra_tree_final = ExtraTreesRegressor(**extra_tree_cv.best_params_).fit(X[selected_feat], y)
np.mean(
    np.sqrt(-cross_val_score(extra_tree_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.12

lassocv_model = LassoCV(alphas=10 ** np.linspace(10, -2, 100) * 0.5, cv=10).fit(X[selected_feat], y)

lasso_final = Lasso(alpha=lassocv_model.alpha_).fit(X[selected_feat], y)
np.mean(
    np.sqrt(-cross_val_score(lasso_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.14

ridge_model = RidgeCV(alphas=10 ** np.linspace(10, -2, 100) * 0.5, cv=10).fit(X[selected_feat], y)
lasso_final = Ridge(alpha=ridge_model.alpha_).fit(X[selected_feat], y)
np.mean(
    np.sqrt(-cross_val_score(lasso_final, X[selected_feat], y, cv=10, scoring="neg_mean_squared_error")))  # 0.13
