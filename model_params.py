from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

"""# LightGBM

lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
               "n_estimators": [100, 1000, 1200, 1500, 5000],
               "max_depth": [-1, 6, 8, 10, 15],
               "colsample_bytree": [1, 0.8, 0.5, 0.4]}

# XGBoost

xgboost_params = {"learning_rate": [0.1, 0.01, 0.5],
                  "max_depth": [5, 8, 15, 20],
                  "n_estimators": [100, 200, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

# Random Forest

rf_params = {"max_depth": [3, 5, 8, 10, 15, None],
             "max_features": [5, 10, 15, 20, 50, 100],
             "n_estimators": [200, 500, 1000],
             "min_samples_split": [2, 5, 10, 20, 30, 50]}

# GBM

gbm_params = {"learning_rate": [0.001, 0.1, 0.01, 0.05],
              "max_depth": [3, 5, 8, 10, 20, 30],
              "n_estimators": [200, 500, 1000, 1500, 5000],
              "subsample": [1, 0.4, 0.5, 0.7]}

# Extra Tree

extra_tree_params = {"max_depth": [3, 5, 8, 10, 15, None],
                     "max_features": [5, 10, 15, 20, 50, 100],
                     "n_estimators": [200, 500, 1000],
                     "min_samples_split": [2, 5, 10, 20, 30, 50]}
                     
# Models

models = [("LightGBM", LGBMRegressor(), lgbm_params),
          ("XGBoost", XGBRegressor(), xgboost_params),
          ("Random Forest", RandomForestRegressor(), rf_params),
          ("GBM", GradientBoostingRegressor(), gbm_params),
          ("Extra Tree", ExtraTreesRegressor(), extra_tree_params)]
"""


lgbm_params = {'colsample_bytree': [0.4], 'learning_rate': [0.01],'max_depth': [6],'n_estimators': [1500]}
xgboost_params = {'colsample_bytree': [0.5], 'learning_rate': [0.1], 'max_depth': [5], 'n_estimators': [200]}
rf_params = {'max_depth': [None], 'max_features': [15], 'min_samples_split': [2], 'n_estimators': [500]}
gbm_params = {'learning_rate': [0.01], 'max_depth': [5], 'n_estimators': [1500], 'subsample': [0.5]}
extra_tree_params = {'max_depth': [15], 'max_features': [50], 'min_samples_split': [5], 'n_estimators': [500]}

regressors = [("LightGBM", LGBMRegressor(), lgbm_params),
              ("XGBoost", XGBRegressor(), xgboost_params),
              ("Random Forest", RandomForestRegressor(), rf_params),
              ("GBM", GradientBoostingRegressor(), gbm_params),
              ("Extra Tree", ExtraTreesRegressor(), extra_tree_params)]