import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
import statsmodels.api as sm

num_feat_colors = ['#FFB6B9', '#61C0BF']
cat_feat_colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]


def grab_col_names(df, cat_th=10, car_th=30):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
            df: Dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                threshold value for numeric but categorical variables
        car_th: int, optinal
                threshold value for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical but cardinal variable list

    Examples
    ------
        You just need to call the function and send the dataframe.

        --> grab_col_names(df)

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 returned lists equals the total number of variables:
        cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                   df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=False)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df).sort_values(by="Ratio", ascending=False)
    return missing_df


def cat_plots(dataframe, cat_col, target):
    print("#" * 70)
    print(dataframe.groupby(cat_col).agg({target: ["count", "mean", "median"]}))
    print("#" * 70)
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    plt.suptitle(cat_col.capitalize(), size=16)
    plt.subplot(1, 3, 1)
    plt.title("Percentages")
    plt.pie(dataframe[cat_col].value_counts().values.tolist(),
            labels=dataframe[cat_col].value_counts().keys().tolist(),
            labeldistance=1.1,
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
            colors=cat_feat_colors,
            autopct='%1.0f%%')

    plt.subplot(1, 3, 2)
    plt.title("Countplot")
    sns.countplot(data=dataframe, x=cat_col, palette=cat_feat_colors)

    plt.subplot(1, 3, 3)
    sns.barplot(data=dataframe, x=cat_col, y=target, palette=cat_feat_colors)

    plt.tight_layout(pad=3)
    plt.show(block=True)


def num_summary(dataframe, col_name, target):
    quantiles = [.01, .05, .1, .5, .9, .95, .99]
    print("#" * 70)
    print(dataframe[col_name].describe(percentiles=quantiles))
    print("#" * 70)

    plt.figure(figsize=(15, 10))
    plt.suptitle(col_name.capitalize(), size=16)
    plt.subplot(1, 3, 1)
    plt.title("Histogram")
    sns.histplot(dataframe[col_name], color=num_feat_colors[0])

    plt.subplot(1, 3, 2)
    plt.title("Box Plot")
    sns.boxplot(data=dataframe, y=col_name, color=num_feat_colors[1])

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=dataframe, y=col_name, x=target)
    plt.tight_layout(pad=1.5)
    plt.show(block=True)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_columns = [col for col in cat_cols if (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df


def fill_na_for_train(dataframe):
    missing_df = missing_values_analysis(dataframe)
    garage_cols = missing_df.index[missing_df.index.str.contains("Garage")]
    for col in garage_cols:
        dataframe[col].fillna("No_Garage", inplace=True)
    dataframe["GarageYrBlt"].replace("No_Garage", 0, inplace=True)

    bsmnt_cols = missing_df.index[missing_df.index.str.contains("Bsmt")]
    for col in bsmnt_cols:
        dataframe[col].fillna("No_Basement", inplace=True)
    dataframe["Electrical"].fillna(dataframe["Electrical"].mode()[0], inplace=True)
    dataframe["MasVnrType"].fillna("None", inplace=True)
    return dataframe


def fill_na_for_test(dataframe):
    missing_df = missing_values_analysis(dataframe)
    garage_cols = missing_df.index[missing_df.index.str.contains("Garage")]
    no_cols = dataframe[garage_cols].dtypes == "O"
    for col in no_cols[no_cols].index.tolist():
        dataframe[col].fillna("No_Garage", inplace=True)

    zero_cols = dataframe[garage_cols].dtypes != "O"
    for col in zero_cols[zero_cols].index.tolist():
        dataframe[col].fillna(0, inplace=True)

    bsmnt_cols = missing_df.index[missing_df.index.str.contains("Bsmt")]
    no_cols = dataframe[bsmnt_cols].dtypes == "O"
    for col in no_cols[no_cols].index.tolist():
        dataframe[col].fillna("No_Basement", inplace=True)

    zero_cols = dataframe[bsmnt_cols].dtypes != "O"
    for col in zero_cols[zero_cols].index.tolist():
        dataframe[col].fillna(0, inplace=True)

    fill_mode_cols = ["Functional", "Utilities", "KitchenQual", "Exterior2nd", "Exterior1st", "SaleType", "MasVnrType",
                      "MSZoning"]
    for col in fill_mode_cols:
        dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
    return dataframe


def encoding(dataframe):
    tmp_df = rare_encoder(dataframe, 0.01)
    drop_rare = ["Street", "Utilities", "RoofMatl", "Heating", "Condition2"]
    tmp_df.drop(drop_rare, axis=1, inplace=True)
    dataframe = pd.get_dummies(dataframe, drop_first=True)
    return dataframe


def knn_for_na(dataframe):
    scaler = RobustScaler()
    dff = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
    imputer = KNNImputer(n_neighbors=5)
    dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
    dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
    return dff


def outlier_threshold(dataframe, variable, q1=.01, q3=.99):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(
        lambda x: up_limit if x > up_limit else low_limit if x < low_limit else x)


def feature_selection(X, y):
    cols = X.columns
    # Backward Elimination
    cols = list(X.columns)
    pmax = 1
    while (len(cols) > 0):
        p = []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    return selected_features_BE


def compute_feature_importance(voting_clf, weights, X, len_features=30):
    """ Function to compute feature importance of Voting Classifier """

    feature_importance = dict()
    for est in voting_clf.estimators_:
        feature_importance[str(est)] = est.feature_importances_

    fe_scores = [0] * len(list(feature_importance.values())[0])
    for idx, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score * weights[idx]
        fe_scores = list(np.add(fe_scores, list(imp_score_with_weight)))

    feature_importances = pd.DataFrame()
    feature_importances["Feature"] = X.columns
    feature_importances["Feature Importance"] = fe_scores
    feature_importances = feature_importances.sort_values("Feature Importance", ascending=False).head(len_features)

    plt.figure(figsize=(10, 10))
    sns.barplot(x="Feature Importance", y="Feature", data=feature_importances)
    plt.title("Features")
    plt.tight_layout()
    plt.savefig("outputs/feature_importances.png")
    plt.show(block=True)
    return feature_importances


def tree_graph(model, tree_idx):
    from lightgbm import plot_tree

    plot_tree(model, tree_index=tree_idx, dpi=300)
    plt.savefig("outputs/lgbm_tree_graph.png")
    plt.show(block=True)


def plot_reg(y, y_pred):
    plt.title("Actual vs Predicted")
    sns.regplot(np.exp(y), y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.show(block=True)
