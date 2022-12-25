from helpers import *
import warnings
import time

warnings.filterwarnings("ignore")


def get_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    df = pd.concat((train, test)).reset_index(drop=True)
    df.drop(["PoolQC", "Id", "MiscFeature", "Alley", "Fence", "FireplaceQu"], axis=1, inplace=True)  # NA Cols
    train_tmp = fill_na_for_train(train)
    test_tmp = fill_na_for_test(test)
    df_tmp = pd.concat((train_tmp, test_tmp)).reset_index(drop=True)
    df = df_tmp[df.columns]
    return df


def encode_(dataframe):
    dataframe['YearBuilt'] = dataframe['YearBuilt'].astype(int)
    dataframe['YearRemodAdd'] = dataframe['YearRemodAdd'].astype(int)
    dataframe['YrSold'] = dataframe['YrSold'].astype(int)
    dataframe["MSSubClass"] = dataframe["MSSubClass"].astype(str)
    neigh_map = {'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1, 'BrkSide': 2, 'OldTown': 2, 'Edwards': 2,
                 'Sawyer': 3, 'Blueste': 3, 'SWISU': 3, 'NPkVill': 3, 'NAmes': 3, 'Mitchel': 4,
                 'SawyerW': 5, 'NWAmes': 5, 'Gilbert': 5, 'Blmngtn': 5, 'CollgCr': 5,
                 'ClearCr': 6, 'Crawfor': 6, 'Veenker': 7, 'Somerst': 7, 'Timber': 8,
                 'StoneBr': 9, 'NridgHt': 10, 'NoRidge': 10}
    dataframe['Neighborhood'] = dataframe['Neighborhood'].map(neigh_map).astype('int')

    dataframe.loc[dataframe["Exterior1st"] == "CBlock", "Exterior1st"] = "AsbShng"
    dataframe.loc[dataframe["Exterior1st"] == "AsphShn", "Exterior1st"] = "AsbShng"
    dataframe.loc[dataframe["Exterior1st"] == "BrkComm", "Exterior1st"] = "AsbShng"
    dataframe.loc[dataframe["Exterior1st"] == "ImStucc", "Exterior1st"] = "CemntBd"
    dataframe.loc[dataframe["Exterior1st"] == "Stone", "Exterior1st"] = "CemntBd"
    dataframe.loc[dataframe["Exterior1st"] == "BrkComm", "Exterior1st"] = "AsbShng"
    dataframe.loc[dataframe["Exterior1st"] == "Wd Sdng", "Exterior1st"] = "MetalSd"

    dataframe.loc[dataframe["Exterior2nd"] == "CBlock", "Exterior2nd"] = "AsbShng"
    dataframe.loc[dataframe["Exterior2nd"] == "AsphShn", "Exterior2nd"] = "Brk Cmn"
    dataframe.loc[dataframe["Exterior2nd"] == "ImStucc", "Exterior2nd"] = "VinylSd"
    dataframe.loc[dataframe["Exterior2nd"] == "Other", "Exterior2nd"] = "VinylSd"
    dataframe.loc[dataframe["Exterior2nd"] == "Stone", "Exterior2nd"] = "Stucco"
    dataframe.loc[dataframe["Exterior2nd"] == "Brk Cmn", "Exterior2nd"] = "Wd Sdng"

    ordinal_cols = ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
                    "BsmtFinType1", "BsmtFinType2", "GarageQual", "GarageCond", "HeatingQC", "KitchenQual",
                    "GarageCond"]

    for feature in ordinal_cols:
        labels_ordered = dataframe.groupby([feature])['SalePrice'].mean().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        dataframe[feature] = dataframe[feature].map(labels_ordered)

    dataframe['FullBath'] = dataframe['FullBath'].astype(int)
    dataframe['HalfBath'] = dataframe['HalfBath'].astype(int)
    dataframe['BsmtFullBath'] = dataframe['BsmtFullBath'].astype(int)
    dataframe['BsmtHalfBath'] = dataframe['BsmtHalfBath'].astype(int)

    return dataframe


def feature_engineering(train_path, test_path):
    print("Feature Engineering Started")
    start_time = time.perf_counter()
    dataframe = get_data(train_path, test_path)
    dataframe = encode_(dataframe)
    dataframe['TotalSF'] = (
            dataframe['BsmtFinSF1'] + dataframe['BsmtFinSF2'] + dataframe['1stFlrSF'] + dataframe['2ndFlrSF'])

    dataframe['TotalBathrooms'] = (dataframe['FullBath'] + (0.5 * dataframe['HalfBath']) + dataframe['BsmtFullBath'] + (
            0.5 * dataframe['BsmtHalfBath']))

    dataframe['TotalPorchSF'] = (
            dataframe['OpenPorchSF'] + dataframe['3SsnPorch'] + dataframe['EnclosedPorch'] + dataframe['ScreenPorch'] +
            dataframe['WoodDeckSF'])

    dataframe['Building_Age'] = (dataframe['YearBuilt'].max() - dataframe['YearBuilt'])
    dataframe['Selling_Age'] = (dataframe['YrSold'] - dataframe['YearBuilt'])

    dataframe['TotalExtQual'] = (dataframe['ExterQual'] + dataframe['ExterCond'])
    dataframe['TotalBsmQual'] = (
            dataframe['BsmtQual'] + dataframe['BsmtCond'] + dataframe['BsmtFinType1'] + dataframe['BsmtFinType2'])
    dataframe['TotalGrgQual'] = (dataframe['GarageQual'] + dataframe['GarageCond'])

    dataframe['TotalQual'] = (dataframe['OverallQual'] + dataframe['TotalExtQual'] + dataframe['TotalBsmQual'] +
                              dataframe['TotalGrgQual'] + dataframe['KitchenQual'] + dataframe['HeatingQC'])

    dataframe['Has_Garage'] = dataframe['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    dataframe["2nd_Floors"] = dataframe['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    dataframe.drop(["YrSold", "PoolArea", "YearRemodAdd", "YrSold", "MoSold", "GarageYrBlt",
                    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], axis=1, inplace=True)

    tmp = encoding(dataframe)
    train = tmp.loc[tmp["SalePrice"].notnull()]
    test_df = tmp.loc[tmp["SalePrice"].isnull()]
    test_df.drop("SalePrice", axis=1, inplace=True)
    dff_train = knn_for_na(train)
    dff_test = knn_for_na(test_df)
    cat_cols, num_cols, cat_but_car = grab_col_names(dff_test)

    for col in num_cols:
        replace_with_thresholds(dff_train, col)
        replace_with_thresholds(dff_test, col)

    scaler_train = RobustScaler()
    X = dff_train.drop("SalePrice", axis=1)
    X = pd.DataFrame(scaler_train.fit_transform(X), columns=X.columns)

    y = dff_train["SalePrice"]
    y = np.log1p(y).to_numpy().ravel()

    scaler_test = RobustScaler()
    test_df = pd.DataFrame(scaler_test.fit_transform(dff_test), columns=dff_test.columns)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"İşlemin süresi: {elapsed_time:.6f} saniye")
    print("Feature Engineering Finished")
    return X, y, test_df
