import numpy as np
import pandas as pd
from sklearn import metrics
import time as time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# simplefilter("ignore", category=ConvergenceWarning)
from RModel import Model
from sklearn.neighbors import KNeighborsRegressor
from rec import RegressionErrorCharacteristic
import TimeBasedCV
from TimeBasedCV import *

# Safely remove the (SettingCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# Error report columns
maes = []
mses = []
rmses = []
mapes = []
aucs = []
r2s = []
instances = []
operating_system = []
model = []
models = []

region_list = [
    'us-west-1'
    , 'us-west-2'
    , 'us-east-1'
    , 'ap-northeast-1'
    , 'eu-west-1'
]

target = "price"  # "price"
features = ['month', 'days', 'instance', 'OS']
split_window = 'day'
# region = 'us-west-1'
year = 2020

target = "price"  # "price"
features = ['month', 'days', 'instance', 'OS']


def auc_rec(targets, predictions):
    reg_metrics = RegressionErrorCharacteristic(targets, predictions)
    return reg_metrics.auc_rec


def train_test_model(model, features, targets, test_features, scaling):

    # Train the model
    model.fit(features, targets)
    # Test the model and get the predictions
    predictions = model.predict(test_features)

    return predictions


def model_in_list(model):
    for id, m in enumerate(models):
        if m.model_name == model:
            return id
    return -1


def update_report(model, color, predictions, targets):
    index = model_in_list(model)
    if index > -1:
        # update
        models[index].predictions.append(predictions)
        models[index].targets.append(targets)
        print(f'+ Model {model} is updated with {str(len(predictions))} new rows', end='\n')
    else:
        # create new one and add it to list
        m = Model(model_name=model, color=color)
        m.predictions.append(predictions)
        m.targets.append(targets)
        models.append(m)


def save_predictions(test, predictions, encoder_map, index, model):
    test['predictions'] = predictions
    # Transform back the features
    for feature in features:
        test[feature] = encoder_map[feature].inverse_transform(test[feature])

    test.to_csv(f"./predictions_{model}_{split_window}_{index}_{region}_{year}.csv")


def error_report(test, model_name, encoder_map):
    # Transform back the features
    for feature in features:
        test[feature] = encoder_map[feature].inverse_transform(test[feature])

    instance_list = test.instance.unique()
    operating_systems = test.OS.unique()

    # Find the error metrics for each instance
    for instance in instance_list:
        for os in operating_systems:
            df = test[test['instance'] == instance]
            # add the os since its a feature
            df = df[df['OS'] == os]
            predictions = df['predictions']
            targets = df[target].copy()

            r2 = metrics.r2_score(targets, predictions)
            mae = metrics.mean_absolute_error(targets, predictions)
            mse = metrics.mean_squared_error(targets, predictions, squared=True)
            rmse = metrics.mean_squared_error(targets, predictions, squared=False)
            mape = metrics.mean_absolute_percentage_error(targets, predictions)
            auc = auc_rec(targets, predictions)

            # Find the metrics for each instance separately
            r2s.append(r2)
            rmses.append(rmse)
            mses.append(mse)
            mapes.append(mape)
            maes.append(mae)
            aucs.append(auc)
            instances.append(instance)
            operating_system.append(os)
            model.append(model_name)


def start(data_path: str):
    models = []

    # Error report columns
    maes = []
    mses = []
    rmses = []
    mapes = []
    aucs = []
    r2s = []
    instances = []
    operating_system = []
    model = []

    df = pd.read_csv(data_path)
    df = df[df.year == year]
    df.reset_index(drop=True)

    sample = df.copy()
    print(len(sample))
    target = "price"  # "price"
    features = ['month', 'days', 'instance', 'OS']
    # Create labelEncoder() for each feature -- avoid problem with inverse labels
    encoder_map = {}
    for feature in features:
        # For each column, we create one instance in the dictionary.
        # Take care we are only fitting now.
        encoder_map[feature] = LabelEncoder().fit(sample[feature])

    # Transform the features
    for feature in features:
        sample[feature] = encoder_map[feature].transform(sample[feature])

    # Sort values in ascending order and drop the duplicated date with prices
    sample = sample.sort_values(by="date.1", ascending=True)
    sample['date.1'] = pd.to_datetime(sample['date.1'])

    # Step 02: split the date based on time
    split = {
        'day': 1,  # 1 days training -- 365 iterations or 564
        'week': 7,  # 7 days -- 52 iterations
        '2week': 14,  # 14 days -- 26 iterations
        'month': 30,  # train: 31, test: 30 -- 10 iterations
        '2month': 60  # Training 59, test: 60 -- 5 iterations --
    }

    date = {
        'day': datetime.date(2020, 1, 2),
        'week': datetime.date(2020, 1, 8),
        '2week': datetime.date(2020, 1, 15),
        'month': datetime.date(2020, 2, 1),
        '2month': datetime.date(2020, 3, 1)
    }

    time_splitter = TimeBasedCV(train_period=1,
                                test_period=1,
                                freq='days')

    # We need array here
    index = 1
    for train_index, test_index in time_splitter.split(sample,
                                                       validation_split_date=date[split_window],
                                                       date_column='date.1'):
        train, test = sample.loc[train_index], sample.loc[test_index]

        # for price
        train_targets = train[target]
        test_targets = test[target]
        # for features
        train_features = train[features]
        test_features = test[features]

        # 1 XGBoost model
        parameters1 = pd.read_excel(f"./parameters/xgboost_{year}.xlsm",
                                    sheet_name=['day', 'week', '2week', 'month', '2month'], index_col=0)
        p1 = parameters1[split_window]

        xgb_regressor = XGBRegressor(objective='reg:squarederror',
                                     colsample_bytree=p1.loc['colsample_bytree', region],
                                     eta=p1.loc['eta', region],
                                     n_estimators=int(p1.loc['n_estimators', region]),
                                     subsample=p1.loc['subsample', region],
                                     max_depth=int(p1.loc['max_depth', region]),
                                     n_jobs=6)

        predictions = train_test_model(model=xgb_regressor, features=train_features, targets=train_targets,
                                       test_features=test_features)
        update_report("XGBoost", "olive", predictions=predictions, targets=test_targets)

        # Save the results of this time split
        test_sample = test.copy()
        test_sample['predictions'] = predictions
        error_report(test_sample, 'xgb', encoder_map)
        print("+ Done xgb")
        # save_predictions(test_sample, predictions, encoder_map, index, "xgb")

        # 2 SVR model
        parameters2 = pd.read_excel(f"./parameters/svr_{year}.xlsm",
                                    sheet_name=['day', 'week', '2week', 'month', '2month'], index_col=0)
        p2 = parameters2[split_window]

        svr_regressor = SVR(max_iter=10_000, kernel=p2.loc['kernel', region],
                            C=p2.loc['C', region], gamma=p2.loc['gamma', region])

        predictions = train_test_model(model=svr_regressor, features=train_features, targets=train_targets,
                                       test_features=test_features)
        update_report("SVR", "mediumseagreen", predictions=predictions, targets=test_targets)
        # Save the results of this time split
        test_sample = test.copy()
        test_sample['predictions'] = predictions
        error_report(test_sample, 'svr', encoder_map)

        # 3 Random Forest
        parameters3 = pd.read_excel(f"./parameters/rf_{year}.xlsm",
                                    sheet_name=['day', 'week', '2week', 'month', '2month'], index_col=0)
        p3 = parameters3[split_window]
        rf = RandomForestRegressor(n_estimators=100,
                                   n_jobs=-1,
                                   max_depth=p3.loc['max_depth', region],
                                   max_features=p3.loc['max_features', region])
        predictions = train_test_model(model=rf, features=train_features, targets=train_targets,
                                       test_features=test_features)
        update_report("RFR", "blue", predictions=predictions, targets=test_targets)
        # Save the results of this time split
        test_sample = test.copy()
        test_sample['predictions'] = predictions
        error_report(test_sample, 'rf', encoder_map)

        # 4 KNN Model
        parameters4 = pd.read_excel(f"./parameters/knn_{year}.xlsm",
                                    sheet_name=['day', 'week', '2week', 'month', '2month'], index_col=0)
        p4 = parameters4[split_window]
        knn = KNeighborsRegressor(n_neighbors=p4.loc['n_neighbors', region],
                                  algorithm=p4.loc['algorithm', region],
                                  n_jobs=6)
        predictions = train_test_model(model=knn, features=train_features, targets=train_targets,
                                       test_features=test_features)
        update_report("KNN", "orange", predictions=predictions, targets=test_targets)
        # Save the results of this time split
        test_sample = test.copy()
        test_sample['predictions'] = predictions
        error_report(test_sample, 'knn', encoder_map)

        index += 1

        results_path = f'all_error_report_{region}_{split_window}'

    # Export report
    frame = {
        "Instance": instances,
        "OS": operating_system,
        "mae": maes,
        "mse": mses,
        "rmse": rmses,
        "mape": mapes,
        "r2": r2s,
        "AUC": aucs,
        "model": model
    }

    report = pd.DataFrame(frame)
    report.to_csv(f"{results_path}_{year}.csv", index=None, header=True)
    print("+ Start ploting Time now: ", time.strftime("%Y%m%d-%H%M%S"))

    # Config the draw canvas
    print("+ Finished and plotting the graph.")
    plt.figure(figsize=(6, 6), dpi=250, edgecolor="black")
    plt.xlabel("Deviation", fontsize=8)
    plt.ylabel("Accuracy", fontsize=8)
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], linestyle="--", lw=0.8, color="r", label="Random Model", alpha=0.75)
    patterns = ['-', '-.', ':', '--']
    markers = ['o', 'v', '^', '*']
    for i, m in enumerate(models):
        m.prepare()
        plt.plot(m.reg_metrics.deviation, m.reg_metrics.accuracy,
                 lw=0.6,
                 label=f"{m.model_name} AUC = {m.reg_metrics.auc_rec:.3f}",
                 alpha=0.95,
                 color=m.color,
                 linestyle=patterns[i % len(patterns)])
        print(f'+ plot model {m.model_name} is done.')
    plt.legend(prop={"size": 6})
    plt.savefig(f"./REC_fixed_{region}_{split_window}.png")
    plt.savefig(f"./REC_fixed_{region}_{split_window}.pdf")

    print("+ Done ploting.. ")
    # plt.show()

if __name__ == "__main__":
    for region in region_list:
        start(f"./all_{region}a_hourly.csv")
