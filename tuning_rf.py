

"""

    tuning_rf.py
    Author: Batool Alkaddah
    Email: batool.kaddah@gmail.com
    This code is part of my Master's Thesis Desseration @Concordia University
    Date: Oct 1st, 2022

"""


import time as time
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from Models.rec_plot.rec_plot.rec import RegressionErrorCharacteristic
import numpy as np

simplefilter("ignore", category=ConvergenceWarning)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

# Safely remove the (SettingCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
import TimeBasedCV
from TimeBasedCV import *
import pandas as pd

# NOTE: change it based on the dictionary: day, week, 2week, month, 2month
split = '2month'

splits_ = {
    'day': 365,  # 1 days training -- 365 iterations or 564
    'week': 7,  # 7 days -- 52 iterations
    '2week': 14,  # 14 days -- 26 iterations
    'month': 30,  # train: 31, test: 30 -- 10 iterations
    '2month': 60  # Training 59, test: 60 -- 5 iterations --
}

# first date to perform the splitting on: The mid between training and testing
date = {
    'day': datetime.date(2021, 1, 2),
    'week': datetime.date(2021, 1, 8),
    '2week': datetime.date(2021, 1, 15),
    'month': datetime.date(2021, 2, 1),
    '2month': datetime.date(2021, 3, 1)
}
operating_systems = [
    'Windows (Amazon VPC)',
    'Red Hat Enterprise Linux (Amazon VPC)',
    'SUSE Linux (Amazon VPC)',
    'Linux/UNIX (Amazon VPC)'
]

maes = []
mses = []
rmses = []
mapes = []
aucs = []
r2s = []
instances = []
operating_system = []

scaler = MinMaxScaler()


def auc_rec(targets, predictions):
    reg_metrics = RegressionErrorCharacteristic(targets, predictions)
    return reg_metrics.auc_rec


def init_data(data_path: str):
    # Step 01: load the data from file
    df = pd.read_csv(data_path)
    df = df[df.year == 2021]

    # Consider the first 10 months from 2021 only
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df = df[df.month.isin(months)]

    df.reset_index(drop=True)
    target = "price"
    features = ['month', 'days', 'instance', 'OS']

    # create copy of the dataFrame
    sample = df.copy()
    sample.reset_index(drop=True)

    # Create labelEncoder() for each feature -- avoid problem with inverse labels
    encoder_map = {}
    for feature in features:
        # For each column, we create one instance in the dictionary.
        # Take care we are only fitting now.
        encoder_map[feature] = LabelEncoder().fit(sample[feature])
    print("encode map", encoder_map)
    # Transform the features
    for feature in features:
        sample[feature] = encoder_map[feature].transform(sample[feature])

    return sample, encoder_map


def start_turning(data_path, region, split):
    target = "price"  # "price"
    features = ['month', 'days', 'instance', 'OS']
    preprocessing = False
    model_name = "RandomForest"

    sample, encoder_map = init_data(data_path)

    # Sort values in ascending order and drop the duplicated date with prices
    sample = sample.sort_values(by="date.1", ascending=True)
    sample['date.1'] = pd.to_datetime(sample['date.1'])

    print(f"Count = {str(len(sample))}")

    # Parameters to tune

    max_depth = [int(x) for x in np.linspace(3, 13, num=7)]
    # Parameters that we want to tune
    random_grid = {
        'max_depth': max_depth,
        'max_features': [1, 2, 3, 4]
    }
    rf = RandomForestRegressor(n_jobs=-1, n_estimators=100)

    model_tuning = RandomizedSearchCV(
        rf,
        param_distributions=random_grid,
        n_iter=10,
        scoring="neg_mean_squared_error",
        n_jobs=6,
        cv=5,
        return_train_score=True
    )

    tscv = TimeBasedCV(train_period=59,
                       test_period=60,
                       freq='days')

    best_global_paramters = {}
    test_id = split
    index = 1
    for train_index, test_index in tscv.split(sample,
                                              validation_split_date=date[split],
                                              date_column='date.1'):
        train, test = sample.loc[train_index], sample.loc[test_index]

        print(f"+ Iteration {index}, Train length:{len(train)}, Test length: {len(test)} in splits: {split}")
        index += 1
        # for price (target)
        train_targets = train[target]
        test_targets = test[target]

        # for features
        train_features = train[features]
        test_features = test[features]

        if preprocessing:
            train_features = scaler.fit_transform(train_features)
            test_features = scaler.transform(test_features)

        # Fit the random search model, NOTE: we used the train data as (train and validate) to avoid the overfitting
        # issue.
        print("+ Start tuning.")
        model_tuning.fit(train_features, train_targets)
        print(f"+ Best parameter for model {model_name}:")
        for parameter in model_tuning.best_params_:
            if parameter in best_global_paramters.keys():
                best_global_paramters[parameter].append(model_tuning.best_params_[parameter])
            else:
                best_global_paramters.update({parameter: [model_tuning.best_params_[parameter]]})
            print(f"+ {parameter} : {model_tuning.best_params_[parameter]}")
        print("--------------------------------------------")

        print(f'+ CV results : {model_tuning.best_score_ * -1}')
        # we chose the best model
        best_model = model_tuning.best_estimator_
        # Now we test the best model with test data that is hide from cross-validation step
        # This point is to avoid the over-fit problem
        predictions = best_model.predict(test_features)
        test['predictions'] = predictions

        # Transform back the features
        for feature in features:
            test[feature] = encoder_map[feature].inverse_transform(test[feature])
        instance_list = test.instance.unique()

        test.to_csv(f"./test_RandomForest_{test_id}{index}_{region}_2021.csv")

        r2 = metrics.r2_score(test_targets, predictions)
        mae = metrics.mean_absolute_error(test_targets, predictions)
        auc = auc_rec(test_targets, predictions)

        print(f'+ Test score r2 = {r2:.3f}, MAE = {mae:.3f}, AUC {auc:.3f}')

        # Find the error metrics for each instance
        for instance in instance_list:
            for os in operating_systems:
                df = test[test['instance'] == instance]
                # add the os since its a feature
                df = df[df['OS'] == os]
                predictions = df['predictions']
                if len(predictions) > 0:
                    targets = df[target].copy()
                    r2 = metrics.r2_score(targets, predictions)
                    mae = metrics.mean_absolute_error(targets, predictions)
                    mse = metrics.mean_squared_error(targets, predictions, squared=True)
                    rmse = metrics.mean_squared_error(targets, predictions, squared=False)
                    mape = metrics.mean_absolute_percentage_error(targets, predictions)
                    auc = auc_rec(targets, predictions)

                    print(f'+ Local predictions of {instance}, {os}: r2 = {r2:.5f}, MAE = {mae:.5f}, AUC {auc:.5f}')

                    # Find the metrics for each instance separately
                    r2s.append(r2)
                    rmses.append(rmse)
                    mses.append(mse)
                    mapes.append(mape)
                    maes.append(mae)
                    aucs.append(auc)
                    instances.append(instance)
                    operating_system.append(os)

    report = pd.DataFrame(best_global_paramters)
    report.to_csv(f"./randomForest_parameters_{region}_{split}_2021.csv", index=None, header=True)


# To iterate over all region without need to change the path everytime
region_list = [
    'us-west-1'
    , 'us-west-2'
    , 'us-east-1'
    , 'ap-northeast-1'
    , 'eu-west-1'
]

if __name__ == "__main__":

    for region in region_list:
        path = f"./all_{region}a_hourly.csv"
        print(time.strftime("%Y%m%d-%H%M%S"))
        start_turning(path, region, split)

        results_path = f'final_error_report_RF_{region}_{split}'

        # Export report
        frame = {
            "Instance": instances,
            "OS": operating_system,
            "mae": maes,
            "rmse": rmses,
            "mape": mapes,
            "r2": r2s,
            "AUC": aucs,
        }

        report = pd.DataFrame(frame)
        report.to_csv(f"{results_path}_2021.csv", index=None, header=True)
