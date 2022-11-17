import time as time
import datetime
import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer
import pandas as pd
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from rec import RegressionErrorCharacteristic

simplefilter("ignore", category=ConvergenceWarning)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# Safely remove the (SettingCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
import TimeBasedCV
from TimeBasedCV import *
import pandas as pd


"""

This code used to tune the k-NNR model to find the best hyper-parameters  using Randomized Search
AWS zone names: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html

"""

# NOTE: change it based on the dictionary: day, week, 2week, month, 2month
split = 'month'

splits_ = {
    'day': 1,  # 1 days training -- 365 iterations or 564
    'week': 7,  # 7 days -- 52 iterations
    '2week': 14,  # 14 days -- 26 iterations
    'month': 30,  # train: 31, test: 30 -- 10 iterations
    '2month': 60  # Training 59, test: 60 -- 5 iterations --
}

# first date to perform the splitting on: The mid between training and testing
# -- change dates based on your data and records
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


# create columns as lists of the final dataFrame errro report
maes = []
mses = []
rmses = []
mapes = []
aucs = []
r2s = []  # no need for r2 in this problem
instances = []
operating_system = []


def auc_rec(targets, predictions):
    """
     This function return the auc-rec values

     :param targets:
     :param predictions:
     :return: auc_rec
     """
    reg_metrics = RegressionErrorCharacteristic(targets, predictions)
    return reg_metrics.auc_rec


def init_data(data_path: str):
    # Step 01: load the data from file
    df = pd.read_csv(data_path)

    # choose the year: 2019, 2020, 2021
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
        # For each column, we create one instance in the dictionary, we are only fitting now.
        encoder_map[feature] = LabelEncoder().fit(sample[feature])

    # Transform the features
    for feature in features:
        sample[feature] = encoder_map[feature].transform(sample[feature])

       return sample, encoder_map


def start_turning(data_path, region, split):
    target = "price"  # "price"
    features = ['month', 'days', 'instance', 'OS']
    model_name = "XGBoost"

    sample, encoder_map = init_data(data_path)

    # Sort values in ascending order and drop the duplicated date with prices
    sample = sample.sort_values(by="date.1", ascending=True)
    sample['date.1'] = pd.to_datetime(sample['date.1'])

    print(f"Count = {str(len(sample))}")

    # Parameters that we want to tune
    random_grid = {
        'eta':
            [
                0.05,
                0.1,
                0.5,
                1.0
            ],
        'n_estimators':
            [
                50,
                70,
                100,
                150,
                200,
                250
            ],
        'subsample':
            [
                0.6,
                0.8,
                1.0,
            ],
        'colsample_bytree':
        # colsample_bytree defines what percentage of features ( columns )
        # will be used for building each tree.
            [
                0.6,
                0.8,
                1.0,
            ],
        'max_depth':
            [int(x) for x in np.linspace(3, 13, num=7)],

    }
    # First create the base model to tune
    xgb = XGBRegressor(objective='reg:squarederror')

    model_tuning = RandomizedSearchCV(
        xgb,
        param_distributions=random_grid,
        n_iter=10,
        scoring="neg_mean_squared_error",
        n_jobs=6, cv=5, return_train_score=True
    )


    tscv = TimeBasedCV(train_period=30,  # change numbers or keep it as it is based on the training window (month, day, week)
                       test_period=31,
                       freq='days')

    best_global_paramters = {}
    test_id = split
    index = 0
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

        # Fit the random search model
        # NOTE: we used the train data as (train and validate) to avoid the overfitting issue.

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

        test.to_csv(f"./test_XGboost_{test_id}_{region}_2021.csv")

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
    report.to_csv(f"./xgboost_parameters_{region}_{split}_2021.csv", index=None, header=True)


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
        print(split)
        start_turning(path, region, split)
        results_path = f'final_error_report_XGB_{region}_{split}'

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