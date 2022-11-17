import numpy as np
import pandas as pd
import settings as set
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

test_mode = True
results = False
model_name = "Random Forest"


# Regression Metrics Report
def regression_metrics_report(y_true: list, y_pred: list, window: str, os_choice: str):
    # R-Squared is a statistical measure of fit that indicates how much variation of a dependent
    # variable is explained by the independent variable(s) in a regression model.

    r2 = "R-Squared"  # coefficient of determination
    rmse = "Root Mean Squared Error"
    mse = "Mean Squared Error"
    mae = "Mean Absolute Error"
    median_ae = "Median Absolute Error"
    mape = 'Mean Absolute Percentage Error'

    r2_result = metrics.r2_score(y_true, y_pred)
    rmse_result = metrics.mean_squared_error(y_true, y_pred, squared=False)
    # perfect mean squared error value is 0.0 __no variance
    mse_result = metrics.mean_squared_error(y_true, y_pred, squared=True)
    mae_result = metrics.mean_absolute_error(y_true, y_pred)
    median_ae_result = metrics.median_absolute_error(y_true, y_pred)
    mape_result = metrics.mean_absolute_percentage_error(y_true, y_pred)
    std_result = np.std(y_pred)

    print(f"-----Regression {model_name} Metrics Report - local: {set.local}-----\n"
          f"The {window} training window used, with {set.splits_number}, OS: {os_choice}\n"
          f"The overall mean {r2}: {r2_result:.5f}\n"
          f"The model {rmse}: {rmse_result:.5f}\n"
          f"The model {mse}: {mse_result:.5f}\n"
          f"The model {mae}: {mae_result:.5f}\n"
          f"The model {median_ae}: {median_ae_result:.5f}\n"
          f"The model {mape}: {mape_result:.5f}\n"
          f"The standard deviation: {std_result:.5f}\n")


def local_regression_metrics_report(window: str, os_choice: str):
    r2 = "R-Squared"  # coefficient of determination
    rmse = "Root Mean Squared Error"
    mse = "Mean Squared Error"
    mae = "Mean Absolute Error"
    median_ae = "Median Absolute Error"
    mape = 'Mean Absolute Percentage Error'

    r2_result = np.mean(set.r2)
    rmse_result = np.mean(set.rmse)
    mse_result = np.mean(set.mse)
    mae_result = np.mean(set.mae)
    median_ae_result = np.mean(set.median_ae)
    mape_result = np.mean(set.mape)
    std_result = np.std(set.r2)

    print(f"-----Regression {model_name} Metrics Report - local:{set.local}-----\n"
          f"The {window} training window used, with {set.splits_number}, OS: {os_choice}\n"
          f"The overall mean {r2}: {r2_result:.5f}\n"
          f"The mean {rmse}: {rmse_result:.5f}\n"
          f"The mean {mse}: {mse_result:.5f}\n"
          f"The mean {mae}: {mae_result:.5f}\n"
          f"The mean {median_ae}: {median_ae_result:.5f}\n"
          f"The mean{mape}: {mape_result:.5f}\n"
          f"The mean standard deviation: {std_result:.5f}\n")


# Split block technique with window of (1 by 1 month prediction)
def start(data_path1: str, data_path2: str, os_type):
    # Step 01: load the data from file
    df = pd.read_csv(data_path1)
    # df2 = pd.read_csv(data_path2)
    # df2 = df2[df2.year == 2020]
    # df.append(df2)
    # df['date']= pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') # %H:%M:%S
    # print(df.info())
    # print(df.head())

    # Encode the categorical data into numerical (days  - instance)
    enc = LabelEncoder()
    # df["days"] = enc.fit_transform(df["days"])
    # df["encode_days"] = enc.fit_transform(df["days"])

    target = "price"
    features = [
        'month',
        'days',
        'OS',
        'instance',  # We use this as Filter
        'region'
    ]
    features_used = [
        'month',
        'encode_days',
        'encode_OS',
        'encode_instance',  # We use this as Filter
        'encode_region'
    ]

    # data filters after the prediction if you want to use that column as a feature
    year_filter = df[df.year == 2020]
    # os_choice = set.os
    # os_filter = year_filter#[year_filter.OS == os_choice]
    # zone_filter = os_filter[os_filter.region == set.zone]
    instance_filter = year_filter[year_filter.instance.isin(set.instance_list)]

    if test_mode:
        sample = instance_filter
        sample = sample.drop_duplicates(subset=['date'])
        sample['date'] = pd.to_datetime(sample['date'])
        sample = sample.sort_values(by="date", ascending=True)
        sample["encode_instance"] = enc.fit_transform(sample["instance"])
        sample["encode_region"] = enc.fit_transform(sample["region"])
        sample["encode_OS"] = enc.fit_transform(sample["OS"])
        sample["encode_days"] = enc.fit_transform(sample["days"])
        print(f"Count = {str(len(sample))}")
    else:
        sample = df.sort_values(by="date", ascending=True)

    # Step 02: split the date based on time
    time_splitter = TimeSeriesSplit(n_splits=set.splits_number)
    # We need array here
    pre_index = 0
    test_case = 0
    for train_index, test_index in time_splitter.split(sample):
        # Dynamic window --Splits
        if set.fixed:
            # Static(fixed) window --Splits
            window = "Static/Fixed"
            train, test = sample.iloc[[ind for ind in train_index if ind >= pre_index]], \
                          sample.iloc[test_index]
        else:
            window = "Dynamic"
            train, test = sample.iloc[train_index], sample.iloc[test_index]

        # for price
        train_targets = train[target]
        test_targets = test[target]

        # if not set.local: set.y_true.append(test_targets.tolist())

        # for features
        # train_features = train[features_used]
        # test_features = test[features_used]

        # Step 03: train the model
        rf = RandomForestRegressor()
        rf.fit(train[features_used], train_targets)

        # Step 04: test model
        predictions = rf.predict(test[features_used])
        # Here we add the prediction column to the test.
        test['predictions'] = predictions
        test = test.drop(features_used, axis=1)
        # Export the test dataframe
        test_case += 1
        # test.to_csv(f'./test_{test_case}.csv', index=False, header=True)

        if set.local:
            # Step 05: append local results
            for os in set.os:
                data = test[test["OS"] == os]
                for instance in set.instance_list:
                    data = data[data["instance"] == instance]
                    if len(data) > 0:
                        r2 = metrics.r2_score(data[target], data['predictions'])
                        print(f'The r2 of OS = {os} for instance {instance} = {r2}')
            set.r2.append(metrics.r2_score(test_targets, predictions))
            set.mae.append(metrics.mean_absolute_error(test_targets, predictions))
            set.rmse.append(metrics.mean_squared_error(test_targets, predictions, squared=False))
            set.mse.append(metrics.mean_squared_error(test_targets, predictions, squared=True))
            set.median_ae.append(metrics.median_absolute_error(test_targets, predictions))
            set.mape.append(metrics.mean_absolute_percentage_error(test_targets, predictions))
        else:
            # Step 05: append global results
            set.y_pred.append(predictions.tolist())

        # create a fixed train/test window
        if set.fixed: pre_index = len(train_index)

    # if set.local:
    #     local_regression_metrics_report(window, os_choice)
    # else:
    #     # Step 06: see the overall results - Global
    #     regression_metrics_report(set.y_true, set.y_pred, window, os_choice)

    # Step 05: see the overall results - local


if __name__ == "__main__":

    set.init()
    if results:
        for i, os in enumerate(set.os):
            start(set.path_file1, set.path_file2, i)
    else:
        start(set.path_file1, set.path_file2, 2)


