import numpy as np
import pandas as pd
from sklearn import metrics
import time as time
import matplotlib.pyplot as plt

# simplefilter("ignore", category=ConvergenceWarning)
from RModel import Model
from sklearn.neighbors import KNeighborsRegressor
from rec import RegressionErrorCharacteristic
import TimeBasedCV
from TimeBasedCV import *

# Safely remove the (SettingCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'

""" 
This code draw the REC curve 
for 4 models for each instance/OS for each region
"""

models = []
operating_systems = [
    'Windows (Amazon VPC)',
    'Red Hat Enterprise Linux (Amazon VPC)',
    'SUSE Linux (Amazon VPC)',
    'Linux/UNIX (Amazon VPC)'
]

region_list = [
    'us-west-1'
    , 'us-west-2'
    , 'us-east-1'
    , 'ap-northeast-1'
    , 'eu-west-1'
]

os = [
    'Windows (Amazon VPC)',
    'Red Hat Enterprise Linux (Amazon VPC)',
    'SUSE Linux (Amazon VPC)',
    'Linux/UNIX (Amazon VPC)'
]

instance_list = [
    'c3.large',
    'c3.medium',
    'c3.xlarge',
    'c3.2xlarge',
    'c3.4xlarge',
    'c3.8xlarge',
    'c4.medium',
    'c4.large',
    'c4.xlarge',
    'c4.2xlarge',
    'c4.4xlarge',
    'c4.8xlarge'
]

maes = []
mses = []
rmses = []
mapes = []
aucs = []
r2s = []
instances = []
operating_system = []
model = []
instance = []

# global parameters

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


def start(data_path: str, region: str):
    df = pd.read_csv(data_path)
    df = df.sort_values(by="date.1", ascending=True)
    instance_list = df.instance.unique()
    os_list = df.OS.unique()

    for instance in instance_list:
        for os in os_list:
            df_os_instance = df[(df.OS == os) & (df.insatnce == instance)]
            test_targets = df_os_instance['price']

            # 1 XGBoost model
            update_report("XGBoost", "olive", predictions=df_os_instance['price_xgb'], targets=test_targets)

            # 2 SVR model
            update_report("SVR", "mediumseagreen", predictions=df_os_instance['price_svr'], targets=test_targets)

            # 3 Random Forest
            update_report("RFR", "blue", predictions=df_os_instance['price_rf'], targets=test_targets)

            # 4 KNN Model
            update_report("KNN", "orange", predictions=df_os_instance['price_knn'], targets=test_targets)

            start_time = time.strftime("%Y%m%d-%H%M%S")
            print("+ Start plotting Time now: ", start_time)
            # Config the draw canvas
            print("+ Start plotting the graph.")
            plt.figure(figsize=(6, 6), dpi=250, edgecolor="black")
            plt.xlabel("Deviation", fontsize=8)
            plt.ylabel("Accuracy", fontsize=8)
            plt.ylim([0, 1])
            plt.plot([0, 1], [0, 1], linestyle="--", lw=0.8, color="r", label="Random Model", alpha=0.75)
            patterns = ['-', '-.', ':', '--']
            # markers = ['o', 'v', '^', '*']
            for i, m in enumerate(models):
                m.prepare()
                plt.plot(m.reg_metrics.deviation, m.reg_metrics.accuracy,
                         lw=0.6,
                         label=f"{m.model_name} AUC = {m.reg_metrics.auc_rec:.3f}",
                         alpha=0.95,
                         color=m.color,
                         linestyle=patterns[i % len(patterns)])
                print(f'+ plot model {m.model_name} is done.')
            # plt.grid(axis='both')
            plt.legend(prop={"size": 6})

            plt.savefig(f"./REC_{instance}_{os}_{region}_{split_window}.png")
            plt.savefig(f"./REC_fixed_{instance}_{os}_{region}_{split_window}.pdf")
            end_ime = time.strftime("%Y%m%d-%H%M%S")
            print(f"+ Done plotting {instance}/{os} it takes me: {start_time - end_ime} to finish!")


if __name__ == "__main__":

    for region in region_list:
        start(f"./predictions_model_{region}a.csv", region)
