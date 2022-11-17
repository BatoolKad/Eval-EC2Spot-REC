import time as time
import matplotlib.pyplot as plt
from RModel import Model
from rec import RegressionErrorCharacteristic
from TimeBasedCV import *
import glob

# Safely remove the (SettingCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'

"""
This code is for plotting (each instance/OS) 
Note: Plotting all instance along the OS after combining all data from all regions

Code structure:
combine all the regions together, filter Instance, filter OS -- plot

"""

models = []

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

region_list = [
    'us-west-1',
    'us-west-2'
    , 'us-east-1'
    , 'ap-northeast-1'
    , 'eu-west-1'
]
target = "price"  # "price"
features = ['month', 'days', 'instance', 'OS']
split_window = 'day'
year = 2020


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
        models[index].predictions = predictions.tolist()
        models[index].targets = targets.tolist()
        print(f'+ Model {model} is updated with {str(len(predictions))} new rows', end='\n')
    else:
        # create new one and add it to list
        m = Model(model_name=model, color=color)
        m.predictions = predictions.tolist()
        m.targets = targets.tolist()
        models.append(m)


def start(df, op_sys: str, instance: str):
    test_targets = df['price']

    # 1 XGBoost model
    update_report("XGBoost", "olive", predictions=df['price_xgb'], targets=test_targets)

    # 2 SVR model
    update_report("SVR", "mediumseagreen", predictions=df['price_svr'], targets=test_targets)

    # 3 Random Forest
    update_report("RFR", "blue", predictions=df['price_rf'], targets=test_targets)

    # 4 KNN Model
    update_report("KNN", "orange", predictions=df['price_knn'], targets=test_targets)

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
    for i, m in enumerate(models):
        m.prepare()
        plt.plot(m.reg_metrics.deviation, m.reg_metrics.accuracy,
                 lw=0.6,
                 label=f"{m.model_name} AUC = {m.reg_metrics.auc_rec:.3f}",
                 alpha=0.95,
                 color=m.color,
                 linestyle=patterns[i % len(patterns)])
        print(f'+ plot model {m.model_name} is done.')
    # plt.grid(axis='y')
    plt.legend(prop={"size": 6})
    op_sys1 = op_sys.split("/")[0]
    plt.savefig(f"./REC_regions_{instance}_{op_sys1}_{split_window}_{year}.png")
    plt.savefig(f"./REC_regions_{instance}_{op_sys1}_{split_window}_{year}.pdf")


if __name__ == "__main__":

    paths = glob.glob("C:/Users/baty/IdeaProjects/Models/Predictions/*_day_2020.csv")
    print(paths)
    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs)
    print("All df length: ",len(df))
    df['OS'] = df['OS'].map(lambda x: x.rstrip('(Amazon VPC)'))
    operating_systems = df.OS.unique()

    for instance in instance_list:
        for os in operating_systems:
            new_df = df[(df.instance == instance) & (df.OS == os)]
            print("+ Instance + OS df length: ",len(new_df))
            start(new_df, os, instance)
