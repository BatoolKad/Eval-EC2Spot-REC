import pandas as pd
import os
from time import sleep

# add file PATH
path = ""

def append(path):
    df = pd.DataFrame()
    info = pd.DataFrame()
    for dir in os.listdir(path):
        df = pd.DataFrame()
        if dir == ".DS_Store":
            continue
        name = f"{dir}"
        dirPath = os.path.join(path, dir)
        for file in os.listdir(dirPath):
            if file == ".DS_Store":
                continue
            filePath = os.path.join(dirPath, file)
            df_2 = pd.read_csv(f"{filePath}")
            df = df.append(df_2,ignore_index=True)
            #print(df.head())
            print(f"file{file} is done!")
        print(df.head())
        # df['date'] = df['date'].map(lambda x: str(x)[:-6])  # remove unwanted time fraction -- n
        #df["date"] = df["date"].str.split("T", n=1, expand=True)
        df["date"] = pd.to_datetime(df["date"])
        df["days"] = df["date"].dt.day_name()
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        dataInfo = df.info()
        info = info.append(dataInfo)
        # Add full PATH
        df.to_csv(f"/All_{dir}.csv", index=0)
        sleep(10)



if __name__ == '__main__':
    append(path)
