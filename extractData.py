import lzma
import os
import pandas as pd

# This code-- to extract the data from .xz files
# --convert them to .txt then .csv files
#Add PATH
directory =""
dirPath = f"{directory}/us-east-1"

# iterate over files in that directory
for i, dirname in enumerate(os.listdir(directory)):
    if dirname == '.DS_Store':  # you can change to ignore any system file
        continue
    path = os.path.join(directory, dirname)
    print(path)
    result_directoy = f"./2021/{dirname}"

    # check if the directory exist-- create ir or not
    if not os.path.isdir(result_directoy):
        os.mkdir(result_directoy)

    for file in os.listdir(path):
        filePath = os.path.join(path, file)  # create the path again
        fileName = file.split('T')[0] + file.split('T')[1]  # extract the region and the date from the name
        fullTextPath = f"{result_directoy}/{fileName}.txt"  # Full path of the new text file
        # textVersion = f"{fullPath}.txt"
        if file == '.DS_Store':  # you can change to ignore any system file
            continue
        print(filePath)

        # Try 'rb'
        with lzma.open(filePath) as f, open(fullTextPath, "wb") as newfile:
            content = f.read()
            newfile.write(content)
            os.remove(filePath)

        df = pd.read_csv(f"{fullTextPath}", delimiter='\t',
                         names=['0', 'price', 'date', 'instance', 'OS', 'region_zone'])
        # inplace=True --the operation would work on the original object.
        # axis=1 -- dropping the column, not the row
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        # df['day'] = df['date'].dt.day_name()
        # df['date'] = df['date'].map(lambda x: str(x)[:-6])
        saveTo = f"{result_directoy}/{fileName}.csv"
        df.to_csv(saveTo, index=0)
        print(file)
