import os
import csv
import json
import sys
import statistics as st

# https://realpython.com/python-rounding/#truncation
def truncate(n, decimals=4):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def getStatistics(data, label):
    values: list(float) = data[label]

    return {
        "mean": truncate(st.mean(values)),
        "std": truncate(st.stdev(values)),
        "min": truncate(min(values)),
        "max": truncate(max(values)),
        "median": truncate(st.median(values))
    }


def getDataFrom(filename, execution, runs):


    with open(filename, "r") as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            # print(row)

            runs["accuracy"].append(float(row["Accuracy"]))
            runs["precision"].append(float(row["Precision"]))
            runs["recall"].append(float(row["Recall"]))
            runs["f1_score"].append(float(row["F1Score"]))
            runs["specificity"].append(float(row["Specificity"]))
            runs["lrp"].append(float(row["LRP"]))
            runs["lrm"].append(float(row["LRM"]))

        #


def main(path):
    # path = "/home/mateus/Documents/pancada/Bladder_Hsapiens_GSE38264/"

    execution: dict = {

    }

    settings = path + "statistics_1.csv"

    gse: str = None

#     with open(settings, "r") as f:
#         csv_reader = csv.DictReader(f)
#         for row in csv_reader:

#             label = row["gse"]

#             execution["gse"] = row["gse"]
#             execution["tissue"] = row["tissue"]
#             execution["samples"] = row["samples"]
#             execution["genes"] = row["genes"]
#             execution["classes"] = row["classes"]
#             execution["downloads"]["csv"] = f"/data/pancada/dataset/{label}/{label}.csv.zip"
#             execution["downloads"]["pca"] = f"/data/pancada/dataset/{label}/{label}-pca.png"
#             execution["downloads"]["tsne"] = f"/data/pancada/dataset/{label}/{label}-tsne.png"
#             execution["downloads"]["results"] = f"/data/pancada/dataset/{label}/{label}.pdf"

#         gse = execution["gse"]


    # try:
    #     # https://linuxhint.com/extract-substring-regex-python/
    #     gse = re.search('GSE(.+?).csv', filename).group(1)
    #     print(gse)
    # except AttributeError:
    #     pass


    data = []
    
    runs = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "specificity": [],
        "lrp": [],
        "lrm": []
    }

    for filename in os.listdir(path):
        #print(path)

        if filename.endswith(".csv"):
            #print(filename)
            #print(execution)
            getDataFrom(path+filename, execution, runs)
            #print(execution)
            
    print(len(runs["accuracy"]))
    
    execution["scores"] = {
        "accuracy": getStatistics(runs, "accuracy"),
        "precision": getStatistics(runs, "precision"),
        "recall": getStatistics(runs, "recall"),
        "f1_score": getStatistics(runs, "f1_score"),
        "specificity": getStatistics(runs, "specificity"),
        "lrp": getStatistics(runs, "lrp"),
        "lrm": getStatistics(runs, "lrm")
    }
    print(execution)

    print(path)
    
    
    with open(path+"a.json", "w", encoding='utf-8') as file:
        json.dump(execution, file, ensure_ascii=False, indent=4)



if __name__ == "__main__":

    try:
        # just check if args are ok
        with open('args.txt', 'w') as f:
            f.write(str(sys.argv))
    except Exception as e:
        raise SystemExit(f"Error: {e}")

    if len(sys.argv) < 2:
        s = "Usage: ./main.py <filename>"

        print(f"\n{s}\n")

        sys.exit(1)

    path: str = sys.argv[1]

    print(path)

    main(path)
