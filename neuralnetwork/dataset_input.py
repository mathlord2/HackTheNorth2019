import csv

emojis = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def getDataSet(fileName):
    output = []
    with open(fileName, mode = "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for row in csvReader:
            if row["Usage"] == "Training":
                output.append([int(row["emotion"]), row["pixels"]])
    return output

print(getDataSet("fer2013.csv")[0])