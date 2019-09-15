#A functional file to process csv pixel information
import csv

def rgbhex(num):
    hex = "#%02x%02x%02x" % (num, num, num)
    return hex

def getDataSet(fileName):
    output = []
    with open(fileName, mode = "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for row in csvReader:
            if row["Usage"] == "Training":
                output.append([int(row["emotion"]), list(map(int, row["pixels"].split()))])
    return output

def process(screen, dots):
    count = 0
    for i in range(48):
        for j in range(48):
            screen.create_oval(j,i,j,i,fill=rgbhex(dots[count]), outline=rgbhex(dots[count]), width=1)
            count += 1

    screen.update()

    while True:
        pass