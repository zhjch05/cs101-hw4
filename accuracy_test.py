test = {}
total = 0
correct = 0
#read in annotation (test)
with open("annotation.txt","r") as annotations:
    for line in annotations:
        image, label = line.split()
        test[image] = label

#read in predictions (prediction)
with open("prediction.txt","r") as predictions:
    for line in predictions:
        image, label = line.split()
        total += 1
        if label == test[image]:
            correct += 1

accuracy = correct/total

print("%.4f" % accuracy)