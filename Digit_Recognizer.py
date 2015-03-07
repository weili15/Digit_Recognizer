import scipy
import numpy as np
import csv as csv
from sklearn import neighbors
import matplotlib.pyplot as mpl

def read_data(csv_data, header=True, test=False): #header=True if there's a header, test = True if this is the test dataset
    data=[]
    labels=[]

    #read in data
    csv_reader = csv.reader(open(csv_data, 'r'))
    #Now we must store in a numpy array
    index=0
    for row in csv_reader:
        index+=1
        #ignore header
        if header & (index==1):
            continue
        #In our training dataset, we have labels, in our test dataset, we don't
        if not test:
            labels.append(int(row[0]))
            row=row[1:]
        data.append(np.array(row))
    return (data,labels)

def predictKNN(train, labels, test, k):
    print 'K nearest neighbor'
    KNN=neighbors.KNeighborsClassifier(k)
    KNN.fit(train,labels)
    predictions=KNN.predict(test)
    print 'done with KNN'
    return predictions


#Read in data and labels from training and test sets
train_data,train_labels=read_data("train.csv", header=True, test=False)
test_data,test_labels=read_data("test.csv", header=True, test=True)

#See effect of different k's on development data
#Divide raining set into 20% development data, 80% training
#We assume there are no order effects (drawing from first 20% is the same as drawing 20% randomly)
dev_index=int(.2*len(train_data))
dev_data=train_data[:dev_index]
dev_labels=train_labels[:dev_index]
new_train_data=train_data[dev_index:]
new_train_labels=train_labels[dev_index:]

k_accuracies=[]
#Try different k's
for k in range(1,6):
    this_prediction=predictKNN(new_train_data,new_train_labels,dev_data,k)
    #Compare with prediction with dev_labels
    accuracy=0
    for i in range(0,len(dev_labels)):
        if this_prediction[i]==dev_labels[i]:
            accuracy+=1
    k_accuracies.append(float(accuracy)/float(len(dev_labels)))

print k_accuracies
mpl.plot(range(1,6),k_accuracies)
mpl.show()
#Maybe try implementing PCA later?  The code is very slow.

#Perform regular classification
knnPredictions=predictKNN(train_data,train_labels,test_data)

#Output to submission csv
predictions_file = open("sub.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
ids=range(1,len(knnPredictions)+1)
open_file_object.writerows(zip(ids, knnPredictions))
predictions_file.close()
