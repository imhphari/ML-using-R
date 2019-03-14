rm(list=ls())
dataset=read.csv("Social_Network_Ads.csv")
dataset
#Import
dataset=dataset[3:5]

#Encoding the target feature as factor

dataset$Purchased=factor(dataset$Purchased,levels=c(0,1))

library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio=0.75)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)
training_set

#Feature Scaling
training_set[-3]=scale(training_set[-3])
test_set[-3]=scale(test_set[-3])


#fitting KNN to the training set and predicting the test set results

library(class)
y_pred=knn(train=training_set[,-3],test=test_set[,-3],cl=training_set[,3],k=5,prob=TRUE)

#Checking using confusion matrix
cm=table(test_set[,3],y_pred)
(cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])