rm(list=ls())
dataset=read.csv("Social_Network_Ads.csv")
dataset
#Import
dataset=dataset[3:5]
view(dataset)
str(dataset)


library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio=0.75)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)
training_set

#scaling
training_set[-3]=scale(training_set[-3])
test_set[-3]=scale(test_set[-3])

#glm-generated linear model
classifier=glm(formula=Purchased~., family=binomial,data=training_set)
summary(classifier)

prob_pred=predict(classifier,type="response",newdata=test_set[-3])
y_pred=ifelse(prob_pred>0.5,1,0)
y_pred

#Making the confusion matrix
cm=table(test_set[,3],y_pred>0.5)
cm

(cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])