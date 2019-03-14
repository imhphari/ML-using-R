#Multiple Linear Regression
rm(list=ls())

dataset=read.csv("50_Startups.csv")
dataset
dataset$State=factor(dataset$State,levels=c('New York','California','Florida'),labels=c(1,2,3))
str(dataset)

library(caTools)
set.seed(123)
split=sample.split(dataset$Profit,SplitRatio=0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Fiting multiple linear regression to the training set
regressor=lm(formula=Profit~.,data = training_set)
summary(regressor)

regressor1=lm(formula=Profit~R.D.Spend,data = training_set)
summary(regressor1)

regressor2=lm(formula=Profit~R.D.Spend+Administration,data = training_set)
summary(regressor2)
y_pred=predict(regressor2,newdata = test_set)
y_pred