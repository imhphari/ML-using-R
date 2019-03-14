#Single linear Regression
rm(list=ls())

dataset=read.csv("Salary_Data.csv",header=7)
dataset
#Splitting dataset
library(caTools)
set.seed(123)
split=sample.split(dataset$Salary,SplitRatio=2/3)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)
#Feature Scaling
regressor=lm(formula=Salary~YearsExperience,data=training_set)
summary(regressor)

#Predicting test set results

y_pred=predict(regressor,newdata=test_set)

#Visualising training set results
library(ggplot2)
ggplot()+
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),colour='red')+
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata=training_set)),colour='blue')+
  ggtitle('Salary vs Experience(Training set)')+xlab('Years of Experience')+ylab('Salary')


             
             
