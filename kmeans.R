rm(list=ls())
dataset=read.csv("Mall_Customers.csv")
dataset
dataset=dataset[4:5]
dataset

set.seed(6)
wcss=vector()
for( i in 1:10) wcss[i]=sum(kmeans(dataset,i)$withinss)

plot(1:10,wcss,type='b',# for both lines and points
    main=paste('The Elbow Method'),
    xlab='No of Clusters',
    ylab='WCSS')

kmeans=kmeans(x=dataset,centers=5)
y_kmeans=kmeans$cluster
y_kmeans

library(cluster)
clusplot(dataset,
         y_kmeans,
         lines=0,#so that no distance lines will appear in our plot
         shade=TRUE,
         color=TRUE,
         labels=2,
         plotchar=FALSE,
         span=TRUE,
         main=paste('Clusters of customers'),
         xlab='Annual income',
         ylab='Spending Score')
         