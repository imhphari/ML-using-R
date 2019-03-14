import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy
from sklearn import metrics

import csv
CSVFILE='/home/mirunalini/Desktop/talk/multiclass/leaves/results-multiclass.csv'
test_df=pd.read_csv(CSVFILE)


#
#with open('results.csv',newline='') as f:
#    r = csv.reader(f)
#    data = [line for line in r]
#with open('results.csv','w',newline='') as f:
#    w = csv.writer(f)
#    w.writerow(['actual','predicted'])
#    w.writerows(data)



actualValue=test_df['actual']
predictedValue=test_df['predicted']

actualValue=actualValue.values
predictedValue=predictedValue.values

target_names = ['class 0', 'class 1','class 2', 'class 3']
#print(classification_report(actualValue,predictedValue, target_names=target_names))
#              
#tn, fp, fn, tp = confusion_matrix(actualValue,predictedValue).ravel()
#print("True negative",tn, "\nFalse Positive",fp,"\nFalse Negative", fn,"\nTrue Positive", tp)

#Confusion matrix
cmt=confusion_matrix(actualValue,predictedValue)
print (cmt)

#Precision
precision=metrics.precision_score(actualValue,predictedValue,average='macro')
print('Precision',precision)

#Recall
recall=metrics.recall_score(actualValue,predictedValue,average='macro')
print('Recall',recall)

#F1-score
F1score=metrics.f1_score(actualValue,predictedValue,average='macro')
print('F1score',F1score)
