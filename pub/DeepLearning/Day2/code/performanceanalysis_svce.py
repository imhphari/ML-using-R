import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy
from sklearn import metrics

import csv
CSVFILE='results.csv'
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

target_names = ['class 0', 'class 1']
print(classification_report(actualValue,predictedValue, target_names=target_names))
              
tn, fp, fn, tp = confusion_matrix(actualValue,predictedValue).ravel()
print("True negative",tn, "\nFalse Positive",fp,"\nFalse Negative", fn,"\nTrue Positive", tp)