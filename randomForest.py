import pandas as pd
import numpy as np
import time
import os, psutil
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
def printmem():
    process = psutil.Process(os.getpid())
    print("   memory:",round(process.memory_info().rss/(10**9),3),'Gbytes')  # in bytes 
    
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from tabulate import tabulate

def calc_performance_multi(y_vals_true, y_vals_pred,labels):
#
# Get the numbers for the confusion matrix
# To get output: cf_matrix[true_label,pred_label]
    cf_matrix = confusion_matrix(y_vals_true, y_vals_pred, labels=labels)
#
# This is a graphic
    cf_disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,display_labels=labels)
#
# Make the header row
    header = [""]
    for column_name in labels:
        header.append('Pred:' + str(column_name))
    table = [header]
#
# Now make the rows with the matrix
    for row_name in labels:
        row = ['True:'+str(row_name)]
        for column_name in labels:
            row_index = labels.index(row_name)
            column_index = labels.index(column_name)
            row.append(cf_matrix[row_index,column_index])
        table.append(row)
    # table = [
    #     [ "", "Predicted Class 1", "Precicted Class 0"],
    #     [ "True Class 1", TP, FN ],
    #     [ "True Class 0", FP, TN ]
    # ]
    print_table_type='fancy_grid'
    print_table = tabulate(table, headers='firstrow', tablefmt=print_table_type)
#    print(print_table)
#
# Get the recall, precision, ands F1 for each individual label
# - return both the "string report" (which you can print)
# - and the "dictionary report" (which you can use for averages and so on)
    report = classification_report(y_vals_true,y_vals_pred)
    report_dict = classification_report(y_vals_true,y_vals_pred,output_dict=True)
#
    results = {"confusionMatrix":cf_matrix,
                    'confusion_matrix_display':cf_disp,
                    'confusion_matrix_print_table':print_table,      
                    "report":report,"report_dict":report_dict}
    return results


def run_fitter_multi(estimator,X_train,y_train,X_test,y_test,labels):
#
# Now fit to our training set
    estimator.fit(X_train,y_train)
#
# Now predict the classes and get the score for our traing set
    y_train_pred = estimator.predict(X_train)
    y_train_score = estimator.predict_proba(X_train)   # NOTE: some estimators have a predict_prob method instead od descision_function
#
# Now predict the classes and get the score for our test set
    y_test_pred = estimator.predict(X_test)
    y_test_score = estimator.predict_proba(X_test)

#
# Now get the performaance
    results_test = calc_performance_multi(y_test,y_test_pred,labels)
    results_train = calc_performance_multi(y_train,y_train_pred,labels)
#
    return results_train,results_test


#df = pd.read_csv('train_expand_S_P.zip')
df = pd.read_csv('train_Normal.zip')
#df = df.head(100000)
#dicMW = {'M': 0, 'F':1}
#df['SEX'] = df['SEX'].map(dicMW)
#colsNo = ['DIFFERENTIAL_DIAGNOSIS', 'PATHOLOGY', 'EVIDENCES', 'INITIAL_EVIDENCE']
#df = df.loc[:, [col for col in df.columns if col not in colsNo]]

X = df.iloc[:, 0:-2].to_numpy()
y = df['severity'].values
#y = df['pathoClass'].values
labels = [1,2,3,4,5]
#labels = list(range(0,49))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

kfolds = 5
skf = StratifiedKFold(n_splits=kfolds)

t0 = time.time()
estimator = RandomForestClassifier(n_estimators=50, max_depth=7,random_state=42)

avg_precision_train = 0.0
avg_recall_train = 0.0
avg_f1_train = 0.0
avg_precision_test = 0.0
avg_recall_test = 0.0
avg_f1_test = 0.0
numSplits = 0.0
#
# Now loop

for train_index, test_index in skf.split(X, y):
    #print("Training")
    numSplits += 1
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]  
#
# Now fit to our training set
    results_train,results_test = run_fitter_multi(estimator,X_train,y_train,X_test,y_test,labels)

    avg_precision_train += results_train['report_dict']['weighted avg']['precision']
    avg_recall_train += results_train['report_dict']['weighted avg']['recall']
    avg_f1_train += results_train['report_dict']['weighted avg']['f1-score']
#
    avg_precision_test += results_test['report_dict']['weighted avg']['precision']
    avg_recall_test += results_test['report_dict']['weighted avg']['recall']
    avg_f1_test += results_test['report_dict']['weighted avg']['f1-score']
#
avg_precision_train /= numSplits
avg_recall_train /= numSplits
avg_f1_train /= numSplits
avg_precision_test /= numSplits
avg_recall_test /= numSplits
avg_f1_test /= numSplits
# 
# Now print
print("Precision train/test ",round(avg_precision_train,3),round(avg_precision_test,3))
print("Recall train/test    ",round(avg_recall_train,3),round(avg_recall_test,3))
print("F1 Score train/test       ",round(avg_f1_train,3),round(avg_f1_test,3))
print("All done! Time:", time.time()-t0)
printmem()

estimator = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)

results_train, results_test = run_fitter_multi(estimator,X_train,y_train,X_test,y_test,labels)

#For severity classification
results_train['confusion_matrix_display'].plot()
plt.savefig('TrainMatrixRF.png')

results_test['confusion_matrix_display'].plot()
plt.savefig('TestMatrixRF.png')

#==================================================================

#For pathology classification
#confTrain = pd.DataFrame(results_train['confusionMatrix'])
#confTest = pd.DataFrame(results_test['confusionMatrix'])

#confTrain.to_csv('confMatTrain1.csv', index=False)
#confTest.to_csv('confMatTest1.csv', index=False)
#==================================================================

filename = 'runForestRunRF.sav'
pickle.dump(estimator, open(filename, 'wb'))
printmem()