import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# step1: brief review dataset with bar plot
df = pd.read_csv("creditcard.csv",low_memory= False)
count_class = pd.value_counts(df['Class'],sort = True)    #.sort_index()
count_class.plot(kind = 'bar')
plt.title("Fraud Distrubition")
plt.xlabel("Fraud Or Not")
plt.ylabel("Frequency")
plt.show()
# From plot we can see extremely unbalanced distribution in y( 0 and 1). !!!There is a need to subsample or oversample in order to gain more recall and accurancy

#step 2: preprocess
#!!! we should make values in each feature in same range. which mean narrow down Amount in (0,1)
from sklearn.preprocessing import StandardScaler
df["normAmount"] = StandardScaler().fit_transform(pd.DataFrame(df["Amount"]))
df = df.drop(columns = ["Time", "Amount"])


#method 1 UnderSAMPLE
X = df[df.columns[df.columns != 'Class']]
Y = df["Class"]

fraud_number = sum(df["Class"] == 1)
fraud_index = np.array(df[df["Class"]==1].index)

normal_index = np.array(df[df.Class==0].index)

# randomly select from normal data
random_normal_index = np.random.choice(normal_index,fraud_number,replace=False)  #select fraud number pieces from normal_index

#merge processed data into one which is the aimed dataset
total_index = np.concatenate([random_normal_index,fraud_index])
subsample = df.loc[total_index]

X_subsample = subsample[subsample.columns[subsample.columns != "Class"]]
Y_subsample = subsample["Class"]


from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.33, random_state=42)
X_subsample_train,X_subsample_test, Y_subsample_train,Y_subsample_test = train_test_split(X_subsample,Y_subsample,test_size= 0.33, random_state= 42)

#Cross Validtion
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import confusion_matrix,recall_score,classification_report
from sklearn.linear_model import LogisticRegression

def KfoldScore(x_train, y_train):
    fold = KFold(n_splits=5, shuffle=False)
    fold.get_n_splits(y_train)

    c_param = [0.01, 0.1, 1, 10,100]
    results_table = pd.DataFrame(index=range(0,len(c_param)), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param

    j = 0
    # calculate avg recall score for each c parameter
    for c in c_param:
        print("c parameter is:" , c)

        recall_accs = []
        i=1
        x_train.reset_index(drop=True,inplace=True)
        y_train.reset_index(drop=True,inplace=True)

        for train_index, test_index in fold.split(y_train):  #interation is x_index, indices is y_index

            lr = LogisticRegression(C =c,penalty="l1")  # add penalty item. Using absolute variance
            lr.fit(x_train.loc[train_index],y_train.loc[train_index])

            y_predicted = lr.predict(x_train.loc[train_index])
            recall_acc = recall_score(y_train.loc[train_index],y_predicted)
            recall_accs.append(recall_acc)
            print('Iteration ',i, ': recall score = ', recall_acc)
            i =+ 1

        results_table[ 'Mean recall score'] [j]= np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.C_parameter[np.argmax(np.array(results_table['Mean recall score']))]
    return best_c

best_c = KfoldScore(X_subsample_train,Y_subsample_train)

#plot confusion matricx
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",verticalalignment='center', color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import itertools
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_subsample_train,Y_subsample_train.values)
y_pred_undersample = lr.predict(X_subsample_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_subsample_test,y_pred_undersample)
np.set_printoptions(precision=2)  #round number

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names , title='Confusion matrix')
plt.show()




#Process the whole dataset
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train,Y_train.values)
y_pred_undersample = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names , title='Confusion matrix')
plt.show()

#From this plot, we can see, if we don't set threhold properly, a high miskill rate will happen
#So next step is to find a sutiable threhold
lr = LogisticRegression(C=0.01, penalty='l1')
lr.fit(X_subsample_train, Y_subsample_train)
y_pred_undersample_proba = lr.predict_proba(X_subsample_test)  #lr 本来就是 probability

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10, 10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i  #????

    plt.subplot(3, 3, j)
    j += 1

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_subsample_test, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names , title='Threshold >= %s' % i)

plt.show()

#Method 2: Oversample

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

credit_cards=pd.read_csv('creditcard.csv')

columns=credit_cards.columns
# The labels are in the last column ('Class'). Simply remove it to obtain features columns
features_columns=columns.delete(len(columns)-1)

features=credit_cards[features_columns]
labels=credit_cards['Class']
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.2, random_state=0)

oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)

best_c = KfoldScore((os_features,os_labels))

lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(os_features,os_labels.values.ravel())
y_pred = lr.predict(features_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()
