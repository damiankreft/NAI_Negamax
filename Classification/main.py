""" Authors:Damian Kreft, Sebastian Kreft
    Required environment: Python3, scikit-learn, category_encoders, pandas"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import category_encoders as ce
import pandas as pd

dataset = load_wine()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)


def wine_dtc():
    """Decision Tree Classifier - Wine"""
    dtc = DTC(random_state=42, max_depth=4)
    dtc.fit(X_train,y_train)
    dtc_predictions = dtc.predict(X_test)
    dtc_correct_count = len(set(dtc_predictions) & set(y_test))
    dtc_accuracy = dtc_correct_count / len(y_test)
    print("DTC accuracy (wine):", dtc_accuracy)

def wine_svm():
    """Support Vector Machine - Wine"""
    global X_train
    global X_test
    global y_train
    global y_test
    svm_kernel = 'rbf'
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    svm_model = SVC(C=1, kernel=svm_kernel)
    svm_model.fit(X_train, y_train)

    svm_y_pred = svm_model.predict(X_test)
    print('SVM accuracy (wine):', accuracy_score(y_test, svm_y_pred), "(%s)" % svm_kernel)
    print('Confusion Matrix:\n', confusion_matrix(y_test, svm_y_pred))
    print('Classification Report:\n', classification_report(y_test, svm_y_pred))


######################################################################################################
##################### Car ######################################### Evaluation #######################
######################################################################################################
path = r"data\car_evaluation.csv"
df = pd.read_csv(path, header=None)
cols = ['buying_price', 'maintenance_cost', 'door_num', 'seats', 'lug_boot', 'safety', 'class']
df.columns = cols



def cars_dtc():
    """Decision Tree Classifier - Car Evaluation"""
    X = df.drop(['class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, df['class'], test_size = 0.33, random_state = 42)
    encoder = ce.OrdinalEncoder(cols=cols[0:6])
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    clf = DTC(random_state=0, max_depth=3)
    clf.fit(X_train, y_train)
    y_pred_gini = clf.predict(X_test)
    print('DTC accuracy (car evaluation): {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

def cars_svm():
    """Support Vector Machine - Car Evaluation"""
    global X_train
    global X_test
    global y_train
    global y_test
    svm_kernel = 'rbf'
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    svm_model = SVC(C=1, kernel=svm_kernel)
    svm_model.fit(X_train, y_train)

    svm_y_pred = svm_model.predict(X_test)
    print('SVM accuracy (car evaluation):', accuracy_score(y_test, svm_y_pred), "(%s)" % svm_kernel)
    print('SVM confusion matrix:\n', confusion_matrix(y_test, svm_y_pred))
    print('SVM classification report:\n', classification_report(y_test, svm_y_pred))

print ('############ WINE ###############')
wine_dtc()
wine_svm()
print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
print ('######### CAR EVALUATION ##########')
cars_dtc()
cars_svm()