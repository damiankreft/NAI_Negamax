from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import category_encoders as ce
import pandas as pd

dataset = load_wine()
print ('############ WINE ###############')
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)

# print("aaaaaaa")
# print(X_train[132][12])


###
### Decision Tree Classifier - Wine
###
dtc = DTC(random_state=42, max_depth=4)
dtc.fit(X_train,y_train)
dtc_predictions = dtc.predict(X_test)
# print("SKLearn Wine predictions:\n",skl_preds)
# print("------------------------------------------------------")
dtc_correct_count = len(set(dtc_predictions) & set(y_test))
dtc_accuracy = dtc_correct_count / len(y_test)
print("DTC accuracy (wine):", dtc_accuracy)
# print("------------------------------------------------------")

###
### Support Vector Machine - Wine
###
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
# print(df.shape)
cols = ['buying_price', 'maintenance_cost', 'door_num', 'seats', 'lug_boot', 'safety', 'class']
df.columns = cols

print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
print ('######### CAR EVALUATION ##########')
###
### Decision Tree Classifier - Car Evaluation
###
# print(df.head())
# print(df.info())
# print(df['class'].value_counts())
X = df.drop(['class'], axis=1)
# y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, df['class'], test_size = 0.33, random_state = 42)
# print(X_train.shape, X_test.shape)
# print(X_train.dtypes)
# print(X_train.head())
encoder = ce.OrdinalEncoder(cols=cols[0:6])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
# print(X_train.head())
# instantiate the DecisionTreeClassifier model with criterion gini index
clf = DTC(random_state=0, max_depth=3)
# fit the model
clf.fit(X_train, y_train)
y_pred_gini = clf.predict(X_test)
print('DTC accuracy (car evaluation): {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))


###
### Support Vector Machine - Car Evaluation
###
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