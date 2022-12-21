import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def removeNullValues(dataset):
    # fill missing data (Categorical columns) --> 'Gender', 'Self_Employed', 'Married' ,'Credit_History, 'Property_Area'
    dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace= True)
    dataset['Married'].fillna(dataset['Married'].mode()[0], inplace= True)
    dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace= True)
    dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace= True)
    dataset['Property_Area'].fillna(dataset['Property_Area'].mode()[0], inplace= True)
    dataset['Education'].fillna(dataset['Education'].mode()[0], inplace= True)
    #replace '+3' field in Dependents column to '4' 
    dataset['Dependents'] = dataset['Dependents'].replace('3+', value = 4)
    dataset['Dependents'] = dataset['Dependents'].astype(np.float64)
    # fill of missing data (Numrical Columns) --> 'Dependents', 'LoanAmount', 'Loan_Amount_Term'
    dataset['Dependents'].fillna(int(dataset['Dependents'].median()), inplace= True)
    dataset['LoanAmount'].fillna(int(dataset['LoanAmount'].mean()), inplace= True)
    dataset['ApplicantIncome'].fillna(int(dataset['ApplicantIncome'].mean()), inplace= True)
    dataset['CoapplicantIncome'].fillna(int(dataset['CoapplicantIncome'].mean()), inplace= True)
    dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].median(), inplace= True)
    return dataset

#outlier --> Q1, Q2, IQR
#Get the outliers out any given column
def removeOutliersFromCol (columnData,dataSet):
    Q1 = columnData.describe()[4]
    Q3 = columnData.describe()[6]
    IQR = Q3 - Q1
    upperLimit = Q3 + (1.5*IQR)
    lowerLimit = Q1 - (1.5*IQR)
    filteredData = dataSet[(columnData < upperLimit) & (columnData > lowerLimit)]
    return filteredData


testData = pd.read_csv("loan-test.csv")
nullFreeDataSet = removeNullValues(testData)
newDataTest = removeOutliersFromCol(nullFreeDataSet["Dependents"],nullFreeDataSet)
newDataTest = removeOutliersFromCol(nullFreeDataSet["LoanAmount"],nullFreeDataSet)
newDataTest = removeOutliersFromCol(nullFreeDataSet["ApplicantIncome"],nullFreeDataSet)
newDataTest = removeOutliersFromCol(nullFreeDataSet["CoapplicantIncome"],nullFreeDataSet)

trainData = pd.read_csv("loan-train.csv")
nullFreeDataSet = removeNullValues(trainData)
newDataTrain = removeOutliersFromCol(nullFreeDataSet["Dependents"],nullFreeDataSet)
newDataTrain = removeOutliersFromCol(nullFreeDataSet["LoanAmount"],nullFreeDataSet)
newDataTrain = removeOutliersFromCol(nullFreeDataSet["ApplicantIncome"],nullFreeDataSet)
newDataTrain = removeOutliersFromCol(nullFreeDataSet["CoapplicantIncome"],nullFreeDataSet)

#The Final Data After Cleaning
CleandDataTest = newDataTest
CleandDataTrain = newDataTrain
#Data replaced to Categorical
CleandDataTrain.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0},'Loan_Status':{'Y':1,'N':0}},inplace=True)

CleandDataTest.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


#Decision Tree Model
X=CleandDataTrain.iloc[:,1:-1].values
y=CleandDataTrain.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_test = dt.predict(CleandDataTest.drop('Loan_ID',axis=1))
y_pred_train = dt.predict(X_test)


# Compute accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred_train)*100
print("Decision Tree model accuracy is : {:.2f}".format(acc))


#Visulization Decision Tree scatter plot
m=CleandDataTrain[CleandDataTrain.Gender == 1]
f=CleandDataTrain[CleandDataTrain.Gender == 0]


plt.title("Applicant Income And Loan Amount")
plt.xlabel("Loan Ammount")
plt.ylabel("Applicant Income")
plt.scatter(m.LoanAmount, m.ApplicantIncome, color = "red", label = "Male", alpha = 0.3)
plt.scatter(f.LoanAmount, f.ApplicantIncome, color = "purple", label = "Female", alpha = 0.3)
#plt.ylim(0, 20000)
plt.legend()
plt.show()


#Naive Bayes classification model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred_NB = gnb.predict(X_test)
y_pred_test_NB = gnb.predict(CleandDataTest.drop('Loan_ID',axis=1))

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
gnb_accuracy=(metrics.accuracy_score(y_test, y_pred_NB)*100)
print("Gaussian Naive Bayes model accuracy is : {:.2f}".format(gnb_accuracy))

#histogram plot
n= plt.hist(m.ApplicantIncome,100)

plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Histogram of Male')
#plt.xlim(0, 60000)
plt.grid(True)
plt.show()



#logistic regression model
from sklearn import linear_model

logr = linear_model.LogisticRegression()

logr.fit(X_train, y_train)
# making predictions on the testing set
y_pred_logr = logr.predict(X_test)
y_pred_test_logr = logr.predict(CleandDataTest.drop('Loan_ID',axis=1))

logr_accuracy=(metrics.accuracy_score(y_test, y_pred_logr)*100)
print("Logistic Regression model accuracy is : {:.2f}".format(logr_accuracy))


#bar chart visualization
plt.bar(CleandDataTrain.Married,CleandDataTrain.ApplicantIncome)
plt.xticks([0.0,1.0], ['No', 'Yes'])
plt.xlabel('Marriage')
plt.ylabel('Applicant Income')
plt.title('bar chart of Applicant Income and Marriage')
plt.show()


#Model's accurancy plot
models_data={'Decision Tree':acc,
             'Logistic Regression':logr_accuracy,
             'Naive Bayes':gnb_accuracy}

models = list(models_data.keys())
values = list(models_data.values())

plt.plot(models,values)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title("bar chart of Model's Accuracy")
plt.show()
