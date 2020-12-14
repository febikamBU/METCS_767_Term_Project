# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
sns.set()
from time import time

# Method to read in the csv file
def reader_csv(path):
    # Reading the bank note data from the from dataset directory.
    df_bcds = pd.read_csv(path)

    # Making a copy of the original my bank note data
    df = df_bcds.copy()
    
    # Displaying only first fews lines of the bank note dataset
    df.head()
    return df, df_bcds

# Function replace ANY feature(s) that has underscore in their names.  
def replace_colnames(df_cols):
    lst = []
    for i, c in enumerate(list(df_cols)):
        lst.append(c)
    feature = [name.replace(' ', '_') for name in lst]
    return feature

# Splitting death events (0 & 1) into training/testing set for both surviving patients.
def train_test_data_split(df):
    
    # Subset features and class label     
    X = np.array(df.iloc[:, 0:-2].values)
    y = np.array(df.iloc[:, -1].values)
    
    # Split/stratifying dataset X into training Xtrain and Xtesting parts (80/20 split).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state= 42)
      
    return X_train, X_test, y_train, y_test

# Feature scaler method
def scaler_features(X_train, X_test):
    # Feature scaling
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.fit_transform(X_test)
    return X_scaled_train, X_scaled_test

# Function to print features and their corresponding correlation side-by-side     
def correlation_table(df):
    df_sort = df.corr().abs()
    df_sort = df_sort.nlargest(30,'class_label')['class_label']
    sort_1 = df_sort.sort_values(kind="quicksort")
    return sort_1

# Veiwing all
def data_corr(df):
    corr = df.corr().abs()
    corr.nlargest(30,'class_label')['class_label']
    return corr

# Evaluating & printing the model
def model_evaluation(y_test, y_pred):

    # Computing the following with confusion matrix
    cf_1 = confusion_matrix(y_test, y_pred)

    # Computing my simple model evaluation metrics - that is, TP, TN, FP etc.,
    fetal_TP = cf_1[1][1]  #TP - true positives
    fetal_FP = cf_1[0][1]  #FP - false positives
    fetal_TN = cf_1[0][0]  #TN - true negativess
    fetal_FN = cf_1[1][0]  #FN - false negatives
    fetal_TPR = round((fetal_TP/(fetal_TP + fetal_FN)) * 100, 2) #TPR = TP/(TP + FN)
    fetal_TNR = round((fetal_TN/(fetal_TN + fetal_FP)) * 100, 2) #TNR = TN/(TN + FP)
    fetal_FPR = round((fetal_FP/(fetal_FP + fetal_TN)) * 100, 2) #FPR = FP/(FP + TN)
    fetal_ACC = round(((fetal_TP + fetal_TN)/(fetal_TP + fetal_TN + fetal_FP + fetal_FN)) * 100, 2)
   
    # Displaying the Performance metrics of the model
    print(f'TP - true positives: {fetal_TP}\nFP - false positives: {fetal_FP}\nTN - true negativess: '
          f'{fetal_TN}\nFN - false negatives: {fetal_FN}\nTPR-true positive rate: {fetal_TPR}%\n'
          f'TNR - true negative rate: {fetal_TNR}%\nFPR - False Positive Rate: {fetal_FPR}%\n')
    print(classification_report(y_test, y_pred))
    print(end='\n')
    
# Method for Kneighborsclassifier_model 
def knn_classifier_model(X_train, X_test, y_train, y_test):
    t0 = time()

    # Create classifier.
    clf = KNeighborsClassifier(n_neighbors = 10, metric = 'euclidean', p = 2)

    # Fit the classifier on the training features and labels.
    t0 = time()
    clf.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Make predictions.
    t1 = time()
    y_pred = clf.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")

    # Evaluate the model.
    accuracy = accuracy_score(y_test, y_pred)

    # Print the reports.
    print("Accuracy: %.2f%%" % (accuracy *100.0), '\n') 
    model_evaluation(y_test, y_pred)
    print("\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return y_pred

# Method for logistic_regression_model
def logistic_regression_model(X_train, X_test, y_train, y_test):
    t0 = time()
    
    # Create classifier
    model = LogisticRegression()
    
    # Fit the classifier on the training features and labels.
    t0 = time()
    model.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Predicting using X_test_norm
    t1 = time()
    y_pred = model.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")

    # Computing for the accuracy, precision, & recall
    result = model.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (result * 100.0), '\n')
    
    # Diplay performance metrics     
    model_evaluation(y_test, y_pred)
    print("\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return y_pred
    
# Method for DecisionTreeClassifier_model    
def decisionTreeClassifier_model(X_train, X_test, y_train, y_test):
    t0 = time()
    
    # Create classifier
    model = DecisionTreeClassifier(criterion='entropy')
    
    # Fit the classifier on the training features and labels.
    t0 = time()
    model.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Predicting using X_test_norm
    t1 = time()
    y_pred = model.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")

    # Computing for the accuracy, precision, & recall
    result = model.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (result * 100.0), '\n')
    
    # Diplay performance metrics     
    model_evaluation(y_test, y_pred)
    print("\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return y_pred
    
# Method for gaussianNB_model    
def gaussianNB_model(X_train, X_test, y_train, y_test):
    t0 = time()
    
    # Create classifier
    model = GaussianNB()
    
    # Fit the classifier on the training features and labels.
    t0 = time()
    model.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Predicting using X_test_norm
    t1 = time()
    y_pred = model.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")

    # Computing for the accuracy, precision, & recall
    result = model.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (result * 100.0), '\n')
    
    # Diplay performance metrics     
    model_evaluation(y_test, y_pred)
    print("\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return y_pred

# Method for gaussianNB_model    
def linearDiscriminantAnalysis_model(X_train, X_test, y_train, y_test):
    t0 = time()
    
    # Create classifier
    model = LinearDiscriminantAnalysis()
    
    # Fit the classifier on the training features and labels.
    t0 = time()
    model.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Predicting using X_test_norm
    t1 = time()
    y_pred = model.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")

    # Computing for the accuracy, precision, & recall
    result = model.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (result * 100.0), '\n')
    
    # Diplay performance metrics     
    model_evaluation(y_test, y_pred)
    print("\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return y_pred


# Method for gaussianNB_model    
def randomForestClassifier_model(df, X_train, X_test, y_train, y_test):
    t0 = time()
    
    # Create classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Fit the classifier on the training features and labels.
    t0 = time()
    model.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Predicting using X_test_norm
    t1 = time()
    y_pred = model.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")

    # Computing for the accuracy, precision, & recall
    result = model.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (result * 100.0), '\n')
    
    # Diplay performance metrics     
    model_evaluation(y_test, y_pred)
    print('\n')
    
    # Create dataframe by zipping RFC feature importances and column names
    rfc_features = pd.DataFrame(zip(model.feature_importances_, df.columns[:-1]), columns = ['Importance', 'Features'])

    # Sort in descending order for easy organization and visualization
    rfc_features = rfc_features.sort_values(['Importance'], ascending=False)
          
    print("\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return y_pred, rfc_features

def baggingClassifier_Model(df):
    
    # Calling & passing train_test_data_split method    
    X_train, X_test, y_train, y_test = train_test_data_split(df)
    
    # Calling & passing scaler_features method      
    X_train_norm, X_test_norm = scaler_features(X_train, X_test)
    
    # Declaring multiple models 
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    dtc = DecisionTreeClassifier()
    gnb = GaussianNB()

    # Building the Bagging models
    models = [lr, knn, gnb, dtc]
    for model in models:
        # Creating Bagging Classifier     
        bag = BaggingClassifier(base_estimator = model, n_estimators = 10, bootstrap = True)

        # Fit the classifier on the training features and labels.
        bag = bag.fit(X_train_norm, y_train)

        # Predicting using X_test_norm
        y_pred_bag = bag.predict(X_test_norm)
        
        # Computing for the accuracy, precision, & recall
        result = bag.score(X_test_norm, y_test)
        print("Accuracy: %.2f%%" % (result * 100.0), [model])
        
    # define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

        
# IMPORTANT: first argument is true values, second argument is predicted probabilities

# we pass y_test and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

# plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.rcParams['font.size'] = 12
# plt.title('ROC curve for diabetes classifier')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.grid(True)
        
        
        
        
        