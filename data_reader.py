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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, f1_score, roc_auc_score, roc_curve
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
    fetal_TP = cf_1[0][0]  #TP - true positives
    fetal_FP = cf_1[0][1]  #FN - false negatives
    fetal_FN = cf_1[1][0]  #FP - false positives
    fetal_TN = cf_1[1][1]  #TN - true negativess
    
    P = fetal_TP + fetal_FP
    N = fetal_FN + fetal_TN
    P_N = P + N
    
    fetal_TPR = round((fetal_TP/(P)) * 100, 2) #TPR = TP/(TP + FN)
    fetal_TNR = round((fetal_TN/(N)) * 100, 2) #TNR = TN/(TN + FP)
    fetal_FPR = round((fetal_FP/(N)) * 100, 2) #FPR = FP/(FP + TN)
    fetal_ACC = round(((fetal_TP + fetal_TN)/(P_N)) * 100, 2)
    fetal_PPV = round((fetal_TP/( fetal_TP + fetal_FP)) * 100, 2) #PPV = TP/(TP + FP)
    fetal_NPV = round((fetal_TN/(fetal_TN + fetal_FN)) * 100, 2) #NPV = TN/(FN + TN)
    fetal_f1 = round((2 * fetal_TP)/((2 * fetal_TP) + fetal_FP + fetal_FN)* 100, 2)
    roc_score = roc_auc_score(y_test, y_pred)
   
    # Displaying the Performance metrics of the model
    print(f'TP - True positives: {fetal_TP}\n'
          f'FP - False positives: {fetal_FP}\n'
          f'TN - True negativess: {fetal_TN}\n'
          f'FN - False negatives: {fetal_FN}\n\n'
          f'TPR - True positive rate: {fetal_TPR}%\n'
          f'TNR - True negative rate: {fetal_TNR}%\n'
          f'FPR - False Positive Rate: {fetal_FPR}%\n'
          f'PPV - Positive predictive value: {fetal_PPV}%\n'
          f'NPV - Negative predictive value: {fetal_NPV}%\n'
          f'F1 - Score: {fetal_f1}%')
    print("ROC - Receiving Operating Characteristics Score {:.2%}".format(roc_score), '\n')
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
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
    print("Accuracy: {:.2%}".format(accuracy), '\n') 
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
    print("Accuracy: {:.2%}".format(result), '\n')
    
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
    model = model.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Predicting using X_test_norm
    t1 = time()
    y_pred = model.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")

    # Computing for the accuracy, precision, & recall
    result = model.score(X_test, y_test)
    print("Accuracy: {:.2%}".format(result), '\n')
    
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
    print("Accuracy: {:.2%}".format(result), '\n')
    
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
    print("Accuracy: {:.2%}".format(result), '\n')
    
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
    print("Accuracy: {:.2%}".format(result), '\n')
    
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
        print("Accuracy: {:.2%}".format(result), [model])
        
    # define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])  
        
        