import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def tprAtFPR(labels, outputs, desiredFPR):
    fpr, tpr, thres = roc_curve(labels, outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = ((tprAbove - tprBelow) / (fprAbove - fprBelow) * (desiredFPR - fprBelow)
             + tprBelow)
    return tprAt, fpr, tpr

def predictTest(trainFeatures, trainLabels, testFeatures):
    # Select the best features as determined by SBS

    # best_features = [True, False, False,  True,  True, False, True,  True,  True , True , True ,False,
    #                 False ,False ,False ,False , True ,False,  True,  True,  True ,False , True, False,
    #                 False , True  ,True  ,True , True , True]
    # avg roc:0.8934105872484194, avg tpr:0.3456632255129843
    # std dev roc: 0.006066215457450894, std dev tpr:0.06621181558487337



    # best_features = [True, True, False, True, True, False, True, True, True, True, True, True, False, True, True, False,
    #                  True, True, True, True, True, False, True, True, True, True, True, True, False, True]
    # avg roc:0.887940008311538, avg tpr:0.34109692144409565
    # std dev roc: 0.00706286071857137, std dev tpr:0.04187287411390205

    
    best_features = [True, False, False,  True , True  ,True,  True, False,  True , True , True ,False,
 False ,False, False, False,  True, False, False , True, False ,False , True, False,
 False, False , True,  True  ,True,  True]
    # avg roc:0.8941461104382497, avg tpr:0.36996069349949323
    # std dev roc: 0.0059241333447850675, std dev tpr:0.06299124129256851


    trainFeatures = trainFeatures[:, best_features]
    testFeatures = testFeatures[:, best_features]
    
    # Make a pipeline using imputation
    knn = KNeighborsClassifier(n_neighbors=77)
    rbf_svm = SVC(kernel='rbf', C=5, gamma='scale', probability=True, class_weight='balanced')
    logistic = LogisticRegression(random_state=729, C=5, dual=False, penalty='l2')
    tree = DecisionTreeClassifier(random_state=729, criterion='gini')
    forest = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'),
                          RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2,
                                                 min_samples_split=10, n_estimators=50))

    estimators=[('knn', knn), ('svc', rbf_svm), ('rf', forest)]

    
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'), VotingClassifier(estimators, voting='soft'))

    
    # Fit the model and get the predicted probabilities
    model.fit(trainFeatures, trainLabels)
    testOutputs = model.predict_proba(testFeatures)[:, 1]

    return testOutputs


if __name__ == '__main__':

    # Read and shuffle the training data
    df = np.loadtxt("spamTrain1.csv", delimiter=',')
    np.random.shuffle(df)

    # Read and shuffle the testing data
    df_test = np.loadtxt("spamTrain2.csv", delimiter=',')
    np.random.shuffle(df_test)

    # Assign the first half of the data to train and the second half to test
    X_train, X_test = np.concatenate((df[:750,:30], df_test[:750, :30]), axis=0), \
                    np.concatenate((df[750:, :30], df_test[750:, :30]), axis=0)

    y_train, y_test = np.concatenate((df[:750, 30], df_test[:750, 30]), axis=0), \
                    np.concatenate((df[750:, 30], df_test[750:, 30]), axis=0)

    # Predict the labels and get the ROC AUC score
    out = predictTest(X_train, y_train, X_test)
    score = roc_auc_score(y_test, out)
    print(f"ROC AUC Score: {score}")

    # Get the full ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, out)

    # Calculate the TPR at a 1% FPR
    desired_fpr = 0.01
    idx = np.argmax(fpr >= desired_fpr)

    # Get the TPR at the found index
    specific_tpr = tpr[idx]

    print(f"True Positive Rate at {desired_fpr * 100:.2f}% False Positive Rate: {tprAtFPR(y_test, out, 0.01)[0]}")