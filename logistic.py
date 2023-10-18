import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np


def predictTest(trainFeatures, trainLabels, testFeatures):
    # Select the best features as determined by SBS
    best_features = [True,  True , True , True , True , True , True , True , True , True , True , True,
  True , True , True , True , True , True , True  ,True , True , True , True , True,
  True , True , True , True , True, True]

    trainFeatures = trainFeatures.loc[:, best_features]
    testFeatures = testFeatures.loc[:, best_features]

    imputer = SimpleImputer(missing_values=-1, strategy='mean')
    imputer.fit(trainFeatures)
    trainFeatures = imputer.transform(trainFeatures)
    testFeatures = imputer.transform(testFeatures)

    scaler = StandardScaler().fit(trainFeatures)
    trainFeatures = scaler.transform(trainFeatures)
    testFeatures = scaler.transform(testFeatures)

    # Make a pipeline using imputation
    model = LogisticRegression(random_state=729, C=0.1, dual=False, penalty='l2')

    # Fit the model and get the predicted probabilities
    model.fit(trainFeatures, trainLabels)
    testOutputs = model.predict_proba(testFeatures)[:,1]

    return testOutputs


if __name__ == '__main__':
    # Read the data
    df = pd.read_csv("spamTrain1.csv", header=None)

    # Split features and label
    X, y = df.drop(columns=[30]), df[30]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=729, stratify=y)

    # Predict the labels and get the ROC AUC score
    out = predictTest(X_train, y_train, X_test)
    score = roc_auc_score(y_test, out)

    # Best ROC AUC Score: 0.9024
    print(f"ROC AUC Score: {score}")

    # Get the full ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, out)

    # Calculate the TPR at a 1% FPR
    desired_fpr = 0.01
    idx = np.argmax(fpr >= desired_fpr)

    # Get the TPR at the found index
    specific_tpr = tpr[idx]

    # Best TPR at 1% FPR: 0.3742
    print(f"True Positive Rate at {desired_fpr * 100:.2f}% False Positive Rate: {specific_tpr * 100:.2f}%")
