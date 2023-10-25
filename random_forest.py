from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
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
    best_features = [True, True, False, True, True, False, True, True, True, True, True, True, False, True, True, False,
                     True, True, True, True, True, False, True, True, True, True, True, True, False, True]

    trainFeatures = trainFeatures[:, best_features]
    testFeatures = testFeatures[:, best_features]

    # Make a pipeline using mean imputation
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'),
                          RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2,
                                                 min_samples_split=10, n_estimators=50))

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
