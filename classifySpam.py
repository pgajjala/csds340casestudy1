from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def predictTest(trainFeatures, trainLabels, testFeatures):
    best_features = [True, False, False,  True , True  ,True,  True, False,  True , True , True ,False,
 False ,False, False, False,  True, False, False , True, False ,False , True, False,
 False, False , True,  True  ,True,  True]

    trainFeatures = trainFeatures[:, best_features]
    testFeatures = testFeatures[:, best_features]
    
    # Make a pipeline using imputation
    knn = KNeighborsClassifier(n_neighbors=77)
    rbf_svm = SVC(kernel='rbf', C=5, gamma='scale', probability=True, class_weight='balanced')
    logistic = LogisticRegression(random_state=729, C=5, dual=False, penalty='l2')
    tree = DecisionTreeClassifier(random_state=729, criterion='gini')
    forest = RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2,
                                                 min_samples_split=10, n_estimators=50)

    estimators=[('knn', knn), ('svc', rbf_svm), ('rf', forest)]

    
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'), StandardScaler(), VotingClassifier(estimators, voting='soft'))

    
    # Fit the model and get the predicted probabilities
    model.fit(trainFeatures, trainLabels)
    testOutputs = model.predict_proba(testFeatures)[:, 1]

    return testOutputs
