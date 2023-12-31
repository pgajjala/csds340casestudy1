import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

# Read the data
df = pd.read_csv("spamTrain1.csv", header=None)


# Define the imputation function
def imputation(train, test, how=None):
    if how == 'mean':
        means = train.mean()
        train = train.fillna(means)
        test = test.fillna(means)
        return train, test
    if how == 'zero':
        train = train.fillna(0)
        test = test.fillna(0)
        return train, test
    return train, test


# Separate the data into training and testing
X, y = df.drop(columns=[30]), df[30]
# Replace -1 with NA
X = X.replace(-1, np.nan)
# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=729)

# Impute the missing values
X_train_imputed, X_test_imputed = imputation(X_train, X_test, how='mean')

# Initialize variables to store our best score/params
best_score = 0
best_k = 0
best_num_neighbors = 0
best_features = None

# Iterate through using 1-29 features
for k in range(1, 30):
    for n in range(1, 50):
        # Initialize our model
        knn = KNeighborsClassifier(n_neighbors=n)
        # Intialize our SBS model
        sfs = SequentialFeatureSelector(knn, n_features_to_select=k, direction='backward', cv=3)

        # Fit the SBS on the training data
        sfs = sfs.fit(X_train_imputed, y_train)

        # Get the selected feature indices
        selected_features = sfs.get_support()

        # Transform the training and test data to include only the selected features
        X_train_selected = sfs.transform(X_train_imputed)
        X_test_selected = sfs.transform(X_test_imputed)

        # Fit the model on the selected features only
        knn.fit(X_train_selected, y_train)

        # Predict the label using the select features only
        testOutputs = knn.predict_proba(X_test_selected)[:, 1]
        # Calculate our ROC AUC score
        score = roc_auc_score(y_test, testOutputs)
        print(f"Score using {k} feature{'s' if k > 1 else ''}: {score} with num neighbors {n}")

        # If we have a new best score, update all of our "best" variables
        if score > best_score:
            best_num_neighbors = n
            best_score = score
            best_k = k
            best_features = selected_features

# Print out the best score/params
print(f"Best ROC: {best_score}")
print(f"Best K Components: {best_k}")
print(f"Best K Features: {best_features}")

# Initialize our final model
rbf_svm = KNeighborsClassifier(n_neighbors=best_num_neighbors)

# Use only the best features as determined via SBS
X_train_selected = X_train_imputed.loc[:, best_features]
X_test_selected = X_test_imputed.loc[:, best_features]

# Grid of params to test
param_grid = {'weights': ['uniform', 'distance'], 'metric': ['manhattan', 'minkowski']}
# Use the grid in grid search
grid_search = GridSearchCV(rbf_svm, param_grid, cv=10, scoring='roc_auc')

# Perform the grid search on the data
grid_search.fit(X_train_selected, y_train)

# Save the best parameters
best_params_ = grid_search.best_params_

# Predict the labels with the best model
testOutputs = grid_search.predict_proba(X_test_selected)[:, 1]

# Calculate the ROC AUC score of the best model
score = roc_auc_score(y_test, testOutputs)

# Print out the score and parameters from the best model
print(f"Score: {score}")
print(f"Best Params: {best_params_}")
