# File name: Ada_Boost.py
# https://en.wikipedia.org/wiki/AdaBoost
# AdaBoost Algorithm

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

import warnings
warnings.filterwarnings("ignore")


class Stump(object):
    """ Weak Learner --> Decision Stumps"""
    """ 
        The weak learners in AdaBoost are decision trees with a single split, called decision stumps.
    """

    def __init__(self):

        self.weak_hypothesis = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)

        # Weak learners hypothesis: h:X --> {-1, 1}
        if self.weak_hypothesis == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost(object):
    """ Adaptive Boosting """
    """
        Adaboost helps you combine multiple “weak classifiers” into a single “strong classifier”. 
        AdaBoost algorithms can be used for both classification and regression problem.
    """

    def __init__(self, n_classifiers=5):
        self.n_classifiers = n_classifiers
        # Ensemble: Create a list to store n stumps
        self.stumps = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initial weights with 1\n
        weights = np.full(n_samples, 1 / n_samples)

        # Iterate through n stumps (T)
        for _ in range(self.n_classifiers):
            # Create a weak learner - stump
            stp = Stump()
            # Set the minimum error to infinity
            min_total_error = float('inf')

            # Find weak learner that minimizes the weighted sum error for misclassified points
            # Greedy search to find best threshold and feature
            for idx in range(n_features):
                X_column = X[:, idx]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # Initially set weak_hypothesis = 1
                    weak_hypothesis = 1
                    # Initialize the predictions - h_xi
                    h_xi_predictions = np.ones(n_samples)
                    # Set the prediction = -1 if the feature is less that threshold
                    h_xi_predictions[X_column < threshold] = -1

                    # Find the sum weights of misclassified points
                    # Aim: select h with low weighted error: E = ∑i=1:n w[h_xi ≠ y_i]
                    weights_of_misclassified_points = weights[h_xi_predictions != y]
                    error_rate = sum(weights_of_misclassified_points)

                    """
                        1.  The classifier weight grows exponentially as the error approaches 0. 
                            Better classifiers are given exponentially more weight.
                        2.  The classifier weight is zero if the error rate is 0.5. 
                            A classifier with 50% accuracy is no better than random guessing, so we ignore it.
                        3.  The classifier weight grows exponentially negative as the error approaches 1. 
                            We give a negative weight to classifiers with worse worse than 50% accuracy. 
                            “Whatever that classifier says, do the opposite!”.
                    """
                    # Check if weighted sum error for misclassified points  > 0.5:
                    if error_rate > 0.5:
                        # flip the error_rate to 1 - error_rate “Whatever that classifier says, do the opposite!”.
                        error_rate = 1 - error_rate
                        # Set weak_hypothesis = -1
                        weak_hypothesis = -1

                    # Store the as a weak learner stump
                    if error_rate < min_total_error:
                        stp.weak_hypothesis = weak_hypothesis
                        stp.threshold = threshold
                        stp.feature_idx = idx
                        min_total_error = error_rate

            # Voting Power - Amount of say: Calculate alpha - The negative logit function multiplied by 0.5
            # add some epsilon ε (small noise term) to prevent zero division - this will happen
            # if total error is 1 or 0
            eps = 1e-10
            stp.alpha = 0.5 * np.log((1.0 - min_total_error + eps) / (min_total_error + eps))
            # Calculate the weak learner predictions
            predictions = stp.predict(X)
            # Update weights
            weights = weights * np.exp(-stp.alpha * y * predictions)
            # Re-normalize
            weights = weights / np.sum(weights)
            # Add to the ensemble: Store the stumps in the list
            self.stumps.append(stp)

            # Use the modified weights to make the next stump in the tree

    def predict(self, X):
        # sign(∑t:T [alpha_t .h(X)]),    h:X --> {-1, 1}
        predictions = [stm.alpha * stm.predict(X) for stm in self.stumps]
        y_prediction = np.sum(predictions, axis=0)
        """
            the final output is just a linear combination of all of the weak classifiers, and then we 
            make our final decision simply by looking at the sign of this sum.
        """
        y_prediction = np.sign(y_prediction)
        return y_prediction


def accuracy(y_tst, predictions):
    return np.sum(y_tst == predictions) / len(y_tst)


def confusion_matrix_accuracy(cm):
    """
    confusion_matrix_accuracy method
    :param cm: {array-like}
    :return: {float}
    """
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    return diagonal_sum / sum_of_all_elements


if __name__ == '__main__':
    # Load Brest Cancer data
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target

    # data = datasets.load_digits()
    """iris = datasets.load_iris()
    X = iris.data
    y = iris.target"""

    # Desired outputs y1, ..., yn ∈{-1, 1}
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size=0.2
                                                        , random_state=5)

    # AdaBoost with 5 weak learners - stumps
    ab = Adaboost(n_classifiers=50)
    # Fit
    ab.fit(X_train, y_train)
    # Predict
    y_predicted = ab.predict(X_test)

    print("\nAccuracy Score: : {%.2f%%}" % (accuracy(y_test, y_predicted) * 100.0))

    cm = confusion_matrix(y_test, y_predicted)
    _confusion_matrix_accuracy = confusion_matrix_accuracy(cm)
    print("Confusion Matrix: {%.2f%%}" % (_confusion_matrix_accuracy * 100.0))
    f1_test2 = f1_score(y_test, y_predicted, average='weighted')
    print("F1 SCORE for test set: {%.2f%%}" % (f1_test2 * 100.0))



