import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=20)

unique_classes = np.unique(y_train)
print("Labels:", unique_classes)
class_data = {label: X_train[y_train == label] for label in unique_classes}

means = {label: np.mean(class_data[label], axis=0) for label in unique_classes}
variances = {label: np.var(class_data[label], axis=0) for label in unique_classes}


def univariate_gaussian_likelihood(x, mean, var):
    coeff = 1 / np.sqrt(2 * np.pi * var)
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return coeff * exponent

def compute_likelihood(test_sample, means, variances, label):
    likelihood = 1
    for feature_index in range(len(test_sample)):
        likelihood *= univariate_gaussian_likelihood(
            test_sample[feature_index], means[label][feature_index], variances[label][feature_index])
    return likelihood


priors = {label: len(class_data[label]) / len(X_train) for label in unique_classes}
print("Priors:", priors)

def classify_sample(test_sample):
    likelihoods = {label: compute_likelihood(test_sample, means, variances, label) for label in unique_classes}
    posteriors = {label: likelihoods[label] * priors[label] for label in unique_classes}
    evidence = sum(posteriors.values())

    normalized_posteriors = {label: posteriors[label] / evidence for label in unique_classes}

    predicted_class = max(normalized_posteriors, key=normalized_posteriors.get)
    return predicted_class, normalized_posteriors


correct_predictions = 0
predictions = []
for test_sample, true_label in zip(X_test, y_test):
    predicted_class, posteriors = classify_sample(test_sample)
    predictions.append(predicted_class)
    if predicted_class == true_label:
        correct_predictions += 1
    #print(f"Test Sample: {test_sample}")
    #print(f"  Posterior Probabilities: {posteriors}")
    #print(f"  Predicted Class: {predicted_class}, True Class: {true_label}")

# Compute accuracy
accuracy = correct_predictions / len(X_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
