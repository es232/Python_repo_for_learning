
"""
============================================================
MACHINE LEARNING FIELD GUIDE
============================================================

Topics:
1. Types of Machine Learning
2. Supervised Learning
   - Linear Regression
   - Polynomial Regression
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Naive Bayes
3. Unsupervised Learning
   - K-Means Clustering
   - Hierarchical Clustering
   - Principal Component Analysis (PCA)
4. Evaluation metrics
5. How to choose an algorithm
6. Real-world use cases and trade-offs

Requirements:
    pip install numpy pandas matplotlib scikit-learn scipy

Run:
    python ml_field_guide.py

Set RUN_ALL = True to run every demonstration.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import (
    make_classification,
    make_blobs,
    make_moons,
    load_iris
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

RUN_ALL = True
SHOW_PLOTS = False
RANDOM_STATE = 42


# ============================================================
# SECTION 1: TYPES OF MACHINE LEARNING
# ============================================================

"""
1. SUPERVISED LEARNING
----------------------
The model learns from labeled examples.

Input: Features (X)
Output: Known target labels or values (y)

Main tasks:
- Regression: predict a numerical value.
- Classification: predict a category.

Examples:
- House price prediction
- Spam email detection
- Loan approval prediction

Advantages:
- Can make useful predictions when labels are reliable.
- Performance can be measured against known answers.

Trade-offs:
- Requires labeled training data.
- Poor-quality labels can produce poor predictions.


2. UNSUPERVISED LEARNING
------------------------
The model discovers patterns in unlabeled data.

Examples:
- Customer segmentation
- Finding groups of similar documents
- Reducing the number of features

Advantages:
- Does not require target labels.
- Helps discover hidden structure.

Trade-offs:
- Discovered groups may not have obvious meaning.
- Evaluation can be difficult without ground truth.


3. REINFORCEMENT LEARNING
-------------------------
An agent learns by taking actions and receiving rewards
or penalties from an environment.

Examples:
- Robot navigation
- Game-playing agents
- Resource allocation

Advantages:
- Useful for sequential decisions.
- Can optimize long-term rewards.

Trade-offs:
- Training can require many interactions.
- Designing a suitable reward function is difficult.

Note:
This file focuses on supervised and unsupervised learning.
"""


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# SECTION 2: REGRESSION ALGORITHMS
# ============================================================


# ------------------------------------------------------------
# ALGORITHM 1: LINEAR REGRESSION
# ------------------------------------------------------------

"""
THEORY
------
Linear Regression predicts a continuous numerical value.

Equation:
    y = mx + c

For multiple features:
    y = b0 + b1*x1 + b2*x2 + ... + bn*xn

The model learns coefficients that minimize the sum of
squared differences between actual and predicted values.

WHEN TO USE
-----------
- House price estimation
- Sales forecasting
- Predicting energy consumption
- Estimating delivery time

ADVANTAGES
----------
- Simple and fast.
- Easy to explain.
- Useful as a baseline model.

TRADE-OFFS
----------
- Cannot naturally capture complex nonlinear relationships.
- Sensitive to outliers.
- Correlated features can make coefficients difficult to interpret.

EVALUATION
----------
MAE: average absolute prediction error.
MSE: average squared prediction error.
RMSE: square root of MSE.
R²: proportion of target variance explained by the model.
"""


def demo_linear_regression():
    print_section("1. LINEAR REGRESSION")

    # Example: predict marks from hours studied.
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([35, 40, 50, 55, 65, 70], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[7]])

    print("Hours studied: 7")
    print("Predicted marks:", round(prediction[0], 2))
    print("Slope:", round(model.coef_[0], 3))
    print("Intercept:", round(model.intercept_, 3))

    # For a real dataset, split data into training and testing sets.
    predicted = model.predict(X)

    print("Training-data MAE:",
          round(np.mean(np.abs(y - predicted)), 2))


# ------------------------------------------------------------
# ALGORITHM 2: POLYNOMIAL REGRESSION
# ------------------------------------------------------------

"""
THEORY
------
Polynomial Regression models nonlinear relationships by creating
polynomial features such as x² and x³.

Example:
    y = b0 + b1*x + b2*x²

It still uses linear regression to learn the coefficients.

WHEN TO USE
-----------
- Curved growth trends
- Nonlinear relationships between variables
- Some scientific and engineering measurements

ADVANTAGES
----------
- Can model curved relationships.
- Easy to build using existing regression tools.

TRADE-OFFS
----------
- High-degree polynomials may overfit.
- Predictions outside the training range can be unreliable.
- Feature count increases with polynomial degree.
"""


def demo_polynomial_regression():
    print_section("2. POLYNOMIAL REGRESSION")

    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([3, 6, 11, 18, 27, 38], dtype=float)

    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    )

    model.fit(X, y)

    print("Input x = 7")
    print("Predicted y:", round(model.predict([[7]])[0], 2))


# ============================================================
# SECTION 3: CLASSIFICATION ALGORITHMS
# ============================================================


# ------------------------------------------------------------
# ALGORITHM 3: LOGISTIC REGRESSION
# ------------------------------------------------------------

"""
THEORY
------
Despite its name, Logistic Regression is primarily used for
classification.

It estimates class probabilities using the logistic (sigmoid)
function:

    p = 1 / (1 + exp(-z))

For binary classification, a threshold such as 0.5 can convert
the predicted probability into a class.

WHEN TO USE
-----------
- Spam detection
- Loan default prediction
- Customer churn prediction
- Basic disease-risk classification

ADVANTAGES
----------
- Fast and interpretable.
- Provides class probabilities.
- Strong baseline for many classification tasks.

TRADE-OFFS
----------
- A basic model learns a linear decision boundary.
- May underperform on complicated nonlinear patterns.
- Feature scaling can help optimization.
"""


def demo_logistic_regression():
    print_section("3. LOGISTIC REGRESSION")

    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=RANDOM_STATE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("First prediction:", predictions[0])
    print("First sample probabilities:", probabilities[0])


# ------------------------------------------------------------
# ALGORITHM 4: K-NEAREST NEIGHBORS (KNN)
# ------------------------------------------------------------

"""
THEORY
------
KNN predicts a sample's class based on the classes of its
nearest training examples.

For classification, it commonly uses majority voting.

Example:
If the five nearest customers contain four "likely to buy"
labels and one "unlikely to buy" label, the prediction is
"likely to buy".

WHEN TO USE
-----------
- Small or medium-sized datasets
- Similarity-based classification
- Simple recommendation prototypes

ADVANTAGES
----------
- Easy to understand.
- Little explicit training.
- Can model nonlinear decision boundaries.

TRADE-OFFS
----------
- Prediction can be slow on large datasets.
- Sensitive to feature scales.
- Performance depends on K and distance metric.
- High-dimensional data can make distances less informative.
"""


def demo_knn():
    print_section("4. K-NEAREST NEIGHBORS")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=iris.target
    )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Example predicted class:", iris.target_names[predictions[0]])


# ------------------------------------------------------------
# ALGORITHM 5: DECISION TREE
# ------------------------------------------------------------

"""
THEORY
------
A Decision Tree repeatedly splits data using feature-based rules.

Example:
    If income > threshold:
        If debt < threshold:
            predict low risk
        Else:
            predict high risk
    Else:
        predict high risk

WHEN TO USE
-----------
- Rule-based decision support
- Loan-risk analysis
- Customer classification
- Explainable prototypes

ADVANTAGES
----------
- Easy to visualize and explain.
- Handles nonlinear relationships.
- Does not require feature scaling.

TRADE-OFFS
----------
- Deep trees can overfit.
- Small data changes can produce a different tree.
- A single tree may generalize worse than an ensemble.
"""


def demo_decision_tree():
    print_section("5. DECISION TREE")

    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=RANDOM_STATE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Feature importances:", model.feature_importances_)


# ------------------------------------------------------------
# ALGORITHM 6: RANDOM FOREST
# ------------------------------------------------------------

"""
THEORY
------
Random Forest combines predictions from many decision trees.

Each tree is trained with randomness in its training samples
and/or candidate features.

Classification typically uses voting across trees.

WHEN TO USE
-----------
- Customer churn
- Fraud detection
- Tabular business datasets
- General-purpose classification baselines

ADVANTAGES
----------
- Often more robust than one decision tree.
- Captures nonlinear relationships.
- Usually does not require feature scaling.
- Can estimate feature importance.

TRADE-OFFS
----------
- Larger and slower than a single tree.
- Harder to explain as a whole.
- Feature importance can be misleading with correlated features.
"""


def demo_random_forest():
    print_section("6. RANDOM FOREST")

    X, y = make_classification(
        n_samples=600,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=RANDOM_STATE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Feature importances:", model.feature_importances_)


# ------------------------------------------------------------
# ALGORITHM 7: SUPPORT VECTOR MACHINE (SVM)
# ------------------------------------------------------------

"""
THEORY
------
SVM finds a decision boundary that separates classes while
seeking a large margin between them.

The kernel trick allows nonlinear decision boundaries without
explicitly constructing every transformed feature.

Common kernels:
- Linear
- Polynomial
- RBF (Radial Basis Function)

WHEN TO USE
-----------
- Text classification with suitable representations
- Small or medium-sized structured datasets
- Classification with complex boundaries

ADVANTAGES
----------
- Can work well in high-dimensional spaces.
- RBF kernels can model nonlinear patterns.
- Often effective on smaller datasets.

TRADE-OFFS
----------
- Parameter and kernel selection matters.
- Scaling is generally important.
- Training can become expensive on large datasets.
- Probabilities require additional calibration.
"""


def demo_svm():
    print_section("7. SUPPORT VECTOR MACHINE")

    X, y = make_moons(
        n_samples=500,
        noise=0.2,
        random_state=RANDOM_STATE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0)
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))


# ------------------------------------------------------------
# ALGORITHM 8: NAIVE BAYES
# ------------------------------------------------------------

"""
THEORY
------
Naive Bayes uses Bayes' theorem to estimate the probability
of a class given the observed features.

It assumes features are conditionally independent given
the class. This assumption is often unrealistic, but the
algorithm can still work well.

GaussianNB assumes each feature follows a Gaussian distribution
within each class.

WHEN TO USE
-----------
- Spam filtering
- Text classification
- Fast classification baselines
- Some document categorization tasks

ADVANTAGES
----------
- Very fast.
- Works well with suitable high-dimensional features.
- Requires relatively little training data.

TRADE-OFFS
----------
- The independence assumption can be inaccurate.
- Probability estimates may be poorly calibrated.
- GaussianNB is not the standard choice for every feature type.
"""


def demo_naive_bayes():
    print_section("8. NAIVE BAYES")

    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=RANDOM_STATE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("First five predictions:", predictions[:5])


# ============================================================
# SECTION 4: UNSUPERVISED LEARNING
# ============================================================


# ------------------------------------------------------------
# ALGORITHM 9: K-MEANS CLUSTERING
# ------------------------------------------------------------

"""
THEORY
------
K-Means divides samples into K clusters.

Basic process:
1. Choose K initial centroids.
2. Assign each sample to its nearest centroid.
3. Recalculate centroids.
4. Repeat until assignments or centroids stabilize.

No target labels are required.

WHEN TO USE
-----------
- Customer segmentation
- Grouping similar products
- Exploring patterns in numerical data

ADVANTAGES
----------
- Simple and fast.
- Scales to relatively large datasets.
- Easy to interpret when clusters are well separated.

TRADE-OFFS
----------
- K must be selected.
- Sensitive to feature scales and initialization.
- Can perform poorly on irregularly shaped clusters.
- Outliers can affect centroids.
"""


def demo_kmeans():
    print_section("9. K-MEANS CLUSTERING")

    X, actual_groups = make_blobs(
        n_samples=400,
        centers=4,
        cluster_std=1.0,
        random_state=RANDOM_STATE
    )

    model = make_pipeline(
        StandardScaler(),
        KMeans(
            n_clusters=4,
            n_init=10,
            random_state=RANDOM_STATE
        )
    )

    cluster_labels = model.fit_predict(X)

    # Silhouette score is meaningful when there are at least
    # two clusters and fewer clusters than samples.
    scaled_X = model.named_steps["standardscaler"].transform(X)

    print("Cluster counts:")
    print(pd.Series(cluster_labels).value_counts().sort_index())

    print(
        "Silhouette score:",
        round(silhouette_score(scaled_X, cluster_labels), 3)
    )

    print(
        "Note: cluster labels are arbitrary numbers, not true categories."
    )


# ------------------------------------------------------------
# ALGORITHM 10: HIERARCHICAL CLUSTERING
# ------------------------------------------------------------

"""
THEORY
------
Hierarchical clustering builds a hierarchy of clusters.

Agglomerative clustering begins with each sample as its own
cluster and repeatedly merges clusters.

The hierarchy can be visualized using a dendrogram when
using suitable SciPy linkage functions.

WHEN TO USE
-----------
- Exploring relationships among samples
- Small or medium datasets
- Grouping similar documents or biological samples

ADVANTAGES
----------
- Does not always require choosing the number of clusters
  before building the hierarchy.
- Reveals nested group structure.
- Multiple distance and linkage strategies are available.

TRADE-OFFS
----------
- Can be computationally expensive.
- Sensitive to distance and linkage choices.
- Scaling matters for numerical features.
"""


def demo_hierarchical():
    print_section("10. HIERARCHICAL CLUSTERING")

    X, _ = make_blobs(
        n_samples=200,
        centers=3,
        random_state=RANDOM_STATE
    )

    X_scaled = StandardScaler().fit_transform(X)

    model = AgglomerativeClustering(n_clusters=3)
    labels = model.fit_predict(X_scaled)

    print("Cluster counts:")
    print(pd.Series(labels).value_counts().sort_index())

    print(
        "Silhouette score:",
        round(silhouette_score(X_scaled, labels), 3)
    )


# ------------------------------------------------------------
# ALGORITHM 11: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ------------------------------------------------------------

"""
THEORY
------
PCA is a dimensionality-reduction technique.

It transforms features into new directions called principal
components, ordered by how much variance they explain.

Example:
A dataset with 100 features may be represented by 10 components
while preserving a large proportion of its variation.

PCA is not a prediction algorithm by itself.

WHEN TO USE
-----------
- Visualizing high-dimensional data
- Reducing the number of numerical features
- Compressing data
- Preprocessing before some ML algorithms

ADVANTAGES
----------
- Can reduce dimensionality and noise.
- May reduce storage and computation.
- Helps visualize data in two or three dimensions.

TRADE-OFFS
----------
- Components can be difficult to interpret.
- Some information is lost when components are discarded.
- Captures variance, not necessarily predictive usefulness.
- Feature scaling is often necessary.
"""


def demo_pca():
    print_section("11. PRINCIPAL COMPONENT ANALYSIS")

    iris = load_iris()
    X = iris.data

    # PCA is sensitive to feature scales, so scale first.
    X_scaled = StandardScaler().fit_transform(X)

    model = PCA(n_components=2)
    X_reduced = model.fit_transform(X_scaled)

    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)

    print(
        "Variance explained by each component:",
        np.round(model.explained_variance_ratio_, 3)
    )

    print(
        "Total variance explained:",
        round(model.explained_variance_ratio_.sum(), 3)
    )


# ============================================================
# SECTION 5: EVALUATION METRICS
# ============================================================

"""
CLASSIFICATION METRICS
---------------------

Accuracy:
    Correct predictions / all predictions.
    Useful when class balance and error costs are appropriate.

Precision:
    Of predicted positives, how many were actually positive?
    Important when false positives are expensive.

Recall:
    Of actual positives, how many did the model find?
    Important when false negatives are expensive.

F1-score:
    Harmonic mean of precision and recall.
    Useful when balancing both matters.

Confusion matrix:
    Shows correct and incorrect predictions by class.

REGRESSION METRICS
------------------

MAE:
    Average absolute error.
    Easy to interpret in the target's units.

MSE:
    Average squared error.
    Penalizes large errors more strongly.

RMSE:
    Square root of MSE.
    Expressed in the target's units.

R²:
    Measures improvement relative to predicting the target mean.
    Can be negative on test data.

CLUSTERING METRICS
------------------

Silhouette score:
    Measures how close each point is to its own cluster
    compared with other clusters.
    Higher is generally better, but interpret in context.
"""


def demo_evaluation_metrics():
    print_section("12. EVALUATION METRICS")

    actual = np.array([100, 200, 300, 400])
    predicted = np.array([110, 190, 310, 390])

    print("Regression MAE:",
          round(np.mean(np.abs(actual - predicted)), 2))

    print("Regression MSE:",
          round(mean_squared_error(actual, predicted), 2))

    print("Regression RMSE:",
          round(np.sqrt(mean_squared_error(actual, predicted)), 2))

    print("Regression R²:",
          round(r2_score(actual, predicted), 3))

    actual_classes = np.array([1, 1, 0, 0, 1, 0])
    predicted_classes = np.array([1, 0, 0, 0, 1, 1])

    print("Classification accuracy:",
          round(accuracy_score(actual_classes, predicted_classes), 3))

    print("Precision:",
          round(precision_score(actual_classes, predicted_classes), 3))

    print("Recall:",
          round(recall_score(actual_classes, predicted_classes), 3))

    print("F1:",
          round(f1_score(actual_classes, predicted_classes), 3))


# ============================================================
# SECTION 6: HOW TO CHOOSE AN ALGORITHM
# ============================================================

"""
PRACTICAL STARTING GUIDE
------------------------

Need to predict a numerical value?
    Start with Linear Regression.
    Try tree-based regression if relationships are nonlinear.

Need to predict a category?
    Start with Logistic Regression.
    Compare Random Forest, SVM, KNN, or Naive Bayes as appropriate.

Need explainable if-then rules?
    Try a shallow Decision Tree or Logistic Regression.

Need a strong baseline for tabular data?
    Try Random Forest and compare with simpler models.

Need to group customers without labels?
    Try K-Means; inspect clusters and validate their usefulness.

Need to explore nested groups?
    Try Hierarchical Clustering.

Need fewer features or a 2D visualization?
    Try PCA after appropriate scaling.

Important:
There is no universally best algorithm. Compare candidates on
held-out data using metrics appropriate to the problem.

Avoid data leakage:
- Split data before fitting preprocessing transformations.
- Fit scalers and other preprocessing only on training data.
- Use a pipeline to keep preprocessing within training folds.
- Do not select a model using the test set repeatedly.
"""


# ============================================================
# SECTION 7: A REUSABLE CLASSIFICATION REPORT
# ============================================================

def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    """
    Train a classification model and print common metrics.

    Args:
        model: An sklearn-compatible estimator or pipeline.
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test labels.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Precision:", round(
        precision_score(y_test, predictions, average="weighted",
                        zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, average="weighted",
                     zero_division=0), 3
    ))
    print("F1:", round(
        f1_score(y_test, predictions, average="weighted",
                 zero_division=0), 3
    ))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


# ============================================================
# SECTION 8: RUN ALL DEMONSTRATIONS
# ============================================================

def main():
    print_section("MACHINE LEARNING FIELD GUIDE")
    print("Python version:", __import__("sys").version.split()[0])
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)

    demonstrations = [
        demo_linear_regression,
        demo_polynomial_regression,
        demo_logistic_regression,
        demo_knn,
        demo_decision_tree,
        demo_random_forest,
        demo_svm,
        demo_naive_bayes,
        demo_kmeans,
        demo_hierarchical,
        demo_pca,
        demo_evaluation_metrics
    ]

    if RUN_ALL:
        for demonstration in demonstrations:
            demonstration()
    else:
        # Change this to any function above to run one topic.
        demo_linear_regression()

    print_section("ALL SELECTED DEMONSTRATIONS COMPLETED")
    print("Next step: practice each algorithm on a real dataset.")


if __name__ == "__main__":
    main()
