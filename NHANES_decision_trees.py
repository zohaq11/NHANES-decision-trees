"""
CSC311 Lab 2: Decision Trees and Accuracy-based Diagnostics

In this lab, we will explore the features in this data set, use `sklearn` to fit a
decision tree to our data, and do some work to select hyperparameters that maximize
accuracy (or minimize the number of our classification mistakes). 

Acknowledgements:
- Thanks to https://www.kaggle.com/code/tobyanderson/health-survey-analysis for some utilities to decode NHANES categories!
- This lab was created in collaboration with, Prof. Sonya Allin, Mustafa Haiderbhai, Carolyn Quinlan, Brandon Jaipersaud and others.
"""

import matplotlib.pyplot as plt # For plotting
import numpy as np # Linear algebra library
import pandas as pd # For reading CSV files and manipulating tabular data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as treeViz
import graphviz
import pydotplus
from IPython.display import display

# Download data if not already present
try:
    data = pd.read_csv("NHANES-heart.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Downloading dataset...")
    import urllib.request
    url = "https://www.cs.toronto.edu/~lczhang/311/lab02/NHANES-heart.csv"
    urllib.request.urlretrieve(url, "NHANES-heart.csv")
    print("Dataset downloaded successfully!")
    data = pd.read_csv("NHANES-heart.csv")

# Part 1: Data Exploration
print("DATA EXPLORATION")   

print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Display basic statistics
print("\nDataset Description:")
print(data.describe())

# Display target variable distribution
print("\nTarget Variable Distribution:")
print(data['target_heart'].value_counts())

# Display categorical variable distributions
print("\nGender Distribution:")
print(data['gender'].value_counts())

print("\nDrink Alcohol Distribution:")
print(data['drink_alcohol'].value_counts())

print("\nChest Pain Ever Distribution:")
print(data['chest_pain_ever'].value_counts())

# Create box plots for numerical features
numerical_features = ['age', 'BMI', 'weight_kg', 'blood_pressure_sys', 'diastolic_bp', 
                      'blood_cholesterol', 'calories', 'family_income']

print("\nCreating box plots for numerical features...")
for feature in numerical_features:
    plt.figure()
    plt.title(f"Box Plot Showing the Distribution of '{feature}'")
    plt.boxplot(data[feature])
    plt.ylabel(feature)
plt.show()

# Tabulate frequency for categorical features
categorical_features =  ['gender', 'chest_pain_ever', 'drink_alcohol', 'target_heart']
for feature in categorical_features:
    print(f"\nFrequency Table for '{feature}':")
    print(data[feature].value_counts())

"""Note: From above, we can see that the dataset is relatively balanced in terms of the 
target variable (heart disease vs no heart disease). Allowing the training data to reflect 
the true prevalence of heart disease could bias the decision tree toward predicting the 
majority class target_heart=0. This may cause the tree to favor splitting criteria that better 
separate the this class while neglecting meaningful splits for the minority class target_heart=1. 
This can lead to false negatives, where cases of heart disease are overlooked, making the model 
less effective in real-world applications where false negatives could be deadly. 
Balancing the dataset ensures the model learns to recognize both classes effectively."""

# Create box plots comparing heart disease vs no heart disease
print("\nCreating comparative box plots...")
for feature in ['blood_cholesterol', 'age', 'calories', 'BMI']:
    plt.figure()
    data.boxplot(column=feature, by='target_heart')
    plt.title(f'{feature} Distribution by Heart Disease')
    plt.xlabel('Heart Disease (0=No, 1=Yes)')
    plt.ylabel(feature)
plt.show()

# Cross-tabulation for categorical variables
for feature in ['gender', 'drink_alcohol', 'chest_pain_ever']:
    print(f"\nCross-tabulation of '{feature}' with Heart Disease:")
    print(pd.crosstab(data['target_heart'], data[feature]))

"""
I expect age to be an informative predictor for "target_heart" as there are
vast differences in the box plots, however that does not seem to be the case
for "blood_cholesterol", "calories", and "BMI". Since there are approx the 
same amount of males and females but somewhat more males seem to have heart 
disease, gender could also be a factor, similarly with experiencing chest
pain, more with also have heart disease thus that is definitely a predictor
however with having alcohol or not, both seem to have a similar split.
"""

# Data quality checks
print("\nMissing Data Summary:")
print(data.isnull().sum())

print("\nChecking for unexpected values...")
if data['age'].max() > 120 or data['age'].min() < 0:
    print("Unexpected age values detected.")
if data['blood_pressure_sys'].min() < 50 or data['blood_pressure_sys'].max() > 250:
    print("Unexpected systolic blood pressure values detected.")
if data['diastolic_bp'].min() < 30 or data['diastolic_bp'].max() > 150:
    print("Unexpected diastolic blood pressure values detected.")

# Feature Engineering
print("\nFEATURE ENGINEERING")
"""
Encoding race_ethnicity and gender as discrete numerical values can create problems because 
decision trees might treat these values as if they have a ranking or meaningful numerical difference. 
Additionally, a value like 3.2 for race_ethnicity has no real-world meaning and does not represent any 
mixed race or valid category. Similarly, using numbers for gender implies a comparison or difference that 
doesn’t exist, leading to misleading splits in the tree. This can result in poor model performance and 
inaccurate predictions.
"""

# Convert categorical features to indicator variables
data_fets = np.stack([
    # gender_female: this code creates an array of booleans, which converted into 0 and 1
    data["gender"] == 2,
    # re_hispanic: this code leverages addition to perform an "or" operation
    (data["race_ethnicity"] == 1) + (data["race_ethnicity"] == 2),
    # re_white
    data["race_ethnicity"] == 3,
    # re_black
    data["race_ethnicity"] == 4,
    # re_asian
    data["race_ethnicity"] == 6,
    # chest_pain_ever
    data["chest_pain_ever"] == 1,
    # drink_alcohol
    data["drink_alcohol"] == 1,
    # numeric values so no transformations are required
    data["age"],
    data["blood_cholesterol"],
    data["BMI"],
    data["blood_pressure_sys"],
    data["diastolic_bp"],
    data["calories"],
    data["family_income"],
], axis=1)

print(f"Feature matrix shape: {data_fets.shape}") # should be (8000, 14)

# Feature names for visualization
feature_names = [
    "gender_female",
    "re_hispanic",
    "re_white",
    "re_black",
    "re_asian",
    "chest_pain",
    "drink_alcohol",
    "age",
    "blood_cholesterol",
    "BMI",
    "blood_pressure_sys",
    "diastolic_bp",
    "calories",
    "family_income"
]

# Data Splitting
print("\nDATA SPLITTING")

# Split the data into X (dependent variables) and t (response variable)
X = data_fets
t = np.array(data["target_heart"])

# First, we will use `train_test_split` to split the data set into
# 6500 training+validation, and 1500 test:
X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=1500/8000, random_state=1)

# Then, use `train_test_split` to split the training+validation data
# into 5000 train and 1500 validation
X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=1500/6500, random_state=1)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_valid.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

def visualize_tree(model, max_depth=5, filename="tree"):
    """
    Generate and return an image representing an Sklearn decision tree.

    Each node in the visualization represents a node in the decision tree.
    In addition, visualization for each node contains:
        - The feature that is split on
        - The entropy (of the outputs `t`) at the node
        - The number of training samples at the node
        - The number of training samples with true/false values
        - The majority class (heart disease or not)
    The colour of the node also shows the majority class and purity

    See here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

    Parameters:
        `model` - An Sklearn decision tree model
        `max_depth` - Max depth of decision tree to be rendered.
         This is useful since the tree can get very large if the max_depth is
         set too high and thus making the resulting figure difficult to interpret.
    """
    dot_data = treeViz.export_graphviz(model,
                                       feature_names=feature_names,
                                       max_depth=max_depth,
                                       class_names=["heart_no", "heart_yes"],
                                       filled=True,
                                       rounded=True)
    # return display(graphviz.Source(dot_data))

    # Convert to image
    graph = graphviz.Source(dot_data)
    output_path = graph.render("images/"+filename, view=True, format="png", cleanup=True)  # saves as 'tree.png' and opens it
    print(f"Decision tree saved to {output_path}")
    return output_path

# Part 2: Decision Tree Implementation and Visualization
print("\nDECISION TREE IMPLEMENTATION")

# Fit a basic decision tree
print("\nFitting basic decision tree (max_depth=3)...")
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
tree.fit(X_train, t_train)

print("Training Accuracy:", tree.score(X_train, t_train))
print("Validation Accuracy:", tree.score(X_valid, t_valid))

# Visualize the tree
visualize_tree(tree, filename="basic_tree")

# Classify a specific example
print("\nClassifying X_train[5] by hand:")
print(dict(zip(feature_names, X_train[5])))
"""
traverse to the left as age=44<=64.5 and then right since chest pain=1 and not
<=0.5 and then left again as age=44<=56.5. thus the resulting node is:
entropy = 0.991, samples = 448, value = [249, 199], class = heart_no
"""

# Underfitting example
print("\nUNDERFITTING EXAMPLE")

tree2 = DecisionTreeClassifier(criterion="entropy", max_depth=1)
tree2.fit(X_train, t_train)

print("Training Accuracy (underfit):", tree2.score(X_train, t_train))
print("Validation Accuracy (underfit):", tree2.score(X_valid, t_valid))
visualize_tree(tree2, filename="underfit_tree")

# Overfitting example
print("\nOVERFITTING EXAMPLE")

tree3 = DecisionTreeClassifier(criterion="entropy", max_depth=25)
tree3.fit(X_train, t_train)

print("Training Accuracy (overfit):", tree3.score(X_train, t_train))
print("Validation Accuracy (overfit):", tree3.score(X_valid, t_valid))
visualize_tree(tree3, max_depth=5, filename="overfit_tree") # visualize only top 5 levels

# min_samples_split examples
print("\nMIN_SAMPLES_SPLIT EXAMPLES")

# Underfitting with min_samples_split
tree4 = DecisionTreeClassifier(criterion="entropy", min_samples_split=2000)
tree4.fit(X_train, t_train)

print("Training Accuracy (min_samples_split=2000):", tree4.score(X_train, t_train))
print("Validation Accuracy (min_samples_split=2000):", tree4.score(X_valid, t_valid))
visualize_tree(tree4, max_depth=5, filename="min_samples_split_2000")

# Overfitting with min_samples_split
tree5 = DecisionTreeClassifier(criterion="entropy", min_samples_split=2)
tree5.fit(X_train, t_train)

print("Training Accuracy (min_samples_split=2):", tree5.score(X_train, t_train))
print("Validation Accuracy (min_samples_split=2):", tree5.score(X_valid, t_valid))
visualize_tree(tree5, max_depth=5, filename="min_samples_split_2")
"""
2 is the lowest value that can be used for min_samples_split and so is used to
make the graph overfit as it would split even for only 2 sample points, leading 
to a deep and complex tree. A large value of 2000 makes it underfit as it is the 
8000 divided by only 4.
"""
# Note: overfitting has high training accuracy (memorizes training data)

# Part 3: Hyperparameter Tuning
print("\nHYPERPARAMETER TUNING")

def build_all_models(max_depths, 
                    min_samples_split, 
                    criterion, 
                    X_train=X_train, 
                    t_train=t_train, 
                    X_valid=X_valid, 
                    t_valid=t_valid):
    """
    Parameters:
        `max_depths` - A list of values representing the max_depth values to be
                       try as hyperparameter values
        `min_samples_split` - A list of values representing the min_samples_split
                       values to try as hyperparameter values
        `criterion` -  A string; either "entropy" or "gini"

    Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
    the training and validation accuracies (via the `score()` method).
    In other words, out[(max_depth, min_samples_split)]['val'] = validation score and
                    out[(max_depth, min_samples_split)]['train'] = training score
    For that combination of (max_depth, min_samples_split) hyperparameters.
    """
    out = {}

    for d in max_depths:
        for s in min_samples_split:
            out[(d, s)] = {}
            # Create a DecisionTreeClassifier based on the given hyperparameters and fit it to the data
            tree = DecisionTreeClassifier(max_depth=d,
                                          min_samples_split=s,
                                          criterion=criterion,
                                          random_state=42)
            tree.fit(X_train, t_train)

            # store the validation and training scores in the `out` dictionary
            out[(d, s)]['train'] = tree.score(X_train, t_train)
            out[(d, s)]['val'] = tree.score(X_valid, t_valid)

    return out

# Hyperparameters values to try in our grid search
criterions = ["entropy", "gini"]
max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

best_hyperparameters = {}

# Iterate over each criterion
for criterion in criterions:
    print(f"\nUsing criterion {criterion}")
    
    res = build_all_models(max_depths, min_samples_split, criterion)
    
    best_train_score = 0
    best_val_score = 0
    best_hyperparameters[criterion] = (None, None)

    # Search for the optimal (max_depth, min_samples_split) given this criterion
    for (d, s), scores in res.items():
        train_score = scores['train']
        val_score = scores['val']
        
        if val_score > best_val_score:
            best_train_score = train_score
            best_val_score = val_score
            best_hyperparameters[criterion] = (d, s)
    
    print(f"Best Hyperparameters for {criterion}: Max Depth = {best_hyperparameters[criterion][0]}, Min Samples Split = {best_hyperparameters[criterion][1]}")
    print(f"Best Validation Score: {best_val_score:.4f}")
    print(f"Corresponding Training Score: {best_train_score:.4f}")

# Part 4: Test Accuracy
print("\nTEST ACCURACY")

# Use the best hyperparameters for entropy criterion
best_tree = DecisionTreeClassifier(
    max_depth=best_hyperparameters['entropy'][0],
    min_samples_split=best_hyperparameters['entropy'][1],
    criterion='entropy')
best_tree.fit(X_train, t_train)

# Report the test accuracy
test_accuracy_entropy = best_tree.score(X_test, t_test)
print("Test Accuracy (ENTROPY):", test_accuracy_entropy)

# Also test with gini criterion
best_tree_gini = DecisionTreeClassifier(
    max_depth=best_hyperparameters['gini'][0],
    min_samples_split=best_hyperparameters['gini'][1],
    criterion='gini')
best_tree_gini.fit(X_train, t_train)

# Report the test accuracy
test_accuracy_gini = best_tree_gini.score(X_test, t_test)
print("Test Accuracy (GINI):", test_accuracy_gini)