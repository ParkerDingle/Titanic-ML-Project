# imports
# Data imports
import pandas as pd
import numpy as np

# allows for creation of a training set and a testing set
from sklearn.model_selection import train_test_split, GridSearchCV

# Cleans the data for the model & converts everything to a number range
from sklearn.preprocessing import MinMaxScaler

# this is the model being used
from sklearn.neighbors import KNeighborsClassifier

# test how well the metrics are working
from sklearn.metrics import accuracy_score, confusion_matrix

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Data Frame
data = pd.read_csv("C:/Users/paddi/OneDrive/Desktop/HTML/Titanic_Data.csv")
data.info()  # says what type of data im working with
print(data.isnull().sum())  # see how many missing values there are

# data cleaning
def preprocess_data(df):
    # Drops irrelevent columns
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, errors="ignore")

    fill_missing_ages(df)  # Fill in missing ages

    # fill Fare if missing (needed for bins)
    if "Fare" in df:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # convert gender into integer
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).astype("int64")

    # encode Embarked if present  << added to remove strings >>
    if "Embarked" in df:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype("int64")

    # ensure integer types where appropriate  << added for consistency >>
    for col in ["Pclass", "SibSp", "Parch"]:
        if col in df:
            df[col] = df[col].astype("int64")

    # creating new column in data
    df["FamilySize"] = df["SibSp"] + df["Parch"]  # Finds family size by combining siblings and parch
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)  # If family size is equal to 0 give it a 1 otherwise keep it 0
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")  # Price of ticket might impact chance of survival
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)  # creates categories for ages to go into

    return df

# Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    # checks if any ages are missing
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)

# call function
data = preprocess_data(data)

# answer you can use to check against model
X = data.drop(columns=["Survived"])
Y = data["Survived"]

# safety: one-hot any leftover object columns (if any)  << added safeguard >>
obj_cols = X.select_dtypes(include="object").columns
if len(obj_cols):
    X = pd.get_dummies(X, columns=obj_cols)

# helps train model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Data processing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hypertuning
def tune_Model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],  # different ways to check the data
        "weights": ["uniform", "distance"]
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)  # Trains the data
    return grid_search.best_estimator_

# This is the model
best_model = tune_Model(X_train, y_train)

# predictions and evaluate
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion matrix:\n{matrix}')
print(matrix)

# plot
def plot_model(matrix):
    plt.figure(figsize = (10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived", "Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    plt.show()

plot_model(matrix)
