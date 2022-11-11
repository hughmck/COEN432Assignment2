import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score

# Reading in dataset and setting column names
colnames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital_status",
            "occupation" , "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income"]
df = pd.read_csv("adult.data", header=None)
df.columns = colnames

# Seperating the columns that have ints as column values and not words (ie. object datatype)
df_nonCategories = df.select_dtypes(exclude='object')
## Change label to binary; 1 if income greater than 50K; 0 if income lessthan or equal to 50K
df["income"] = np.where(df["income"].str.contains(">50K"), 1, 0)

df_Categories = df.select_dtypes(include='object')
cols = df_nonCategories.columns

# For binary classification, change categorical to binary values (ie. One Hot Encoding)
df = pd.get_dummies(df)

# Getting X and y from input dataset
y = df['income'].values
df.pop("income")
X = df.values

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Normalizing the data
scale = preprocessing.StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# Further split the training data into 80% training and 20% validation
val_split = round(len(df)*0.2)
x_train_tr, y_train_tr = X_train[:val_split], y_train[:val_split]
x_train_va, y_train_va = X_train[val_split:], y_train[val_split:]

# Finding best hyperparameter K for K-Nearest Neighbours (KNN)
model_choices = []
valid_acc = []

for k in range(1, 15):
    knn = KNN(k)
    knn.fit(x_train_tr, y_train_tr)
    train_sc = knn.score(x_train_tr, y_train_tr)
    test_sc = np.mean(cross_val_score(knn, x_train_tr, y_train_tr, cv=10))
    model_choices.append(k)
    valid_acc.append(test_sc)
    # print("K value  : ", k,
    #       ", Train Score:", train_sc,
    #       ", Test Score : ", test_sc)

# use the best K to predict test data
best_valid_K = model_choices[valid_acc.index(max(valid_acc))]
knn = KNN(n_neighbors=best_valid_K)
knn.fit(X_train, y_train)

# Predictions
yh_train = knn.predict(X_train)
yh_test = knn.predict(X_test)

# Training Accuracy
acc_train = np.mean(yh_train == y_train)
acc_test = np.mean(yh_test == y_test)
print(f'best K = {best_valid_K}, Train accuracy = {acc_train}, Test accuracy = {acc_test}')


if __name__ == '__main__':
    print('Hi')


