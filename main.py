import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, roc_curve, roc_auc_score, classification_report

def PreProcessing():

    # Reading in dataset and setting column names
    colnames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital_status",
                "occupation" , "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "native_country", "income"]
    df = pd.read_csv("adult.data", header=None)
    df.columns = colnames

    # Separating the columns that have ints as column values and not words (ie. object datatype)
    # df_nonCategories = df.select_dtypes(exclude='object')

    # Change label to binary; 1 if income greater than 50K; 0 if income lessthan or equal to 50K
    df["income"] = np.where(df["income"].str.contains(">50K"), 1, 0)

    # df_Categories = df.select_dtypes(include='object')
    # cols = df_nonCategories.columns

    # For binary classification, change categorical to binary values (ie. One Hot Encoding)
    df = pd.get_dummies(df)

    # Getting X and y from input dataset
    y = df['income'].values
    df.pop("income")
    X = df.values

    # Splitting data into training 90% and testing set 10%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    # Normalizing the data
    scale = preprocessing.StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    return (df, X_train, X_test, y_train, y_test)

def KNNEval(df,X_train, X_test, y_train, y_test):
    # Further split the training data into 80% training and 20% validation
    val_split = round(len(df)*0.2)
    x_train_tr, y_train_tr = X_train[:val_split], y_train[:val_split]

    # Finding best hyperparameter K for K-Nearest Neighbours (KNN)
    model_choices = []
    valid_acc = []

    for k in range(1, 15):
        knn = KNN(k)
        knn.fit(x_train_tr, y_train_tr)
        test_sc = np.mean(cross_val_score(knn, x_train_tr, y_train_tr, cv=10))
        model_choices.append(k)
        valid_acc.append(test_sc)

    # use the best K to predict test data
    best_valid_K = model_choices[valid_acc.index(max(valid_acc))]
    knn = KNN(n_neighbors=best_valid_K)
    knn.fit(X_train, y_train)

    # Predictions
    yh_train = knn.predict(X_train) # Train prediction of income
    yh_test = knn.predict(X_test) # Test prediction of income

    # Training Accuracy
    acc_train = np.mean(yh_train == y_train)
    acc_test = np.mean(yh_test == y_test)

    result = f'best K = {best_valid_K}, Train accuracy = {acc_train}, Test accuracy = {acc_test}'

    # Confusion Matrix
    metrics.accuracy_score(y_test, yh_test)
    confusion_matrix(y_test, yh_test)
    plot_confusion_matrix(knn, X_test, y_test)

    plt.savefig("ConfusionMatrix.png",  dpi=300) #bbox_inches='tight',
    plt.show()

    # Classification Report
    # 0: <=50K
    # 1: >50K
    cr = classification_report(y_test, yh_test)

    return result, cr

def ROC(X_train,y_train, X_test,y_test):

    models = [KNN(),
              DTC(),
              GaussianNB()]

    perf = {}
    # Get ROC curves for different models to compare
    for model in models:
        fit = model.fit(X_train, y_train)
        y_test_prob = fit.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        auroc = roc_auc_score(y_test, y_test_prob)
        perf[type(model).__name__] = {'fpr': fpr, 'tpr': tpr, 'auroc': auroc}

    plt.clf()
    i = 0
    for model_name, model_perf in perf.items():
        plt.plot(model_perf['fpr'], model_perf['tpr'], label=model_name)
        plt.text(0.4, i + 0.1, model_name + ': AUC = ' + str(round(model_perf['auroc'], 2)))
        i += 0.1

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC in predicting Income')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="upper left")
    plt.show()
    plt.savefig("roc_curve.png", dpi=300)# bbox_inches='tight'
    plt.show()

if __name__ == '__main__':
    df, X_train, X_test, y_train, y_test = PreProcessing()
    result, cr = KNNEval(df, X_train, X_test, y_train, y_test)
    print(result)
    print(cr)
    ROC(X_train, y_train, X_test, y_test)

