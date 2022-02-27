from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from dython import nominal
import sweetviz
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

# before operations

dataset = pd.read_csv(
    "C:/Users/asude/Downloads/archive(1)/hrTrain.csv", na_values=[np.nan])

dataset = dataset.drop(
    ['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'], axis=1)

# relationship between both categorical and numerical features
nominal.associations(dataset, figsize=(30, 15), mark_columns=True)

"""
# get dataset analyze report
report = sweetviz.analyze(dataset, "Attrition")
report.show_html()
"""
###############################################################################

# encoding categorical features
mapping = {"BusinessTravel": {"Non-Travel": 1,
                              "Travel_Rarely": 2, "Travel_Frequently": 3},
           "Attrition": {"Yes": 1, "No": 0}}
dataset = dataset.replace(mapping)

# range normalization for MonthlyIncome
dataset.MonthlyIncome = (dataset.MonthlyIncome - dataset.MonthlyIncome.min()) / \
    (dataset.MonthlyIncome.max() - dataset.MonthlyIncome.min())

# check for missing values and duplicates
print(dataset.isnull().sum()/len(dataset))
print(dataset.nunique(axis=0))
sn.heatmap(pd.DataFrame(dataset.isnull().sum()/len(dataset), columns=['data']))
plt.show()

duplicate = dataset[dataset.duplicated()]

###############################################################################

# get categorical and numerical features
categorical = dataset.select_dtypes(include='object')

numerical = dataset.select_dtypes(include=['float64', 'int64'])
discrete = dataset.select_dtypes(include=['float64', 'int64'])
continuous = dataset.select_dtypes(include=['float64', 'int64'])

numeric_discrete = ['Attrition', 'BusinessTravel', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

for ftr in list(numeric_discrete):
    continuous = continuous.drop([ftr], axis=1)

numeric_continuous = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                      'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
                      'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

for ftr in list(numeric_continuous):
    discrete = discrete.drop([ftr], axis=1)


# find outliers and replace with borders
def find_outliers(feature):
    q1, q3 = np.percentile(feature, [25, 75])
    iqr = q3 - q1
    min_thresold = q1 - (iqr * 1.5)
    max_thresold = q3 + (iqr * 1.5)
    feature.loc[feature > max_thresold] = max_thresold
    feature.loc[feature < min_thresold] = min_thresold


for ftr in list(continuous):
    find_outliers(dataset[ftr])

###############################################################################

# graphics for continuous features
for index in numeric_continuous:
    sn.displot(continuous[index])
plt.show()

# graphics for continuous features with in one graphic
fig, ax = plt.subplots(7, 2, figsize=(9, 15))
sn.distplot(dataset['Age'], ax=ax[0, 0])
sn.distplot(dataset['DailyRate'], ax=ax[0, 1])
sn.distplot(dataset['DistanceFromHome'], ax=ax[1, 0])
sn.distplot(dataset['HourlyRate'], ax=ax[1, 1])
sn.distplot(dataset['MonthlyIncome'], ax=ax[2, 0])
sn.distplot(dataset['MonthlyRate'], ax=ax[2, 1])
sn.distplot(dataset['NumCompaniesWorked'], ax=ax[3, 0])
sn.distplot(dataset['PercentSalaryHike'], ax=ax[3, 1])
sn.distplot(dataset['TotalWorkingYears'], ax=ax[4, 0])
sn.distplot(dataset['TrainingTimesLastYear'], ax=ax[4, 1])
sn.distplot(dataset['YearsAtCompany'], ax=ax[5, 0])
sn.distplot(dataset['YearsInCurrentRole'], ax=ax[5, 1])
sn.distplot(dataset['YearsSinceLastPromotion'], ax=ax[6, 0])
sn.distplot(dataset['YearsWithCurrManager'], ax=ax[6, 1])
plt.tight_layout()
plt.show()

# graphics for categorical features
categorical_index_list = categorical.columns.values.tolist()

for index in categorical_index_list:
    sn.factorplot(data=categorical, kind='count', aspect=3, size=5, x=index)

for index in numeric_discrete:
    sn.factorplot(data=discrete, kind='count', aspect=3, size=5, x=index)

# box plot and histograms for continuous features
for var in numeric_continuous:
    # boxplot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    fig = dataset.boxplot(column=var)
    fig.set_ylabel(var)

    # histogram
    plt.subplot(1, 2, 2)
    fig = dataset[var].hist(bins=20)
    fig.set_ylabel('No. of Employees')
    fig.set_xlabel(var)
    plt.grid(False)
    plt.show()

# drop Attrition feature for train models
numerical = numerical.drop('Attrition', axis=1)
# define target feature
target = dataset['Attrition']
# define random_state for testing
random_state = 4


#more accurate when drop some features
#founded by testing
numerical = numerical.drop('MonthlyRate', axis=1)
numerical = numerical.drop('PercentSalaryHike', axis=1)
numerical = numerical.drop('PerformanceRating', axis=1)
numerical = numerical.drop('RelationshipSatisfaction', axis=1)


# get every options for categorical features as a new feature
dataset_cat = pd.get_dummies(categorical)
# finally merge all categorical features and numeric
dataset_final = pd.concat([numerical, dataset_cat], axis=1)

# Test technic number 2 --> %90 train, %10 test
x_train, x_test, y_train, y_test = train_test_split(
    dataset_final, target, test_size=0.1, random_state=random_state)


# oversampling for attrition(increase yes count)
oversampler = SMOTE(random_state=random_state, sampling_strategy=1.0)
smote_train, smote_target = oversampler.fit_resample(x_train, y_train)

# evaluation results


def evaluation_result(model_predictions):
    print()
    print("Accuracy: ", accuracy_score(y_test, model_predictions))
    print(classification_report(y_test, model_predictions))
    mae = metrics.mean_absolute_error(y_test, model_predictions)
    mse = metrics.mean_squared_error(y_test, model_predictions)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, model_predictions)
    print("Results of sklearn.metrics:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)

# model trained with number2 test technic

def model_implement(model):
    model.fit(smote_train, smote_target)
    model_predictions = model.predict(x_test)
    evaluation_result(model_predictions)

# Random Forest
model = RandomForestClassifier()
model_implement(model)

# Logistic Regression
model = LogisticRegression()
model_implement(model)

# Naive Bayes
model = GaussianNB()
model_implement(model)

###############################################################################
# Test technic number 1 --> n-fold cross validation (n=10)
cv = KFold(n_splits=10, random_state=random_state, shuffle=True)
#evaluate metric results
def scores():
    scores = cross_validate(model, smote_train, smote_target,
                             scoring=scoring, cv=cv, n_jobs=-1)
    print()
    print(scores.keys())
    print(scores['test_acc'])
    print()

scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'roc_auc': 'roc_auc',
           'f1_macro': 'f1_macro',
           'mae': 'neg_mean_absolute_error',
           'mse': 'neg_mean_squared_error',
           'r2':'r2'}
model = LogisticRegression(solver='liblinear')
scores()

# Base model
model = RandomForestClassifier(n_estimators=100, bootstrap=False)
scores()
"""
# find best parameters for random forest
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                               n_iter=100, cv=3, verbose=2, random_state=random_state, n_jobs=-1)
# Fit the random search model
rf_random.fit(smote_train, smote_target)
rf_random.best_params_
best_random = rf_random.best_estimator_
model = best_random
scores()
"""
model = GaussianNB()
scores()

# examples of plots
"""
#box plot example
dataset.TotalWorkingYears.plot(kind ='box')
plt.show()

#box plot example
dataset.YearsAtCompany.plot(kind ='box')
plt.show()
"""

"""
#scatter plot example
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(dataset.MonthlyRate, dataset.TotalWorkingYears)
ax.set_xlabel('MonthlyRate')
ax.set_ylabel('TotalWorkingYears')
plt.show()
"""
###############################################################################
# after operations
# analyze correlation after data pre-processing
nominal.associations(dataset, figsize=(30, 15), mark_columns=True)
