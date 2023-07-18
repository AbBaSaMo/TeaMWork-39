import openpyxl as openpyxl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

frameworks = {
    'APAC APS CPS234': 1,
    'APAC Malaysia PDPA': 2,
    'COBIT 2019': 3,
    'EU GDPR': 4,
    'ISO 27001 v2022': 5,
    'ISO 27002 v2022': 6,
    'NIST 800-53 rev 4': 7,
    'PCI DSS v4': 8
}

pd.set_option('display.max_columns', None)

# READ TRAINING AND TEST DATA
training_data = pd.read_excel('training_data.xlsx')
testing_data = pd.read_excel('testing_data.xlsx')

training_data_X = training_data.drop('Primary recommendation', axis=1).drop('Name', axis=1)
training_data_Y = training_data['Primary recommendation']

# PRE-PROCESSING OF TRAINING AND TEST DATA
enc_training_data_X = training_data_X.copy()
enc_training_data_Y = training_data_Y.copy()

# group columns by the pre-processing that must be done
categorical_columns = ['IndustryName', 'Microsoft', 'Country']
ranged_columns = ['AnnualTurnover', 'CyberSecurityBudget', 'Staff', 'RiskImpactLevel']  # average, scale or categorise
set_columns = ['BusinessRoles', 'InformationHoldings', 'SecurityTargets']


# convert all nominal-categorical data using label encoding
enc_training_data_Y = LabelEncoder().fit_transform(enc_training_data_Y)
for column in categorical_columns:
    enc_training_data_X[column] = LabelEncoder().fit_transform(enc_training_data_X[column])

# convert all set type data using one-hot encoding
for column in set_columns:
    # convert the comma delimited string of items to a list i.e. 'A, B, C' -> ['A', 'B', 'C']
    enc_training_data_X[column] = enc_training_data_X[column].str.split(',')

    # convert the list to binary format
    enc_data = pd.get_dummies(
        enc_training_data_X[column]
        .apply(pd.Series)
        .stack()
    ).groupby(level=0).sum()

    # add this data to the training data [each item in the list is added as a column] and remove the old format
    enc_training_data_X = pd.concat([enc_training_data_X, enc_data], axis=1)
    enc_training_data_X = enc_training_data_X.drop(column, axis=1)

# convert all ranged type data
for column in ranged_columns:
    enc_training_data_X[column] = LabelEncoder().fit_transform(enc_training_data_X[column])

print(enc_training_data_X)

# TRAIN AND TEST THE MODEL
model = DecisionTreeClassifier()
model.fit(enc_training_data_X, enc_training_data_Y)

prediction = model.predict(feature_test)
decoded_prediction = encoder.inverse_transform(prediction)
# for pred in decoded_prediction:

# # todo verify model with the testing data
# print(prediction)
# accuracy = accuracy_score(target_test, prediction)
# print(accuracy)