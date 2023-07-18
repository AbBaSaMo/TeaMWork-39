import openpyxl as openpyxl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score


def preprocess_data(data):
    # group columns by the pre-processing that must be done
    categorical_columns = ['IndustryName', 'Microsoft', 'Country']
    ranged_columns = ['AnnualTurnover', 'CyberSecurityBudget', 'Staff', 'RiskImpactLevel']
    set_columns = ['BusinessRoles', 'InformationHoldings', 'SecurityTargets']

    # convert all nominal-categorical data using label encoding
    for column in categorical_columns:
        data[column] = LabelEncoder().fit_transform(data[column])

    # convert all set type data using one-hot encoding
    for column in set_columns:
        # convert the comma delimited string of items to a list i.e. 'A, B, C' -> ['A', 'B', 'C']
        data[column] = data[column].str.split(',')

        # convert the list to binary format
        enc_data = pd.get_dummies(
            data[column]
            .apply(pd.Series)
            .stack()
        ).groupby(level=0).sum()

        # add this data to the training data [each item in the list is added as a column] and remove the old format
        data = pd.concat([data, enc_data], axis=1)
        data = data.drop(column, axis=1)

    # convert all ranged type data
    for column in ranged_columns:
        data[column] = LabelEncoder().fit_transform(data[column])

    return data


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

company_names = testing_data['Name']
testing_data = testing_data.drop('Name', axis=1)

# PRE-PROCESSING OF TRAINING AND TEST DATA
label_encoder = LabelEncoder()
enc_training_data_Y = label_encoder.fit_transform(training_data_Y.copy())
enc_training_data_X = preprocess_data(training_data_X.copy())
enc_testing_data = preprocess_data(testing_data.copy())

# TRAIN AND TEST THE MODEL
model = DecisionTreeClassifier()
model.fit(enc_training_data_X, enc_training_data_Y)

prediction = model.predict(enc_testing_data)
prediction = label_encoder.inverse_transform(prediction)

for framework, company in zip(prediction, company_names):
    print(company + ' -> ' + framework)

# verify model with the testing data
# print(prediction)
# accuracy = accuracy_score(target_test, prediction)
# print(accuracy)