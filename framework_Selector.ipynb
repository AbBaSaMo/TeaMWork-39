# TeaMWork-39

#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[126]:


data = pd.read_excel('Training data.xlsx')
data = data.drop(data.index[-13:])
#the above just gets rid of the companies that have yet to be declared to a framework

drop_columns = ['AnnualTurnover', 'CyberSecurityBudget', 'Staff', 'Min Risk Impact', 'Max Risk Impact', 'Regions you operate in ']
#non-cateogrical columns are dropped so they are not encoded
categorical_data = data.copy()
for column in drop_columns:
    del categorical_data[column]
    #non-categorical data is temporarily deleted to be added back later so it can be encoded seperately


# In[127]:


encoder = LabelEncoder()
for feature in categorical_data:
    categorical_data[feature] = encoder.fit_transform(data[feature])
    #encodes the categorical data to a numerical format
    
encoded_data = categorical_data.copy()
for column in drop_columns:
    encoded_data[column] = data[column]
    #adds back numerical data

feature_var = encoded_data.drop('Primary recommendation', axis=1)
#the features that will decide the frameworks are seperated from the target variable (the framework)
feature_var = feature_var.drop('Name', axis= 1)
target_var = encoded_data['Primary recommendation']


# In[128]:


feature_train, feature_test, target_train, target_test = train_test_split(feature_var, target_var, test_size = 0.3, random_state = 42)

#the training and testing sets are created using the library


# In[129]:


model = DecisionTreeClassifier()
model.fit(feature_train, target_train)
#the model is trained on the data

prediction = model.predict(feature_test)

#decoded_prediction = encoder.inverse_transform(prediction)
#for pred in decoded_prediction:
#    print (prediction)
#this decoder above seems like it should work but does not decode as expected, ill sort it.
    
prediction


# In[130]:


accuracy = accuracy_score(target_test, prediction)
print (accuracy)
# :( #


# In[ ]:





# In[ ]:



