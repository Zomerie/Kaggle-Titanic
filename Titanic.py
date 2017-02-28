# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:19:13 2017

@author: ZMercado
"""
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_rows = 100

data = pd.read_csv('train.csv')
data.head()
desc=data.describe()
'''print (desc)'''

'''processed or not?'''
def status(feature):
    print 'Processing',feature,': ok'
    

grouped = data.groupby(['Sex','Pclass'])
group_stat=grouped.median()
print (group_stat)

def process_age(call_data):
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            return 35
        elif row['Sex']=='female' and row['Pclass'] == 2:
            return 28
        elif row['Sex']=='female' and row['Pclass'] == 3:
            return 21.5
        elif row['Sex']=='male' and row['Pclass'] == 1:
            return 40
        elif row['Sex']=='male' and row['Pclass'] == 2:
            return 30
        elif row['Sex']=='male' and row['Pclass'] == 3:
            return 25
        
    call_data.Age = call_data.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')
    
process_age(data)
desc=data.describe()
print (desc)

def process_sex(call_data):
    
    call_data['Sex'] = call_data['Sex'].map({'male':1,'female':0})
    
    status('sex')
process_sex(data)

'''missing values'''
data.Fare.fillna(data.Fare.mean(),inplace=True)
data.drop('Cabin',axis=1,inplace=True)
data.drop('Embarked',axis=1,inplace=True)
data.drop('Name',axis=1,inplace=True)
data.drop('Ticket',axis=1,inplace=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)

info_data=data.info()
print (info_data)
targets = data.Survived

train=data
train.drop('Survived',axis=1,inplace=True)

forest = RandomForestClassifier(max_features='auto')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

test = pd.read_csv('test.csv')
test_new =  pd.read_csv('test.csv')
process_age(test_new)
process_sex(test_new)
'''missing values'''
test_new.Fare.fillna(data.Fare.mean(),inplace=True)
test_new.drop('Cabin',axis=1,inplace=True)
test_new.drop('Embarked',axis=1,inplace=True)
test_new.drop('Name',axis=1,inplace=True)
test_new.drop('Ticket',axis=1,inplace=True)

output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)
