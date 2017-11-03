import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
combine = pd.concat([train.drop('Survived',1),test])

#print(train.describe())
#print(train.isnull().sum())
#print(test.info())

surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]
surv_col = "blue"
nosurv_col = "red"


'''plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
            axlabel='Age')
plt.subplot(332)
sns.barplot('Sex', 'Survived', data=train)
plt.subplot(333)
sns.barplot('Pclass', 'Survived', data=train)
plt.subplot(334)
sns.barplot('Embarked', 'Survived', data=train)
plt.subplot(335)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(336)
sns.barplot('Parch', 'Survived', data=train)
plt.subplot(337)
sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color=surv_col)
sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()'''

#print("Median age survivors: %.1f, Median age non-survivers: %.1f"\ %(np.median(surv['Age'].dropna()), np.median(nosurv['Age'].dropna())))


#print("There are %i unique ticket numbers among the %i tickets." \%(train['Ticket'].nunique(),train['Ticket'].count()))

'''plt.figure(figsize=(10,8))
foo = sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)
plt.show()'''

'''tab = pd.crosstab(train['Embarked'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Port embarked')
dummy = plt.ylabel('Percentage')
plt.show()'''

'''CLEAN DATA'''
train['Title'] = train.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())
test['Title'] = test.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())


train['Title'] = train['Title'].replace(['Don','Capt','Sir','Dr'], 'Mr')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
train.Title.loc[(train.Title !=  'Master') & (train.Title !=  'Mr') & (train.Title !=  'Miss')
             & (train.Title !=  'Mrs')] = 'Others'

test['Title'] = test['Title'].replace(['Don','Capt','Sir','Dr'], 'Mr')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
test.Title.loc[(test.Title !=  'Master') & (test.Title !=  'Mr') & (test.Title !=  'Miss')
             & (test.Title !=  'Mrs')] = 'Others'

title_codes = {"Mr":0,"Mrs":1,"Master":2,"Miss":3,"Others":4}
train['Title'].replace(title_codes,inplace=True)
test['Title'].replace(title_codes,inplace=True)

train = train.drop('Name', 1)
test = test.drop('Name', 1)

sex_codes = {"female":1,"male":0}
train["Sex"].replace(sex_codes,inplace=True)
test["Sex"].replace(sex_codes,inplace=True)

#print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#print(test[pd.isnull(test['Fare'])])

median_fare = test.Fare.loc[(test.Pclass == 3) & (test.Embarked == 'S')].median()
test.Fare.fillna(median_fare, inplace=True)

#print(train.head())
#print(train.isnull().sum())
#print(test.isnull().sum())

train = train.drop('Cabin', 1)
test = test.drop('Cabin', 1)

#print(train.Embarked.isnull().sum(axis=0))
#print(train[pd.isnull(train['Embarked'])])
train.Embarked.fillna('S', inplace=True )

port_codes = {"S":2,"C":1,"Q":0}
train["Embarked"].replace(port_codes,inplace=True)
test["Embarked"].replace(port_codes,inplace=True)


#FILL NAN AGE
train_sub = train[['Age','Title','Fare','SibSp']]
X_train_age  = train_sub.dropna().drop('Age', axis=1)
y_train_age  = train['Age'].dropna()
X_test_age1 = train_sub.loc[np.isnan(train.Age)].drop('Age', axis=1)
rfr = RandomForestRegressor(n_estimators = 300)
rfr.fit(X_train_age, y_train_age)
y_train_pred_age = np.round(rfr.predict(X_test_age1),1)
train.Age.loc[train.Age.isnull()] = y_train_pred_age

test_sub = test[['Age','Title','Fare','SibSp']]
X_test_age  = test_sub.dropna().drop('Age', axis=1)
y_test_age  = test['Age'].dropna()
X_test_age2 = test_sub.loc[np.isnan(test.Age)].drop('Age', axis=1)
rfr = RandomForestRegressor(n_estimators = 300)
rfr.fit(X_test_age, y_test_age)
y_test_pred_age = np.round(rfr.predict(X_test_age2),1)
test.Age.loc[test.Age.isnull()] = y_test_pred_age

train = train.drop('Ticket', 1)
test = test.drop('Ticket', 1)
#print(train.isnull().sum())
#print(test.isnull().sum())


X_train, y_train = train.iloc[:, train.columns != 'Survived'].values, train.iloc[:, train.columns == 'Survived'].values.ravel()


'''KERAS NN MODEL'''
model = Sequential()

model.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 30, epochs = 400)

scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


'''
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, epochs=150, batch_size=10)

scores = model.evaluate(train_X, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))'''