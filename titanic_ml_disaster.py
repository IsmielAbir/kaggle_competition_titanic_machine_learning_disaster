import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


test = pd.read_csv('test.csv')
test.head()

train = pd.read_csv('train.csv')
train.head()


concatenated_data = pd.concat([train, test])

sns.boxplot(x='Pclass', y='Age', data=concatenated_data)



features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

X

y = train["Survived"]
y



sns.heatmap(concatenated_data.corr())



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=109, max_depth=5, random_state=2)
model.fit(X, y)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)