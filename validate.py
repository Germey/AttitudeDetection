from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from random import random, randrange, uniform

total = 10000

x_data = []
y_data = []
for i in range(5, total):
    x_data.append([i, uniform(0, 5), uniform(0, 5)])
    y_data.append(1)

for i in range(5, total):
    x_data.append([uniform(0, 5), i, uniform(0, 5)])
    y_data.append(0)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=20)

lsvc = LogisticRegression()
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)
print('Accuracy:', lsvc.score(x_test, y_test))
print('Report', classification_report(y_test, y_predict, target_names=['no', 'yes']))
