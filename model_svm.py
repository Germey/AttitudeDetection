from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import json
import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

file = '20180917.json'

items = json.loads(open(file, encoding='utf-8').read())

print(len(items))

# items = list(filter(lambda x: np.mean(x['pose_std']) and x['expression_mean'], items))
items = list(filter(lambda x: x['expressions'] and x['poses'], items))
# items = list()
print(len(items))

# print(items[0])
s = StandardScaler()

x_data_poses = list(map(lambda x: x['pose_std'], items))
print(x_data_poses[0])

x_data_poses = s.fit_transform(x_data_poses)
print(x_data_poses[0])

x_data, y_data = [], []

x_data_expressions = list(map(lambda x: x['expression_mean'], items))
print(x_data_expressions[0])

x_data_expressions = s.fit_transform(x_data_expressions)
print(x_data_expressions[0])

x_data_srs = list(map(lambda x: x['sr'], items))

for i1, i2, i3, item in zip(x_data_poses, x_data_expressions, x_data_srs, items):
    x_data.append(list(itertools.chain(i1, i2, i3)))
    y_data.append(item['label'])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=20)

lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)
print('Accuracy:', lsvc.score(x_test, y_test))
print('Report', classification_report(y_test, y_predict, target_names=['no', 'yes']))
