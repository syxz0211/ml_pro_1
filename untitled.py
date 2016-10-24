import numpy as np
import pandas as pd
from sklearn import svm
##对非数字特征赋值,暂定为1,2,3,4,5
data = pd.read_csv('car_clean.csv', header=None)
##随机地取出1000用作训练集,其余作为测试机
data_train = data.sample(n=1000)
data_test = data.ix[~data.index.isin(data_train.index)]
X_train = np.array(data_train)[:, 0:6]
X_test = np.array(data_test)[:, 0:6]
y_train = data_train[6]
y_test = data_test[6]
clf = svm.SVC()
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
##输出测试的正确率
accuracy = len([i for i in (pre-y_test) if i==0])/len(X_test)
