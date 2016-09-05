import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.utils import shuffle
np.random.seed(0)
n_estimators = 30
n_classes = 3
plot_colors = ['r', 'b', 'y']
cmap = plt.cm.RdYlBu

clf = RF(n_estimators, class_weight = 'balanced_subsample', n_jobs=-1)
x_train = np.load('X_train.npy')[:80000,:]
y_train = np.load('y_train.npy')[:80000]
y_train[np.where(y_train == 3)] = 2
y_train[np.where(y_train == 1)] = 3
y_train[np.where(y_train == 2)] = 1
y_train[np.where(y_train == 3)] = 2

x_test = np.load('X_valid.npy')
y_test = np.load('y_valid.npy')
y_test[np.where(y_test == 3)] = 2
y_test[np.where(y_test == 1)] = 3
y_test[np.where(y_test == 2)] = 1
y_test[np.where(y_test == 3)] = 2

x_test, y_test = shuffle(x_test, y_test)

jh_test = x_test[:,2]
hk_test = x_test[:,3]
y_test = y_test

jh_train = x_train[:,2]
hk_train = x_train[:,3]

jh_hk_train = np.vstack((jh_train, hk_train)).T
jh_hk_test = np.vstack((jh_test, hk_test)).T

hk_min, hk_max = hk_test.min() - 0.1, hk_test.max() + 0.1
jh_min, jh_max = jh_test.min() - 0.1, jh_test.max() + 0.1

hk_grid, jh_grid = np.meshgrid(np.arange(hk_min, hk_max, 0.02), np.arange(jh_min, jh_max, 0.02))

estimator_alpha = 1.0/n_estimators

clf.fit(jh_hk_train, y_train)
for tree in clf.estimators_:

    Z = tree.predict(np.c_[jh_grid.ravel(), hk_grid.ravel()])
    Z = Z.reshape(hk_grid.shape)
    cs = plt.contourf(hk_grid, jh_grid, Z, alpha=estimator_alpha,cmap=cmap)

class_labels = ['Red Dwarf', 'Other', 'Red Giant',]
reverse_classes = [1, 2, 0]
reverse_colors = plot_colors[::-1]
for i,c in zip(reverse_classes, reverse_colors):
    idx = np.where(y_test == i)
    plt.scatter(jh_hk_test[idx,1], jh_hk_test[idx,0], c=c, label = class_labels[i], cmap=cmap)

plt.title('Random Forest with 100 trees')
plt.xlabel('H - K')
plt.ylabel('J - H')
plt.legend(loc='best')
plt.xlim(hk_min, hk_max)
plt.ylim(jh_min, jh_max)
plt.axis('tight')
plt.savefig('decision_surface_100.png')
plt.show()
