from sklearn.model_selection import train_test_split
from custom_estimator import custom_estimator
from data import triple_xor

x, y = triple_xor(10000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
ce = custom_estimator()
ce.fit(x, y)

print('Mean accuracy: ', ce.score(x_test, y_test))
print('Selected threshold: ', ce._threshold_binarizer._threshold)