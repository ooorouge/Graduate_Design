from All_in_1 import fromcsv
from All_in_1 import Training_Config1
import numpy as np
#zonggong 123
# x 0 0 1
#       ^ index = 122 length=123
x, xs, ys = fromcsv(length=123, size=200)

train_indices = np.random.choice(len(x), round(len(x)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x))) - set(train_indices)))

test_actuals = [np.argmax(x) for x in ys[test_indices]]
test_preds = [np.argmax(x) for x in ys[test_indices]]

test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])

for i in range(len(test_actuals)):
    print(test_actuals[i], end=' and preds is ')
    print(test_preds[i])
print('total acc = {}%'.format(test_acc*100))