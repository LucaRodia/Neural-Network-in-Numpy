import numpy as np

y = np.array([0, 1, 1, 2])
z = np.array([2, 3, 0, 2])

acc = np.mean(y == z)
print(acc)