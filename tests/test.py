import numpy as np

#np.random.seed(1)
x = np.random.choice(9, 8, replace = False).reshape((4, 2))
print(x)
print(x[:,1])
print(np.max(x[:,1]))
print(np.argmax(x[:,1]))
print(x[np.argmax(x[:,1])])
print(x[np.argmax(x[:,1])][0])
