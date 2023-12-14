import numpy as np 


l = [[["h", 1], ["b", 2]], [["c", 1], ["d", 2]], [["c", 1], ["d", 2]]]

l = np.asarray(l)
print(l.shape)
print(l.reshape(l.shape[0] * 2, 2))