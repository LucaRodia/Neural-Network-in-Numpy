# categorical cross entryopy
import math
import numpy as np

softmax_output = np.array([[0.7, 0.2, 0.1],  #valori inventati
                  [0.1, 0.4, 0.5],
                  [0.02, 0.9, 0.08]])  

class_target = [0, 1, 1]
print(softmax_output[[0, 1, 2], class_target])
loss = -np.log(softmax_output[range(len(softmax_output)), class_target])
print(loss)
