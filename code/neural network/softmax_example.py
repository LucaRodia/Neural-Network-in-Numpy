import math
import numpy as np
import nnfs

nnfs.init()
E = math.e

layer_outputs = [[4.8, 2, 1.21],
                 [2.12, 0.65, 3.24],
                 [6, 1.38, 1]]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) #axis dice di fare la somma lungo le righe. keepdims trasforma una lista di 3 valori
                                        # in una lista con dentro una lista di 3 valori

print(np.sum(layer_outputs, axis=1, keepdims=True))  #semplicemente mantiene l'ordinamento (matrice con 3 righe) e fa la somma

