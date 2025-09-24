#generatore dei dati

import numpy as np
import matplotlib.pyplot as plt



def generateData(points, classes):    #classes sono i tipi di punti (ci sono 3 tipi)
    X = np.zeros((points*classes, 2))
    Y = np.zeros(points*classes, dtype="uint8")

    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))  #associo a ogni coppia (x,y) un numero. Quella è la copia n-esima
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]  #i due argomenti della funzione sono il vettore della x e il vettore della y.
    #   poi np.c_ li mette a coppie (x1, y1), (x2, y2), ...  Infatti parto da vettori [x1, x2, ..], [y1, y1, ...]
    #   Quindi per ogni (xi, yi) creo xi come ri*sin(ti*2.5) e yi = ri*cos(ti*2.5)
        Y[ix] = class_number # mi salvo il tipo di punto
    return X, Y
'''
x, y = generateData(100, 3)
plt.scatter(x[:, 0], x[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
#plt.scatter(x[:, 0], x[:, 1], s=20) #no attributi
plt.show()
'''


