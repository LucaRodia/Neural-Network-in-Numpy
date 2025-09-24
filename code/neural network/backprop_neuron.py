import numpy as np







def backPropagate(dvalue, z, x, w, b):
    dw = []
    for i in range(len(w)):
        dw_i = dvalue * (1. if z > 0 else 0.) * x[i]
        dw.append(dw_i)
    db = dvalue * (1. if z > 0 else 0)

    #aggiorna
    k = 0.001 #se non uso la k i valori vanno fuori scala (va a z = -28.74 e smette di cambiare, z<0)
    for i in range(len(w)):
        w[i] -= k*dw[i]
    b -= k*db

    return w, b

def main():
    inputs = [0.2, 5, 4.1]
    weights = [0.5, 0.4, -0.03]
    bias = 0.6
    dvalue = 2
    z = np.dot(inputs, weights) + bias #passaggio in avanti
    print("0° volta, z: ", z)


    for j in range(100):
        temp_z = z
        w, b = backPropagate(dvalue, z, inputs, weights, bias)

        z = np.dot(inputs, w) + bias
        D_z = temp_z - z
        weights = w
        bias = b

        print(f"{j+1}° volta, z = ", z, ",  delta = ", D_z)

    print("weights: ", weights)
    print("bias: ", bias)




main()