#voglio minimizzare l'output dopo la ReLU
import numpy as np
np.random.seed(42)

def main():
    X = np.array([[0.5, 6, 4], [3, 1.58, 2.13], [0.4, 8, 3.6]]) #n_inputs = 3
    dvalues = np.array([[1., 1., 1.], [2., 2., 2.],[3., 3., 3.], [4., 4., 4.]])
    layer1 = Layer_Dense(3, 4) #4 neuroni
    activation_ReLU1 = ReLU()

    # layer1.forward(X)
    #print("before relu")
    #print(layer1.outtput)
    # activation_ReLU1.forward(layer1.output) 
    #print("Output dopo la ReLU")
    #print(activation_ReLU1.output)


    for i in range(100):
        layer1.forward(X)
        activation_ReLU1.forward(layer1.output)
        if i%10 == 0:
            print(f"{i}° ciclo")
            print(activation_ReLU1.output)

        activation_ReLU1.backward(dvalues)
        layer1.backward(activation_ReLU1.inputs)
        
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    
    def backward(self, dvalues):
        dinputs = np.dot(dvalues, self.weights.T)
        dweights = np.dot(self.inputs.T, dvalues)
        dbias = np.sum(dvalues, axis=0, keepdims=True)

        k = 0.001
        applyGradient(self.weights, dweights, k)
        applyGradient(self.biases, dbias, k)

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        output = np.copy(inputs)
        for i in range(len(output)):
            for j  in range(len(output[i])):
                if output[i][j] <= 0:
                    output[i][j] = 0
        self.output = output

    def backward(self, dvalues):
        #print(dvalues)
        for i in range(len(self.inputs)):
            #print(self.output[i])
            for j in range(len(self.inputs[i])):
                #print("riga i-esima")
                #print(self.output[i])
                #print("dvalues: ", dvalues)
                #print("i: ", i, "  j: ", j)
                if self.inputs[i][j] <= 0:
                    self.inputs[i][j] = 0

        


def applyGradient(values, dvalues, k):
    #values e dvalues devono avere la stessa dimensione
    for i in range(len(values)):
        for j in range(len(values[i])):
            values[i][j] -= k*dvalues[i][j]

            





main()