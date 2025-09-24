# neural network

#classe layer_dense per layer rete
#inizializzare pesi con valori piccoli e bias a 0
import numpy as np
import nnfs
import matplotlib.pyplot as plt
from generatore_dati import generateData
from timeit import timeit
import time

#np.random.seed(0)



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #np.random usa la distribuzione gaussiana, e alcuni valori possono essere più grandi di 1
                                                                #0,1 li riporta nel range desiderato (1 -1)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inpunts = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    #   non so perché ma crea una classe relu
        '''
        output_noReLU = np.dot(inputs, self.weights) + self.biases
        output_ReLu = []

        for value in output_noReLU:
            if value > 0:
                output_ReLu.append(value)
            else:
                output_ReLu.append(0)
        self.output = output_ReLu
        '''

    def backward(self, dvalues):
        self.dweights = np.dot(self.inpunts.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        



class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  # evidentemente fa lo stesso lavoro del for scritto da me 

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  #tolgo il max così evito di trovarmi dei valori troppo grandi
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1) #lo rende una matrice --> (-1, 1) dovrebbe renderlo un vettore colonna
            diag = np.diagflat(single_output)
            s_sT = np.dot(single_output, single_output.T)

            Jacobian_matrix = diag-s_sT
            self.dinputs[index] = np.dot(Jacobian_matrix, single_dvalue)

class Softmax_CCE:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inpunts, y_true):
        self.activation.forward(inpunts)  #cre un suo output interno
        self.output = self.activation.output #copio output softmax nel complesso
    #   userò l'output per fare il backward pass. L'unica cosa che mi serve è l'output della softmax
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()  #per ogni predizione (n righe) ho una derivata (i colonne) --> devo sottrarre quella giusta
        self.dinputs[range(samples), y_true] -= 1 
        self.dinputs = self.dinputs/samples         

class Loss:
    def calculate(self, output, y):  #output -> nostra predizione, y -> risultati corretti
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        clipped_pred = np.clip(y_pred, 1e-7, 1-1e-7)  #mette i dati tra 1e-7 e 1- 1e-7
        
        if len(y_true.shape) == 1: #in questo caso l'array delle risposte corrette viene passato come [1, 2, 0, ...] lista monodimensionale
            correct_confidences = clipped_pred[range(len(clipped_pred)), y_true]  #qui prendo l'output della softmax (la probabilità di estrarre il valore corrretto)

        elif len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            #correct_confidences = np.sum(clipped_pred*y_true, axis=1)  #prodotto elemento per elemento --> alla fine farò 1 * predizione elemento
            correct_confidences = clipped_pred[range(len(clipped_pred)), y_true]
            #aggiustato correttamente

        negative_log = -np.log(correct_confidences)
        return negative_log

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])  #quanti elementi ha ogni array risposta-possibile

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues  #divide elemento per elemento
        self.dinputs = self.dinputs / samples #divido per il numero di esempi che ho (poi sommerò per esempi, le colonne)

class Optimizer_SGD:

    def __init__(self, learning_rate = 0.1, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iterations = 0


    def update_learning_rate(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *(1 / (1 + self.decay * self.iterations))


    def update_parameters(self, layer):

        if self.momentum:

            if not hasattr(layer, "weight_momentum"):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.biases_momentum = np.zeros_like(layer.biases)

            weights_update = self.momentum * layer.weight_momentum - self.current_learning_rate * layer.dweights
            biases_update = self.momentum * layer.biases_momentum - self.current_learning_rate * layer.dbiases

            # sommo perché ho il contributo negativo del gradiente precedente e quello positivo del momentum (che mi fa "continuare dritto")
            #layer.weights += weights_update  
            #layer.biases  += biases_update
        
        else: 
            weights_update = -self.current_learning_rate*layer.dweights
            biases_update = -self.current_learning_rate*layer.dbiases

        layer.weights += weights_update
        layer.biases += biases_update   
    
    def update_iterations(self):
        self.iterations += 1



def main():
    X, y = generateData(100, 3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Softmax_CCE()

    for epoch in range(15001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)
        optimizer = Optimizer_SGD(learning_rate=0.5, decay=0.01, momentum=0.5)

        if epoch%1000 == 0:
            print("EPOCH: ", epoch)
            print("Loss: ", loss)
            predictions = np.argmax(loss_activation.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y)
            print("accuracy: ", accuracy)
            
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)  #gli passo solo dinputs perché dweights e dbiases non vengono propagati indietro,
    #   ma modificano soltanto quel layer
        dense1.backward(activation1.dinputs)

        #print("dbiases \n", dense1.dbiases)
        optimizer.update_learning_rate()
        optimizer.update_parameters(dense1)
        optimizer.update_parameters(dense2)
        optimizer.update_iterations()


'''
print("Tempo di esecuzione:   \n", exetution_time)
t1 = timeit(lambda: f1(), number=10)
print(t1)
'''


main()
