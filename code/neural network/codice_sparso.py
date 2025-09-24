
''''
def f1(softmax_output, class_target):
    softmax_loss = Softmax_CCE()
    softmax_loss.backward(softmax_output, class_target)
    dvalues1 = softmax_loss.dinputs

def f2(softmax_output, class_target):
    activation = Activation_Softmax()
    activation.output = softmax_output
    loss = Loss_CategoricalCrossEntropy()
    loss.backward(softmax_output, class_target)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs
'''
'''
    t1 = timeit(lambda: f1(softmax_output, class_target), number=1000)
    t2 = timeit(lambda: f2(softmax_output, class_target), number=1000)
    print("t1: ", t1, "  t2: ", t2)
    print(t2/t1)
'''

'''
start_time = time.time() #tempo iniziale
end_time = time.time() #tempo corrente
exetution_time = end_time - start_time
'''