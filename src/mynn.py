import numpy as np
import pickle

model = pickle.load(open('myfirstnn.pickle','rb'))

class MyFirstNN(object):
    def __init__(self):
        self.weights = model.get('weights')
        self.bias = model.get('bias')

    def sigmoid_function(self, x):
        return 1/(1 + np.exp(-x))

    def neuralnet(self, x_input):
        result = np.dot(x_input, self.weights) + self.bias
        return self.sigmoid_function(result)

    def predict(self, x_input):
        x_input = np.array(x_input)
        pred = self.neuralnet(x_input)
        pred = list(pred)
        if pred:
            pred = round(pred[0])
        else:
            pred = 0    
        return pred
    
