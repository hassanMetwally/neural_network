import numpy as np


# Data
X = np.array(([2, 9], [1, 5], [3, 6]),dtype=float)
Y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([4,8]), dtype=float)

# Data rescale
X =X/np.amax(X, axis=0)     # maximum of X array
xPredicted = xPredicted/ np.amax(xPredicted, axis=0) # maximum of xPredicted (our
Y = Y/ 100                   # max test score is 100


class NeuralNetwork():
    def __init__(self):
        self.featureSize = 2
        self.outputSize = 1
        self.hiddenLayerSize = 3

        #Layer 1
        self.W1 = np.random.randn(self.hiddenLayerSize, self.featureSize)

        #Layer 2
        self.W2 = np.random.randn(self.outputSize, self.hiddenLayerSize)

    def forwardPropagation(self, X):
        #Layer 1
        self.Z1 = np.dot(X, self.W1.T)
        self.A1 = self.sigmoid(self.Z1)

        #Layer 2
        self.Z2 = np.dot(self.A1, self.W2.T)
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def backPropagation(self, X, Y, h):
        #layer 2
        self.dZ2 = (Y - h) * self.sigmoidPrime(h)
        self.dW2 = np.dot(self.A1.T,self.dZ2)

        # layer 1
        self.dZ1 = np.dot(self.dZ2, self.W2) * self.sigmoidPrime(self.A1)
        self.dW1 = np.dot(X.T, self.dZ1)
  
        self.W2 += self.dW2.T
        self.W1 += self.dW1.T



    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

    def sigmoidPrime(self, z):
        return z * (1 - z)


    def train(self, X, Y):
        h = self.forwardPropagation(X)
        self.backPropagation(X, Y, h)


    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")


    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forwardPropagation(xPredicted)))


nn = NeuralNetwork()



for i in range(10000): # trains the NN 1,000 times
    print ("# " + str(i) + "\n")
    print ("Input (scaled): \n" + str(X))
    print ("Actual Output: \n" + str(Y))
    print ("Predicted Output: \n" + str(nn.forwardPropagation(X)))
    print ("Loss: \n" + str(np.mean(np.square(Y - nn.forwardPropagation(X)))) )# mean sum squared
    print ("\n")
    nn.train(X, Y)
 

nn.saveWeights()
nn.predict()




