import numpy as np
from loss import LossFunction

class NeuralNetwork :
    def __init__(self, input_size, output_size, hidden_size = 64, learning_rate = 0.01) :
        self.intput_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
                
        # Initialisation de petits poids et des biais
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        
    def relu(self, x) :
        return np.maximum(0,x) # fonction d'activation
    
    # Permet d'obtenir des probabilités pour les trois types de fleurs (donnant au total 100%)
    def softmax(self, x) :
        exp_x = np.exp(x - np.max(x))  # Stabilité numérique
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def forward(self, x) :
        Z1 = np.dot(x, self.W1) + self.b1 # Premier produit matriciel pour les neuronnes de la première couche caché
        self.A1 = self.relu(Z1) # Activation ReLU
        Z2 = np.dot(self.A1, self.W2) + self.b2 # Deuxième produit matriciel pour les neuronnes de la seconde couche caché
        self.A2 = self.relu(Z2) # Activation ReLU
        Z3 = np.dot(self.A2, self.W3) + self.b3 # Couche de sortie
        return self.softmax(Z3) # Pas de softmax car on utilise MSE (Régréssion)
    
    def backward(self, x, Y_true, Y_pred) :
        dZ3 = LossFunction.mse_derivative(Y_true, Y_pred)
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dA2 = np.dot(dZ3, self.W3.T)

        dZ2 = dA2 * (self.A2 > 0)  # Dérivée de ReLU (si A2 > 0 alors 1 sinon 2)
        dW2 = np.dot(self.A1.T, dZ2) / x.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / x.shape[0]

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.A1 > 0)  # Dérivée de ReLU (si A3 > 0 alors 1 sinon 2)
        dW1 = np.dot(x.T, dZ1) / x.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / x.shape[0]

        # Mise à jour des poids
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1