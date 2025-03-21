import numpy as np

class NeuralNetwork :
    def __init__(self, input_size, output_size, hidden_size = 64) :
        intput_size = 4
        output_size = 3
        learning_rate = 0.01
        
        # Initialisation de petits poids et des biais
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        b2 = np.zeros((1, hidden_size))
        W3 = np.random.randn(hidden_size, output_size) * 0.01
        
    def relu(x) :
        np.maximum(0,x) # fonction d'activation
        
    def forward(self, x) :
        Z1 = np.dot(x, self.weight1) + self.b1 # Premier produit matriciel pour les neuronnes de la première couche caché
        A1 = self.relu(Z1) # Activation ReLU
        Z2 = np.dot(A1, self.W2) + self.b2 # Deuxième produit matriciel pour les neuronnes de la seconde couche caché
        A2 = self.relu(Z2) # Activation ReLU
        E0 = np.dot(A2, self.W3) # Couche de sortie
        return E0