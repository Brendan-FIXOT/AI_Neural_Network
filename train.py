import numpy as np
from model import NeuralNetwork
from train_visualization import TrainingVisualizer

class Train :
    def __init__(self, nn, data, target, visualizer) :
        self.nn = nn
        self.visualizer = visualizer
        # Mélanger les données
        indices = np.arange(len(data))  
        np.random.shuffle(indices)  

        self.data = data[indices]  # Appliquer le mélange
        self.target = target[indices]  # Appliquer le mélange
        
    
    def train(self):
        for i in range(len(self.data)) :
            # Récupérer la ligne i et forcer une forme 2D
            Features = self.data[i].reshape(1, -1)

            # Faire la propagation avant
            prediction = self.nn.forward(Features)
            
            true = np.array([1, 0, 0]) if self.target[i] == 0 else np.array([0, 1, 0]) if self.target[i] == 1 else np.array([0, 0, 1])
            
            # Vérification de la prédiction
            if np.argmax(prediction) == np.argmax(true):  
                print("✅ Bonne prédiction :", prediction, "Attendu:", true)
                self.visualizer.add_result(True)
            else:
                print("❌ Mauvaise prédiction :", prediction, "Attendu:", true)
                self.visualizer.add_result(False)
                
            # Rétropropagation
            self.nn.backward(Features, true, prediction)