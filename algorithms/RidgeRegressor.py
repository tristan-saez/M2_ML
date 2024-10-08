import numpy as np 
import pandas as pd 

from sklearn.linear_model import Ridge

  
class RidgeRegressor() : 
      
    def __init__( self, learning_rate, iterations, l2_penality ) : 
        #On initialise les paramètres renseignés
        self.learning_rate = learning_rate         
        self.iterations = iterations         
        self.l2_penality = l2_penality 
          
               
    def fit( self, X, Y ) : 
          
        # self.m correspond au nombre de paramètres et sera utilisé pour calcul du gradient (division par nombre de paramètres
        # self.n correspond au nombre de features et sera utilisé pour taille de la matrice des poids     
        self.m, self.n = X.shape 
          
        #On initialise la matrice des poids à zéro       
        self.W = np.zeros( self.n ) 
          
        self.b = 0        
        self.X = X         
        self.Y = Y 
          
        #Multiples itérations (correspondant au nombre d'epochs) d'apprentissage et d'adaptation des points pour
        #minimiser l'écart entre résultat trouvé et résultat recherché.
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self
      
    #Fonction d'apprentissage 
    def update_weights( self ) : 
        #On prédit pour cette epoch les valeurs pour comparer le résultat obtenu avec celui attendu           
        Y_pred = self.predict( self.X ) 
          
        #La méthode d'apprentissage marche par descente de gradient. On doit donc calculer les gradients qui viendront
        # mettre à jours nos poids qui permettent de prédire les valeurs.
        #Pour obtenir ces gradients on utilise la formule si dessous qui compare les résultats obtenus aux résultats
        # attendu et divise par le nombre de paramètres 
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +               
               ( 2 * self.l2_penality * self.W ) ) / self.m      
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
          
        #A partir des gradients calculer juste avant, on vient mettre à jours les poids qui servent à prédire les valeurs
        #Les poids sont mis à jour dans le but qu'ils puissent prédire avec plus de précision par la suite
        #De plus, la modification des poids est limitée par le learning rate. Cela fait que les poids du modéle ne 
        #Sont pas dépendant d'une seul epoch mais demande ducoup plusieurs epochs d'apprentissage.
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db         
        return self
      
    #On prédit les valeurs de sorties en fonction des valeurs d'entrée et de la matrice de poids établie
    #et entraîné plus tôt
    def predict( self, X ) :     
        return X.dot( self.W ) + self.b 
    


class RidgeRegressorSklearn():
    #Mise au point d'un modèle de ridge régression en utilisant les fonctions sklearn
    def __init__( self , l2_penality) : 
        self.l2_penality = l2_penality 
        self.modele_ridge_regressor = Ridge(alpha=self.l2_penality )
    
    def fit( self, X_train, Y_train ):
        self.modele_ridge_regressor.fit(X_train, Y_train)
    
    def predict(self, X_test):
        return self.modele_ridge_regressor.predict(X_test)
    