# Importing libraries 
  
import numpy as np 
import pandas as pd 

from sklearn.linear_model import Ridge

  
# Ridge Regression 
  
class RidgeRegressor() : 
      
    def __init__( self, learning_rate, iterations, l2_penality ) : 
        #On initialise les parametres renseignés
        self.learning_rate = learning_rate         
        self.iterations = iterations         
        self.l2_penality = l2_penality 
          
               
    def fit( self, X, Y ) : 
          
        # no_of_training_examples, no_of_features  
        # self.m correspon au nombre de parametres (utiliser pour calcul du gradient)   
        # self.n correspond au nombre de features (utiliser pour taille de la matrice des poids)    
        self.m, self.n = X.shape 
          
        #On initialise la matrice des poids à zéro       
        self.W = np.zeros( self.n ) 
          
        self.b = 0        
        self.X = X         
        self.Y = Y 
          
        #multiples itérations (correspondant au nombre d'époch) d'aprentissage et d'adaptation des points pour
        #minimiser l'écart entre résultat trouvé et résultat recherché.
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self
      
    #Fonction d'aprentissage 
    def update_weights( self ) : 
        #On prédit une premiere fois les valeurs pour comparer le résultat obtenu avec celui attendu           
        Y_pred = self.predict( self.X ) 
          
        #La méthode d'aprentissage marche par descente de gradient. On doit donc calculer les gradients qui viendront
        # mettre à jours nos poids qui permetent de prédire les valeurs.
        #Pour obtenir ces gradients on utilise la formule si dessous qui compare les résultats obtenus aux résultats
        # attendu et divise par le nombre de parametres 
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +               
               ( 2 * self.l2_penality * self.W ) ) / self.m      
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
          
        #A partir des gradients calculer juste avant, on vient metre a jours les poids qui servent à prédires les valeur
        #Les poids sont mis à jours dans le but qu'ils puissent prédire avec plus de précision par la suite
        #De plus, la modification des poids est limiter par le learning rate. Cela fait que le réseau n'est pas 
        #Dépendant d'une seul epochs mais demande ducoup plusieurs epochs d'aprentissage.
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db         
        return self
      
    #On prédit les valeurs de sorties en fonction des valeurs d'entrée et de la matrice de poid établie
    #et entrainné plus tot
    def predict( self, X ) :     
        return X.dot( self.W ) + self.b 
    


class RidgeRegressorSklearn():
    #Mise au point d'un modéle de ridge régression en utilisant les fonction sklearn
    def __init__( self , l2_penality) : 
        self.l2_penality = l2_penality 
        self.modele_ridge_regressor = Ridge(alpha=self.l2_penality )
    
    def fit( self, X_train, Y_train ):
        self.modele_ridge_regressor.fit(X_train, Y_train)
    
    def predict(self, X_test):
        return self.modele_ridge_regressor.predict(X_test)
    