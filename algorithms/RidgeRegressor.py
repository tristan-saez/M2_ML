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
      
    # Hypothetical function  h( x )  
    def predict( self, X ) :     
        return X.dot( self.W ) + self.b 
    


class RidgeRegressorSklearn():

    def __init__( self , l2_penality) : 
        self.l2_penality = l2_penality 
        self.modele_ridge_regressor = Ridge(alpha=self.l2_penality )
    
    def fit( self, X_train, Y_train ):
        self.modele_ridge_regressor.fit(X_train, Y_train)
    
    def predict(self, X_test):
        return self.modele_ridge_regressor.predict(X_test)

"""
def RidgeRegressorSklearn(l2_penality, X_train, Y_train):
    modele_ridge_regressor = Ridge(alpha=1.0)
    modele_ridge_regressor.fit(X_train, Y_train)
"""
    
    
"""
def ridge_regressor(X_train, X_test, Y_train, Y_test ):
    import numpy as np
    import pandas as pd
    from numpy.linalg import inv
    

    LAMBDA=2
    # Define dataset (X,y)
    #X0 = np.array([[0.8,  1.2,  0.5,  -0.7, 1.0],
    #          [1.0,  0.8,  -0.4, 0.5,  -1.2],
    #          [-0.5, 0.3,  1.2,  0.9,  -0.1],
    #          [0.2,  -0.9, -0.7, 1.1,  0.5]])

    #y0 = np.array([3.2, 2.5, 1.8, 2.9])

    X=X_train.to_numpy()
    Y=Y_test.to_numpy()

    #X[["T6","T9","T12","T15","T18","Ne6","Ne9","Ne12","Ne15","Ne18","Vdir6","Vvit6","Vdir9","Vvit9","Vdir12","Vvit12","Vdir15","Vvit15","Vdir18","Vvit18","Vx"]]=data.to_numpy()
    #X=data.to_numpy()
    #y=data.index.to_numpy()
    #test1=X.nanmean(axis=0)
    #test2=X.nanstd(axis=0)
    #test3=X0.mean(axis=0)
    #test4=X0.std(axis=0)
    # Scale predictors
    X_scale = (X-X.mean(axis=0))/X.std(axis=0)

    # RIDGE REGRESSION MODEL - coefficients estimation
    # X*X^T + LAMBDA*I
    x1 = np.matmul(X_scale.T, X_scale) + LAMBDA*np.identity(21)
    # Transpose obtained matrix - (X*X^T + LAMBDA*I)^{-1}
    x1_inv = inv(x1)
    # ( (X*X^T + LAMBDA*I)^{-1} ) * X^T
    x2 = np.matmul(x1_inv, X_scale.T)
    # ( ( (X*X^T + LAMBDA*I)^{-1} ) * X^T ) * Y
    coef = np.matmul(x2, Y)
    # Estimated coeficients
    print(coef)

    # predictions
    np.matmul(X_scale, coef)+Y.mean()


    #pass

#https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
#faire une fonction stnaderdised(...) Qui standardise les donnée en soustrayant par la moyenne et divisant par l'écar type
"""
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv

LAMBDA = 2    # shrinkage parameter

# Define dataset (X,y)
X = np.array([[0.8,  1.2,  0.5,  -0.7, 1.0],
              [1.0,  0.8,  -0.4, 0.5,  -1.2],
              [-0.5, 0.3,  1.2,  0.9,  -0.1],
              [0.2,  -0.9, -0.7, 1.1,  0.5]])

y = np.array([3.2, 2.5, 1.8, 2.9])

# Scale predictors
X_scale = (X-X.mean(axis=0))/X.std(axis=0)

# RIDGE REGRESSION MODEL - coefficients estimation
# X*X^T + LAMBDA*I
x1 = np.matmul(X_scale.T, X_scale) + LAMBDA*np.identity(5)
# Transpose obtained matrix - (X*X^T + LAMBDA*I)^{-1}
x1_inv = inv(x1)
# ( (X*X^T + LAMBDA*I)^{-1} ) * X^T
x2 = np.matmul(x1_inv, X_scale.T)
# ( ( (X*X^T + LAMBDA*I)^{-1} ) * X^T ) * Y
coef = np.matmul(x2, y)
# Estimated coeficients
print(coef)

# predictions
np.matmul(X_scale, coef)+y.mean()


Ensuite il faut un moyen de tester plusieur LAMBDA jusqu'à trouver le bon.
Idée : tester 0.1, 0.5, 1, 2, 5, 10, 100, 1000 
Puis rechercher meilleur en prenant moitier haute basse plusieurs fois jusqu'à obtenir LAMBDA le plus précis

"""