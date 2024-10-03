# Importing libraries 
  
import numpy as np 
import pandas as pd 

  
# Ridge Regression 
  
class RidgeRegressor() : 
      
    def __init__( self, learning_rate, iterations, l2_penality ) : 
          
        self.learning_rate = learning_rate         
        self.iterations = iterations         
        self.l2_penality = l2_penality 
          
    # Function for model training             
    def fit( self, X, Y ) : 
          
        # no_of_training_examples, no_of_features         
        self.m, self.n = X.shape 
          
        # weight initialization         
        self.W = np.zeros( self.n ) 
          
        self.b = 0        
        self.X = X         
        self.Y = Y 
          
        # gradient descent learning 
                  
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self
      
    # Helper function to update weights in gradient descent 
      
    def update_weights( self ) :            
        Y_pred = self.predict( self.X ) 
          
        # calculate gradients       
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +               
               ( 2 * self.l2_penality * self.W ) ) / self.m      
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
          
        # update weights     
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db         
        return self
      
    # Hypothetical function  h( x )  
    def predict( self, X ) :     
        return X.dot( self.W ) + self.b 
    
    
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