import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from algorithms import NormalizeOzone
from algorithms import RidgeRegressor



def main():
    """

    Mettez votre l'appel de chaque fonction ici.
    Exemple :

    DecisionTree.main(data)
    """
    #car_seats_bdd = pd.read_csv("data/Carseats.csv")
    #ozone_bdd = pd.read_table("data/ozone_complet.txt", sep = ";")
    #print(ozone_bdd)
    cwd=os.getcwd()
    X_train, X_test, Y_train, Y_test = NormalizeOzone.PreProcessingOzone(cwd+'/data/ozone_complet.txt', ';')
    #RidgeRegressor.ridge_regressor(X_train, X_test, Y_train, Y_test )
    # Driver code 
  
    # Model training     
    model = RidgeRegressor.RidgeRegressor( iterations = 1000,learning_rate = 0.01, l2_penality = 5 ) 
    model.fit( X_train, Y_train ) 
      
    # Prediction on test set 
    Y_pred = model.predict( X_test )     
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) )      
    print( "Real values      ", Y_test[:3] )     
    print( "Trained W        ", round( model.W[0], 2 ) )     
    print( "Trained b        ", round( model.b, 2 ) ) 
    
    i=0
    mean_absolute_error=0
    for index in Y_pred.index:
        mean_absolute_error+=abs(Y_pred[index]-Y_test[index])
        i+=1
    mean_absolute_error=mean_absolute_error/i
    print("mean absoulte error : ", mean_absolute_error)


if __name__ == '__main__':
    main()