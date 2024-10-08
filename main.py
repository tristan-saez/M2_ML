import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from algorithms import NormalizeOzone
from algorithms import RidgeRegressor
from algorithms import RidgeRegressorSklearn
import time




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

    start_time1 = time.time()
  
    # Model training     
    model = RidgeRegressor.RidgeRegressor( iterations = 1000,learning_rate = 0.01, l2_penality = 5 ) 
    model.fit(X_train, Y_train) 
      
    # Prediction on test set 
    Y_pred = model.predict( X_test )     
    
    i=0
    mean_absolute_error=0
    for index in Y_pred.index:
        mean_absolute_error+=abs(Y_pred[index]-Y_test[index])
        i+=1
    mean_absolute_error=mean_absolute_error/i
    print("--- %s secondes pour le ridge regressor sans sklearn ---" % (time.time() - start_time1))
    print("mean absoulte error de ridge regressor sans sklearn : ", mean_absolute_error)



###############################################################################################
#A PARTIR DE MAINTENANT METHODE AVEC sklearn
###############################################################################################


    start_time2 = time.time()
  
    # Model training     
    model2 = RidgeRegressorSklearn.RidgeRegressorSklearn( iterations = 1000,learning_rate = 0.01, l2_penality = 5 ) 
    model.fit(X_train, Y_train) 
      
    # Prediction on test set 
    Y_pred = model.predict( X_test )     

    i=0
    mean_absolute_error=0
    for index in Y_pred.index:
        mean_absolute_error+=abs(Y_pred[index]-Y_test[index])
        i+=1
    mean_absolute_error=mean_absolute_error/i
    print("--- %s secondes pour le ridge regressor sans sklearn ---" % (time.time() - start_time2))
    print("mean absoulte error de ridge regressor sans sklearn : ", mean_absolute_error)
    

if __name__ == '__main__':
    main()