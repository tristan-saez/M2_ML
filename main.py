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
    model = RidgeRegressor.RidgeRegressor( iterations = 1000,learning_rate = 0.01, l2_penality = 0.1 ) 
    model.fit( X_train, Y_train ) 
      
    # Prediction on test set 
    Y_pred = model.predict( X_test )     
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) )      
    print( "Real values      ", Y_test[:3] )     
    print( "Trained W        ", round( model.W[0], 2 ) )     
    print( "Trained b        ", round( model.b, 2 ) ) 

    i=0
    mean_absolute_error=0
    for pred in Y_pred:
        mean_absolute_error+=abs(pred-Y_test[i])
        i+=1
    mean_absolute_error=mean_absolute_error/i
    print("mean absoulte error : ", mean_absolute_error)
    """
    print("1")
    print(Y_test)
    print("2")
    print(Y_pred)
    print("3")
    print(Y_test.shape)
    print("4")
    print(Y_pred.shape)
    print("5")
    print(Y_test.index)
    print("6")
    print(Y_pred.index)
    print("ok")

    # Visualization on test set  
    Y_test_plot=Y_test.to_numpy()   
    Y_pred_plot=Y_pred.to_numpy()    
    Y_test_plot_index=Y_test.index.to_numpy()    
    Y_pred_plot_index=Y_pred.index.to_numpy()     
    print(Y_pred_plot.shape)
    print(Y_test_plot.shape)
    print(Y_test_plot_index.shape)
    plt.scatter( Y_test_plot_index, Y_test_plot, color = 'blue' )     
    plt.plot( Y_test_plot_index, Y_pred_plot, color = 'orange' )
    plt.show() 
    """

if __name__ == '__main__':
    main()