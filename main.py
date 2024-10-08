import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from algorithms import NormalizeOzone
from algorithms import RidgeRegressor
import time




def main():
    """

    Mettez votre l'appel de chaque fonction ici.
    Exemple :

    DecisionTree.main(data)
    """
    #On obtient les données à répartir en base de test et d'aprentissage
    cwd=os.getcwd()
    X_train, X_test, Y_train, Y_test = NormalizeOzone.PreProcessingOzone(cwd+'/data/ozone_complet.txt', ';')
    #La fonction PreProcessingOzone() renvoie répartie 70% des données dans la base d'aprentissage et 30% 
    #dans la base de test

    #mise en place d'un timer pour calcul du temps d'execution
    start_time1 = time.time()
  
    #On appel la classe RidgeRegressor pour créer le modéle. Ce dernier est parametrer pour aprendre sur 1000 epochs avec 
    #un taut d'aprentissage de 0.01. De plus, le coefficient de pénalité associer à l'algorythme de Ridge est de 5
    model = RidgeRegressor.RidgeRegressor( iterations = 1000,learning_rate = 0.01, l2_penality = 5 ) 
    #On entraine le modéles avec les précédents paramatre et la base d'entrainement
    model.fit(X_train, Y_train) 
      
    #On prédit les valeurs de notre base de test à partir du modéle entrainné
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

    #mise en place d'un timer pour calcul du temps d'execution
    start_time2 = time.time()
  
    #On appel la classse RidgeRegressorSklearn, qui vient créer le modéle. Ce modéle utilise les methodes issus de sklearn
    #De plus, le coefficient de pénalité associer à l'algorythme de Ridge est toujours de 5 pour comparer au modéle précédent.
    model2 = RidgeRegressor.RidgeRegressorSklearn(l2_penality = 5) 
    #On entraine le modéles avec les précédents paramatre et la base d'entrainement
    model2.fit(X_train, Y_train) 
      
    #On prédit les valeurs de notre base de test à partir du modéle entrainné
    Y_pred2 = model2.predict( X_test )     

    i=0
    y=0
    mean_absolute_error2=0
    for index in Y_test.index:
        mean_absolute_error2+=abs(Y_pred2[y]-Y_test[index])
        i+=1
        y+=1
    mean_absolute_error2=mean_absolute_error2/i
    
    print("--- %s secondes pour le ridge regressor avec sklearn ---" % (time.time() - start_time2))
    print("mean absoulte error de ridge regressor avec sklearn : ", mean_absolute_error2)
    

if __name__ == '__main__':
    main()