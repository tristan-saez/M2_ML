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
    #On obtient les données à répartir en base de test et d'apprentissage
    cwd=os.getcwd()
    X_train, X_test, Y_train, Y_test = NormalizeOzone.PreProcessingOzone(cwd+'/data/ozone_complet.txt', ';')
    #La fonction PreProcessingOzone() renvoie répartie 70% des données dans la base d'apprentissage et 30% 
    #dans la base de test

    #Mise en place d'un timer pour calcul du temps d'exécution
    start_time1 = time.time()
  
    #On appelle la classe RidgeRegressor pour créer le modèle. Ce dernier est paramétré pour apprendre sur 1000 epochs avec 
    #un taux d'apprentissage de 0.01. De plus, le coefficient de pénalité associer à l'algorithme de Ridge est de 5
    model = RidgeRegressor.RidgeRegressor( iterations = 1000,learning_rate = 0.01, l2_penality = 5 ) 
    #On entraine le modèle avec les précédents paramètres et la base d'entrainement
    model.fit(X_train, Y_train) 
      
    #On prédit les valeurs de notre base de test à partir du modèle entraîné
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
#A PARTIR DE MAINTENANT METHODE AVEC SKLEARN
###############################################################################################

    #Mise en place d'un timer pour calcul du temps d'exécution
    start_time2 = time.time()
  
    #On appel la classe RidgeRegressorSklearn, qui vient créer le modèle. Ce modèle utilise les méthodes issues de sklearn
    #De plus, le coefficient de pénalité associer à l'algorithme de Ridge est toujours de 5 pour comparer au modèle précédent.
    model2 = RidgeRegressor.RidgeRegressorSklearn(l2_penality = 5) 
    #On entraine le modèle avec les paramètres précédents et la base d'entrainement
    model2.fit(X_train, Y_train) 
      
    #On prédit les valeurs de notre base de test à partir du modèle entraîné
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