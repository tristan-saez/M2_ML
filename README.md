# M2_ML (Projet M2 - Machine Learning)
![PyPI - Version](https://img.shields.io/pypi/v/scikit-learn?logo=scikit-learn&label=scikit-learn&color=f49b3c)
![PyPI - Version](https://img.shields.io/pypi/v/pandas?logo=pandas&label=pandas&color=10074c)
![PyPI - Version](https://img.shields.io/pypi/v/cvxopt?logo=cvxopt&label=cvxopt)


## Description

---

Ce projet a été effectué dans le cadre des enseignements de Machine Learning en cinquième année
à l'ISEN Yncrea Ouest de Brest. Le travail s'orientait autour de la reproduction d'algorithme
de Machine Learning from scratch.

---

## Utiliser le programme

1. Installer les librairies nécessaire avec la commande suivante : ```pip install -r requirements.txt```
2. Lancez le programme : ```python main.py```

## Listes des librairies principales utilisées

1. ```scikit-learn```
2. ```cvxopt```
3. ```pandas```

## Explications concernant les algorithmes SVM

### Classe ```svm_reg``` : Modèle de régression

La classe « svm_reg » implémente un modèle de régression utilisant la méthode des Machines à Vecteurs de Support pour la régression. Le but principal de cette classe est de trouver une fonction qui, étant donné un ensemble de données d’entraînement, prédit avec précision une valeur continue pour des nouvelles observations. La classe prend en entrée les paramètres « C », « gamma », et « m_type », qui influencent respectivement la régularisation, la largeur du noyau (pour les noyaux non linéaires), et le type de noyau à utiliser pour transformer les données. L’apprentissage du modèle est réalisé en résolvant un problème d’optimisation quadratique.

> Le paramètre « C » contrôle le compromis entre le degré d'ajustement du modèle aux données d’entraînement et la complexité du modèle. En d'autres termes, « C » influence la quantité de régularisation appliquée au modèle. Il agit comme une pénalité pour les erreurs d’entraînement, ce qui signifie qu’il détermine à quel point le modèle est autorisé à avoir des erreurs dans les prédictions lors de l'apprentissage.

> Le paramètre « gamma » influence la complexité de la frontière de décision en contrôlant l’étendue d’influence d’un seul point de données. En d'autres termes, « gamma » détermine comment un point de données individuel affecte la décision du modèle. Il est particulièrement pertinent pour les noyaux non linéaires tels que le noyau RBF, polynomial, et sigmoïde.

### Classe ```svm_class``` : Classifieur One-vs-All*

La classe « svm_class » implémente un classificateur SVM multiclasses basé sur plusieurs modèles « svm_reg ». L'approche One-vs-All consiste à entraîner un régressseur SVM pour chaque classe afin de distinguer cette classe de toutes les autres. Le modèle « svm_reg » est capable de prédire des valeurs réelles, ce qui est utile pour la classification lorsqu'il est utilisé dans une approche One-vs-All.
Pour chaque classe unique dans « Y », on crée une instance de « svm_reg ». Par exemple, si « Y » a deux classes {0, 1}, deux modèles « svm_reg » seront créés.
Les étiquettes « Y » sont modifiées pour chaque modèle :
•	Les points appartenant à la classe actuelle sont étiquetés comme 1.
•	Les points appartenant aux autres classes sont étiquetés comme -1.
La classe dont le modèle « svm_reg » retourne la plus grande valeur est choisie comme prédiction finale.
Si les valeurs de décision sont par exemple {-0.5, 0.8}, le modèle prédit la classe « 1 » car sa valeur (0.8) est la plus élevée
