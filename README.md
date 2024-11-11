# Application-du-MLPClassifier-pour-la-reconnaissance-des-emotions-dans-la-parole
Ce projet propose un modèle de reconnaissance des émotions à partir de la parole. Ce système est inspiré d'applications réelles dans les centres d'appels, où l'analyse des émotions permet aux employés d'ajuster leur approche en fonction de l'état émotionnel du client. Cette technologie pourrait ainsi offrir une expérience plus efficace aux employés

## Contexte et motivation

La reconnaissance des émotions dans la parole (Recognizing Emotions in Speech, RES) est un domaine en croissance, visant à identifier les émotions humaines à partir de caractéristiques vocales telles que le ton, la hauteur, et le rythme. Elle s’appuie sur des indices vocaux universels, tout comme les animaux peuvent comprendre nos émotions grâce à des indices vocaux. Toutefois, en raison de la subjectivité des émotions et des défis liés à l'annotation des données, La reconnaissance des émotions dans la parole demeure une tâche complexe.

Ce projet met en œuvre un modèle de classification capable de détecter des émotions dans des échantillons audio en extrayant des caractéristiques audio spécifiques, comme les coefficients MFCC, le spectrogramme Chroma et Mel. Le modèle est entraîné et évalué sur l’ensemble de données **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song).

## Objectif

Ce projet vise à construire un modèle de machine learning robuste permettant la reconnaissance des émotions de la parole en exploitant des techniques de traitement du signal audio avec les bibliothèques **librosa** et **sklearn**. L'ensemble de données utilisé pour l'entraînement est **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song), un jeu de données d'expressions vocales et de chansons évalué pour sa validité émotionnelle.

## Technologies utilisées

- **Python** pour la mise en œuvre de l'algorithme et du modèle
- **Librosa** pour l'extraction des caractéristiques audio (MFCC, chroma, mel spectrogram)
- **scikit-learn** pour la classification des émotions avec MLPClassifier et autres modèles
- **GridSearchCV** pour l'optimisation des hyperparamètres

## Structure et méthodologie

### 1. Extraction des caractéristiques audio

Nous utilisons une fonction extract_feature pour extraire les caractéristiques audio pertinentes (MFCC, chroma, et mel) à partir des fichiers audio d’entraînement. Ces caractéristiques se définissent comme suit :

- **MFCC** (Mel Frequency Cepstral Coefficients) : Représente la distribution fréquentielle de puissance à court terme, utile pour capturer des aspects caractéristiques de la voix humaine.
- **Chroma** : Représente les 12 classes de hauteur de son, indiquant la présence d’une certaine tonalité musicale.
- **Mel** : Spectrogramme basé sur la fréquence Mel, utile pour l’analyse des intensités de fréquence.

### 2. Préparation des données

Les données sont chargées à l'aide de la fonction load_data, qui segmente l’ensemble en ensembles d'entraînement et de test. L'extraction des caractéristiques est réalisée pour chaque fichier audio, suivi d'une division des échantillons en caractéristiques (X) et labels d’émotions (y) et de la normalisation pour uniformiser les caractéristiques.

### 3. Modélisation et entraînement

- **Le modèle utilisé est un MLPClassifier** (perceptron multicouches), un réseau de neurones multicouche qui optimise la fonction de perte logarithmique. Ce modèle est particulièrement adapté aux tâches de classification en raison de sa capacité à modéliser des relations non linéaires dans les données. Les paramètres du modèle sont réglés pour maximiser la précision tout en évitant le surajustement.

- **Autres Modèles Testés** : RandomForestClassifier, SVM, et XGBoost avec des hyperparamètres optimisés par GridSearchCV pour comparer la performance du MLPClassifier.

### 4. Validation croisée

Nous avons utilisé 5-fold cross-validation pour une évaluation rigoureuse des modèles.

### 5. Évaluation

Les métriques incluent la précision, le rapport de classification et la matrice de confusion pour chaque modèle testé.

## Données
Le projet utilise la version réduite du jeu de données RAVDESS qui fournit des enregistrements audio de différentes émotions exprimées par des acteurs. Ce dataset est structuré de manière à représenter des émotions variées sous plusieurs modalités. Vous pouvez obtenir l’ensemble de données RAVDESS [e-ici](https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view).

## Prérequis

### Environnement de developpement

Il est recommandé d’utiliser JupyterLab pour une exécution interactive des blocs de code mais vous pouvez aussi utiliser d'autres environnement comme Visual studio code, Spyder, etc. :

```bash
jupyter lab
```

```bash
code .
```

### Librairies Python

Les bibliothèques nécessaires sont :

- **librosa** pour le traitement audio
- **scikit-learn** pour la classification et l'évaluation des modèles
- **xgboost** pour le modèle XGBoost
- **numpy** pour la manipulation de données numériques
- **soundfile** pour la lecture des fichiers audio

### Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/votre-utilisateur/votre-repo.git
cd votre-repo
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Hyperparamètres utilisés
Les hyperparamètres optimisés sont les suivants :

- **MLP (Perceptron Multicouche)** :

    - hidden_layer_sizes : [(200,), (300,), (400,)]
    - alpha : [0.0001, 0.001, 0.01]
    -learning_rate_init : [0.001, 0.01, 0.1]

- **Random Forest** :

    - n_estimators : [100, 200, 300]
    - max_depth : [10, 20, 30]
    - min_samples_split : [2, 5, 10]
      
- **SVM** :

    - C : [0.1, 1, 10]
    - kernel : ['linear', 'rbf', 'poly']
      
- **XGBoost** :

    - n_estimators : [50, 100, 200]
    - learning_rate : [0.01, 0.1, 0.2]
    - max_depth : [3, 5, 7]
      
## Résultats et Evaluation
Les résultats incluent :

- **Scores de validation croisée** : Mesure de la robustesse des modèles.
- **Métriques avancées** : Précision, rappel, et F1-score pour chaque émotion.
- **Matrices de confusion** : Représentation visuelle des erreurs de classification.

Les meilleurs paramètres pour chaque modèle sont obtenus via **GridSearchCV**. Les performances sont évaluées par validation croisée, permettant de sélectionner le modèle le plus adapté.

## Conclusion
Ce projet offre une solution robuste pour l'analyse des émotions basée sur des signaux audio, utilisant une sélection d'hyperparamètres et une validation croisée pour garantir des performances optimales. La combinaison de MFCC, Chroma, et Melspectrogramme permet d'améliorer la précision des modèles. Après avoir comparé différents paramètres de différents modèles, les meilleurs paramètres pour MLPClassifier ont été les plus satisfaisants et sont enregistrés et utilisés pour les prédictions. Les performances des autres modèles (Random Forest, SVM, XGBoost) sont également rapportées pour comparaison.

## Contribution
Les contributions sont les bienvenues ! Vous pouvez soumettre des pull requests pour ajouter des fonctionnalités, améliorer la précision ou enrichir la documentation.




