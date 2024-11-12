import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fonction pour extraire les caractéristiques (MFCC, Chroma, Mel)
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))
    return result

# Dictionnaire d'émotions et émotions observées
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Chargement des données et extraction des caractéristiques
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("L:\\JEU DE DONNEES\\ravdess data\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Diviser les données
x_train, x_test, y_train, y_test = load_data(test_size=0.2)

# Normalisation des données
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Encoder les étiquettes
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Fonction de validation croisée et d'optimisation pour un modèle donné
def perform_grid_search(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print(f"Meilleurs paramètres pour {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Meilleure précision pour {model.__class__.__name__}: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

# Hyperparamètres et optimisation

# MLP
mlp_param_grid = {
    'hidden_layer_sizes': [(200,), (300,), (400,)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}
best_mlp = perform_grid_search(MLPClassifier(batch_size=256, epsilon=1e-08, learning_rate='adaptive', max_iter=500),
                               mlp_param_grid, x_train, y_train)

# Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
best_rf = perform_grid_search(RandomForestClassifier(random_state=42), rf_param_grid, x_train, y_train)

# SVM
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}
best_svm = perform_grid_search(SVC(probability=True, random_state=42), svm_param_grid, x_train, y_train)

# XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
best_xgb = perform_grid_search(XGBClassifier(random_state=42), xgb_param_grid, x_train, y_train_encoded)

# Évaluation des modèles
def evaluate_model(model, x_test, y_test, y_test_labels):
    y_pred = model.predict(x_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred) if isinstance(model, XGBClassifier) else y_pred
    print(f"\nÉvaluation de {model.__class__.__name__}")
    print("Rapport de classification :\n", classification_report(y_test_labels, y_pred_labels, target_names=observed_emotions))
    print("Matrice de confusion :\n", confusion_matrix(y_test_labels, y_pred_labels))
    print(f"Précision : {accuracy_score(y_test_labels, y_pred_labels) * 100:.2f}%")

# Évaluer les modèles avec les meilleurs paramètres
evaluate_model(best_mlp, x_test, y_test_encoded, y_test)
evaluate_model(best_rf, x_test, y_test_encoded, y_test)
evaluate_model(best_svm, x_test, y_test_encoded, y_test)
evaluate_model(best_xgb, x_test, y_test_encoded, y_test)
