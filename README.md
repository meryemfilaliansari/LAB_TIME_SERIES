# Analyse de Séries Temporelles avec Modèles Classiques et Non-Linéarités

## Description du Projet

Ce notebook présente une **analyse complète et pédagogique** des séries temporelles, depuis la génération synthétique de données jusqu'à l'application de modèles de prévision avancés. Il couvre les aspects théoriques et pratiques de l'analyse de séries temporelles, avec un focus particulier sur l'impact des **fonctions d'activation non-linéaires** (ReLU, Sigmoïde, Tanh) sur le comportement temporel.

---

## Objectifs du Notebook

1. **Générer une série temporelle synthétique** avec tendance, saisonnalité et bruit
2. **Explorer et visualiser** les composantes de la série
3. **Tester la stationnarité** (tests ADF et KPSS)
4. **Appliquer des modèles de prévision classiques** (lissage exponentiel, ARIMA, SARIMA)
5. **Introduire les non-linéarités** via les fonctions d'activation
6. **Préparer les données** pour des approches de réseaux de neurones

---

## Packages et Bibliothèques

### Packages Principaux
```python
- numpy (calcul numérique)
- pandas (manipulation de données)
- matplotlib (visualisation)
- seaborn (visualisations statistiques)
- statsmodels (modèles de séries temporelles)
- scipy (tests statistiques)
- scikit-learn (métriques d'évaluation, normalisation)
```

### Installation
```bash
pip install statsmodels pmdarima scipy scikit-learn
```

---

## Structure du Notebook

### **PARTIE 1 : GÉNÉRATION ET EXPLORATION DES DONNÉES**

#### Cellule 1-2 : Installation et Import
- Installation des packages nécessaires
- Import des bibliothèques
- Configuration des styles de graphiques

#### Cellule 3 : Paramètres de Génération
```
n = 60 observations (5 ans × 12 mois)
beta0 = 100 (intercept)
beta1 = 2 (coefficient de tendance)
A = 20 (amplitude saisonnière)
P = 12 (période mensuelle)
sigma = 5 (écart-type du bruit)
train_size = 48 (4 ans)
test_size = 12 (1 an)
```

#### Cellule 4 : Génération de la Série
**Formule mathématique** :
```
Y(t) = beta0 + beta1×t + A×sin(2π×t/P) + ε(t)
où ε ~ N(0, σ²)
```

La série est composée de :
1. **Tendance linéaire** : `100 + 2×t`
2. **Saisonnalité sinusoïdale** : `20×sin(2π×t/12)`
3. **Bruit blanc gaussien** : `ε ~ N(0, 5²)`

#### Cellule 5 : Statistiques Descriptives
- Aperçu des premières et dernières observations
- Statistiques descriptives complètes
- Coefficient de variation, Skewness, Kurtosis

#### Cellule 6-7 : Visualisation des Composantes
- Graphiques de la série complète
- Décomposition visuelle : Tendance + Saisonnalité + Bruit
- Superposition pour comprendre la décomposition additive

#### Cellule 8 : Séparation Train/Test
- **Train** : 48 observations (80%) - 2020-01 à 2023-12
- **Test** : 12 observations (20%) - 2024-01 à 2024-12

---

### **PARTIE 2 : ANALYSE STATISTIQUE APPROFONDIE**

#### Cellule 9 : Statistiques par Année et Mois
- Statistiques agrégées par année (2020-2024)
- Analyse mensuelle (pattern saisonnier)
- Boxplots et graphiques en barres

#### Cellule 10 : Test de Normalité du Bruit
- **Test de Shapiro-Wilk**
- **Test de Jarque-Bera**
- Visualisations : histogramme, Q-Q plot, série temporelle
- Confirmation que le bruit suit bien une distribution N(0, σ²)

---

### **PARTIE 3 : TESTS DE STATIONNARITÉ**

#### Cellule 11 : Tests ADF et KPSS

**Test ADF (Augmented Dickey-Fuller)** :
- H0 : Série NON stationnaire (racine unitaire)
- H1 : Série stationnaire
- Résultat : p-value > 0.05 → Série **NON STATIONNAIRE**

**Test KPSS** :
- H0 : Série stationnaire
- H1 : Série NON stationnaire
- Résultat : p-value < 0.05 → Série **NON STATIONNAIRE**

#### Cellule 12 : Moyenne et Variance Mobiles
- Calcul de la moyenne mobile (fenêtre de 12 mois)
- Calcul de l'écart-type mobile
- Visualisation de la non-stationnarité

---

### **PARTIE 4 : ANALYSE DES AUTOCORRÉLATIONS**

#### Cellule 13 : ACF et PACF

**ACF (AutoCorrelation Function)** :
- Mesure la corrélation entre Y(t) et Y(t-k)
- Décroissance lente → Confirme la non-stationnarité
- Pics périodiques tous les 12 lags → Saisonnalité

**PACF (Partial AutoCorrelation Function)** :
- Corrélation partielle après élimination des lags intermédiaires
- Aide à identifier l'ordre p du modèle AR

---

### **PARTIE 5 : DÉCOMPOSITION SAISONNIÈRE**

#### Cellule 14 : Seasonal Decompose

**Modèle additif** : `Y(t) = T(t) + S(t) + R(t)`
- **T(t)** : Tendance (mouvement long terme)
- **S(t)** : Saisonnalité (variations périodiques)
- **R(t)** : Résidus (variations aléatoires)

#### Cellule 15 : Analyse des Résidus
- Test de normalité (Shapiro-Wilk)
- Test d'autocorrélation (Ljung-Box)
- Visualisations : série temporelle, histogramme, Q-Q plot, ACF
- Vérification que les résidus = bruit blanc

---

### **PARTIE 6 : DIFFÉRENCIATION**

#### Cellule 16 : Application de la Différenciation

Types de différenciation :
1. **Différenciation d'ordre 1** : `∇Y(t) = Y(t) - Y(t-1)`
   - Élimine la tendance
   
2. **Différenciation saisonnière** : `∇₁₂Y(t) = Y(t) - Y(t-12)`
   - Élimine la saisonnalité
   
3. **Double différenciation** : `∇∇₁₂Y(t)`
   - Élimine tendance ET saisonnalité

**Tests ADF sur séries différenciées** :
- Les séries différenciées deviennent **STATIONNAIRES**

---

### **PARTIE 7 : MODÈLES DE PRÉVISION**

#### Cellule 17 : Modèle 1 - SMA (Simple Moving Average)
```
Prévision = Moyenne des k dernières observations
Formule : Ŷ(t+1) = (1/k) × Σ Y(t-i)
```
- Fenêtre : 12 mois
- **Limites** : Ne capture ni tendance ni saisonnalité

#### Cellule 18 : Modèle 2 - SES (Simple Exponential Smoothing)
```
Prévision pondérée avec poids exponentiels décroissants
Formule : Ŷ(t+1) = α×Y(t) + (1-α)×Ŷ(t)
```
- Paramètre α optimisé automatiquement
- **Limites** : Ne capture ni tendance ni saisonnalité

#### Cellule 19 : Modèle 3 - Holt (Double Exponentiel)
```
Capture la TENDANCE via deux équations :
- Niveau : l(t) = α×Y(t) + (1-α)×(l(t-1) + b(t-1))
- Tendance : b(t) = β×(l(t) - l(t-1)) + (1-β)×b(t-1)
```
- Paramètres α et β optimisés
- **Amélioration** : Capture la tendance

#### Cellule 20 : Modèle 4 - Holt-Winters (Triple Exponentiel)
```
Capture TENDANCE + SAISONNALITÉ
Trois équations :
- Niveau : l(t)
- Tendance : b(t)
- Saisonnalité : s(t)
Prévision : Ŷ(t+h) = (l(t) + h×b(t)) × s(t-m+h)
```
- Paramètres α, β, γ optimisés
- **Meilleur modèle classique** pour cette série
- Métriques : MAE, RMSE, MAPE, R²

#### Cellule 21 : Comparaison des Modèles de Lissage
Tableau comparatif des 4 modèles avec :
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R²
- Capacités (tendance, saisonnalité)

#### Cellule 22 : Modèle 5 - AR (AutoRégressif)
```
Y(t) = c + φ₁Y(t-1) + φ₂Y(t-2) + ... + φₚY(t-p) + ε(t)
```
- Ordre p = 5
- Appliqué sur série différenciée
- Prévisions inversées à l'échelle originale

#### Cellule 23 : Modèle 6 - ARIMA
```
ARIMA(p, d, q) = AR(p) + I(d) + MA(q)
- p : ordre AR (autorégressif)
- d : ordre d'intégration (différenciation)
- q : ordre MA (moyenne mobile)
```
- Paramètres : ARIMA(5, 1, 5)
- Modélisation directe sur série originale
- Différenciation intégrée dans le modèle

#### Cellule 24 : Modèle 7 - SARIMA
```
SARIMA(p, d, q)(P, D, Q)s
- (p, d, q) : composantes non-saisonnières
- (P, D, Q) : composantes saisonnières
- s : période saisonnière (12)
```
- Paramètres : SARIMA(1,1,1)(1,1,1)₁₂
- **Meilleur modèle statistique** pour séries avec saisonnalité
- Capture tendance ET saisonnalité simultanément

---

### **PARTIE 8 : NON-LINÉARITÉS ET FONCTIONS D'ACTIVATION**

#### Cellule 25 : Définition des Fonctions d'Activation

**Fonctions implémentées** :
1. **ReLU** : `max(0, x)`
   - Crée des ruptures brutales
   
2. **Sigmoïde** : `1 / (1 + e^(-x))`
   - Crée des transitions douces
   - Sortie bornée entre 0 et 1
   
3. **Tanh** : `tanh(x)`
   - Saturation progressive
   - Sortie bornée entre -1 et 1
   
4. **Leaky ReLU** : `max(αx, x)` avec α=0.01
   - Variante de ReLU avec pente pour valeurs négatives
   
5. **Softplus** : `log(1 + e^x)`
   - Approximation lisse de ReLU

#### Cellule 26 : Application aux Composantes

**8 séries non-linéaires créées** :

1. **Tendance avec ReLU** :
   - Rupture brutale à t=30
   - `tendance_relu = tendance + 10 × ReLU(t - 30)`

2. **Saisonnalité avec Sigmoïde** :
   - Amplitude modulée progressivement
   - `25 × sigmoid((t-30)/8) × sin(2πt/P)`

3. **Composante de Saturation (Tanh)** :
   - Saturation progressive après t=45
   - `15 × tanh((t-45)/10)`

4. **Bruit avec Leaky ReLU** :
   - Bruit asymétrique

5-8. **Séries combinées** :
   - Avec ReLU appliquée
   - Avec Sigmoïde appliquée
   - Avec Tanh appliquée
   - Combinaison multiple

#### Cellule 27 : Visualisation des Effets Temporels
- Impact sur les composantes individuelles
- Comparaison série originale vs non-linéaire
- 4 graphiques détaillés :
  - Tendance linéaire vs ReLU
  - Saisonnalité originale vs Sigmoïde
  - Composante de saturation Tanh
  - Bruit gaussien vs Leaky ReLU

#### Cellule 28 : Analyse des Séries Complètes
Comparaison visuelle des séries complètes :
- Série avec rupture (ReLU)
- Série avec transition douce (Sigmoïde)
- Série avec saturation (Tanh)
- Série avec combinaison multiple

#### Cellule 29 : Fonctions d'Activation sur Signaux Synthétiques
Démonstration pédagogique :
- Visualisation pure des fonctions
- Application sur signal sinusoïdal
- Application sur signal rampe (linéaire)
- Grille de 9 graphiques explicatifs

#### Cellule 30 : Préparation pour Réseaux de Neurones

**Normalisation Min-Max** :
- Mise à l'échelle [0, 1] pour toutes les séries

**Création de séquences** :
```python
def create_sequences(data, sequence_length=12):
    # Utiliser 12 observations passées pour prédire la suivante
    return X_sequences, y_targets
```

**Structure de sortie** :
- X : (n_samples, sequence_length) → séquences d'entrée
- y : (n_samples,) → valeurs cibles
- Prêt pour LSTM/RNN

#### Cellule 31 : Zoom sur les Transitions
Analyse détaillée avec zoom temporel :
- Période ReLU (t=25 à t=35)
- Période Sigmoïde (t=20 à t=40)
- Période Tanh (t=40 à t=50)
- Comparaison directe des 3 fonctions

**Synthèse pédagogique** :
- ReLU → Ruptures BRUTALES (dérivée discontinue)
- Sigmoïde → Transitions DOUCES (dérivée continue)
- Tanh → SATURATION (valeurs bornées)

---

## Métriques d'Évaluation

Toutes les prévisions sont évaluées avec :

1. **MAE (Mean Absolute Error)** :
   ```
   MAE = (1/n) × Σ|y_true - y_pred|
   ```

2. **RMSE (Root Mean Squared Error)** :
   ```
   RMSE = √[(1/n) × Σ(y_true - y_pred)²]
   ```

3. **MAPE (Mean Absolute Percentage Error)** :
   ```
   MAPE = (100/n) × Σ|((y_true - y_pred) / y_true)|
   ```

4. **R² (Coefficient de détermination)** :
   ```
   R² = 1 - (SS_res / SS_tot)
   ```

---

## Concepts Clés Expliqués

### Stationnarité
Une série est **stationnaire** si :
- Moyenne constante dans le temps
- Variance constante dans le temps
- Covariance dépend seulement du décalage, pas du temps

### Décomposition Additive
```
Y(t) = Tendance(t) + Saisonnalité(t) + Résidus(t)
```

### Modèle SARIMA
Le plus complet pour séries avec saisonnalité :
```
SARIMA(p,d,q)(P,D,Q)s
```
- (p,d,q) : partie non-saisonnière (AR, Intégration, MA)
- (P,D,Q)s : partie saisonnière avec période s

### Fonctions d'Activation
Essentielles pour les réseaux de neurones :
- Introduisent la **non-linéarité**
- Permettent de modéliser des relations complexes
- Chaque fonction a des propriétés différentes

---

## Résultats Attendus

### Meilleur Modèle Classique
**Holt-Winters** :
- Capture tendance ET saisonnalité
- MAE minimal parmi les modèles de lissage
- MAPE faible (<10% généralement)

### Meilleur Modèle Statistique
**SARIMA(1,1,1)(1,1,1)₁₂** :
- Modélisation complète
- Métriques optimales
- Intervalles de confiance disponibles

### Impact des Non-Linéarités
- **ReLU** : Idéale pour ruptures structurelles
- **Sigmoïde** : Parfaite pour transitions progressives
- **Tanh** : Excellente pour effets de saturation

---

## Extensions Possibles

1. **Réseaux de Neurones Récurrents**
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Units)
   - Utiliser les données normalisées et les séquences

2. **Modèles Hybrides**
   - ARIMA + Neural Networks
   - Décomposition + LSTM

3. **Séries Multivariées**
   - VAR (Vector AutoRegression)
   - Modèles avec variables exogènes

4. **Deep Learning Avancé**
   - Transformers pour séries temporelles
   - Attention mechanisms
   - Prophet de Facebook

5. **Analyse Fréquentielle**
   - Transformée de Fourier
   - Analyse spectrale
   - Détection de périodicités cachées

---

## Références et Ressources

### Livres
- *Time Series Analysis* - James D. Hamilton
- *Forecasting: Principles and Practice* - Rob J Hyndman & George Athanasopoulos
- *Introduction to Time Series and Forecasting* - Peter J. Brockwell & Richard A. Davis

### Documentation
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Scikit-learn Time Series](https://scikit-learn.org/)
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)

### Articles
- Box, G. E. P., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control
- Holt, C. C. (2004). Forecasting seasonals and trends by exponentially weighted moving averages
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory (LSTM)

---

## Configuration Requise

### Python Version
- Python 3.8 ou supérieur

### Mémoire
- Minimum : 4 GB RAM
- Recommandé : 8 GB RAM

### Système
- Windows / Linux / macOS
- Jupyter Notebook ou JupyterLab
- VS Code avec extension Python (optionnel)

---

## Auteur

**FILALI ANSARI Meryem**

Analyse réalisée dans le cadre de l'étude des séries temporelles et de l'apprentissage automatique.

---

## Licence

Ce projet est à usage éducatif et pédagogique.

---

## Troubleshooting

### Erreur d'installation des packages
```bash
# Utiliser pip avec --upgrade
pip install --upgrade statsmodels pmdarima scipy scikit-learn
```

### Warnings durant l'exécution
```python
# Les warnings sont désactivés dans le notebook
import warnings
warnings.filterwarnings('ignore')
```

### Erreur de convergence ARIMA/SARIMA
- Essayer différents ordres (p, d, q)
- Vérifier la stationnarité de la série
- Augmenter le nombre d'itérations max

---

## Support

Pour toute question ou suggestion :
- Ouvrir une issue sur le repository
- Consulter la documentation des packages
- Vérifier les références bibliographiques

---

## Points Clés à Retenir

- **Génération de données** : Série synthétique avec tendance + saisonnalité + bruit

- **Tests statistiques** : ADF et KPSS pour la stationnarité

- **Décomposition** : Seasonal decompose pour séparer les composantes

- **Différenciation** : d=1 pour tendance, D=1 pour saisonnalité

- **Modèles classiques** : SMA → SES → Holt → Holt-Winters (du plus simple au plus complet)

- **Modèles statistiques** : AR → ARIMA → SARIMA (intégration progressive de la complexité)

- **Non-linéarités** : ReLU, Sigmoïde, Tanh pour modéliser des comportements complexes

- **Préparation ML** : Normalisation et création de séquences pour réseaux de neurones

---

## Structure du Répertoire

```
mssror/
├── FILALI_ANSARI_MERYEM.ipynb    # Notebook principal
├── README.md                       # Ce fichier
├── data/                           # (optionnel) Données externes
├── figures/                        # (optionnel) Graphiques exportés
└── models/                         # (optionnel) Modèles sauvegardés
```

---

**Date de dernière mise à jour** : Décembre 2025

**Version du notebook** : 1.0

**Statut** : Complet et fonctionnel
