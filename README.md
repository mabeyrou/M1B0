# Monitoring
Pour visualiser le monitoring des entraînements du modèle lancez :

```bash
mlflow ui
```
# Processus de 
Le script d'entrainement : `retrain.py`
## Avec l'ancien modèle (`models/model_2024_08.pkl`)
Le modèle `model_2024_08.pkl` a été entraîné avec l'ancien jeu de donné (`data/df_old.csv`)
### Réentrainement avec les anciennes données (+50 epochs)
R²=0.86
### Réentrainement avec les nouvelles données (+50 epochs)
R²=0.88
### Réentrainement avec les nouvelles données (+300 epochs)
R²=0.88
Pas d'amélioration du R².
La courbe `val_loss` reste stable, même après 300 epochs.
Il n'est probablement pas utile d'augment autant les epochs

## Avec le nouveau modèle (`models/model_2025_06.pkl`, 100 epochs)
J'ai entraîné un modèle vierge (`model_2024_08.pkl`) directement avec le nouveau jeu de donné (`data/df_new.csv`), avec 100 epochs.
R²=0.88
### Réentrainement avec les nouvelles données (+50 epochs)
R²=0.88
### Réentrainement avec les anciennes données (+50 epochs)
R²=0.86
Mais avec un val_loss encore en baisse (plus qu'avec les nouvelles données lors du réentrainement)
### Réentrainement avec les nouvelles données (+300 epochs)
R²=0.87
Et le val_loss repart à la hausse.