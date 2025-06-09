## Veille technologique
### MSE (Mean Squared Error - Erreur Quadratique Moyenne)
Le MSE mesure la moyenne des carrés des écarts entre les valeurs prédites et les valeurs réelles. Il s'agit de l'une des métriques les plus couramment utilisées pour évaluer les modèles de régression.  

`MSE = (1/n) × Σ(yi - ŷi)²`
n = nombre d'observations
yi = valeur réelle
ŷi = valeur prédite  

Le MSE pénalise fortement les grandes erreurs grâce à l'élévation au carré, ce qui le rend particulièrement sensible aux valeurs aberrantes. Une prédiction avec un écart de 10 contribuera 100 fois plus qu'une prédiction avec un écart de 1. Cette propriété peut être avantageuse si vous voulez fortement pénaliser les grosses erreurs, mais peut aussi rendre votre modèle vulnérable aux outliers.
L'unité du MSE est le carré de l'unité de la variable cible, ce qui peut rendre son interprétation moins intuitive. Un MSE de 0 indique une prédiction parfaite.

### MAE (Mean Absolute Error - Erreur Absolue Moyenne)
Le MAE calcule la moyenne des valeurs absolues des écarts entre les prédictions et les valeurs réelles. Il représente l'erreur moyenne en valeur absolue.  

`MAE = (1/n) × Σ|yi - ŷi|`  

Contrairement au MSE, le MAE traite toutes les erreurs de manière égale, indépendamment de leur magnitude. Il est plus robuste aux valeurs aberrantes car il n'élève pas les erreurs au carré. Le MAE conserve la même unité que la variable cible, ce qui facilite son interprétation directe.
Par exemple, si vous prédisez des prix en euros et obtenez un MAE de 50, cela signifie que vos prédictions s'écartent en moyenne de 50 euros de la réalité. Cette métrique donne une vision plus "démocratique" des erreurs, car chaque erreur contribue proportionnellement à sa magnitude.

### R² (Coefficient of Determination - Coefficient de détermination linéaire de Pearson)
Le R² mesure la proportion de la variance de la variable dépendante qui est expliquée par le modèle. Il indique dans quelle mesure le modèle capture la variabilité des données par rapport à un modèle de référence (généralement la moyenne).  

`R² = 1 - (SSres/SStot)`
SSres = Σ(yi - ŷi)² (somme des carrés des résidus)
SStot = Σ(yi - ȳ)² (somme totale des carrés, avec ȳ = moyenne des yi)  

Le R² varie généralement entre 0 et 1, bien qu'il puisse être négatif dans certains cas pathologiques. Une valeur de 1 indique que le modèle explique parfaitement la variance des données, tandis qu'une valeur de 0 signifie que le modèle n'est pas meilleur qu'une simple prédiction par la moyenne.
Un R² de 0.8 signifie que 80% de la variance des données est expliquée par le modèle, laissant 20% de variance non expliquée. Cette métrique est particulièrement utile pour comparer différents modèles ou pour communiquer la performance d'un modèle à des non-experts, car elle s'exprime en pourcentage de variance expliquée.  

Le R² peut être trompeur avec des modèles très complexes ou en présence de sur-apprentissage. Il tend à augmenter mécaniquement avec l'ajout de variables, même si elles n'apportent pas d'information pertinente. C'est pourquoi on utilise souvent le R² ajusté qui pénalise la complexité du modèle.

### Erreur d'Entraînement (Training Loss)
C'est le calcul de l'erreur des prédictions faites par le modèle sur les données d'entraînement. Elle est calculée à chaque passe (epoch) sur les données d'entraînement. Le coût diminue normalement lors de l'entraînement. La fonction de coût calcule la différence entre la valeur prédite et la valeur labellisée.

*   Affecte directement les ajustements de poids dans le modèle.
*   Est censée diminuer à mesure que l'entraînement progresse.
*   Peut fournir des informations sur la façon dont le modèle s'adapte aux données d'entraînement.

### Erreur de Validation (Validation Loss)
Évalue les performances du modèle sur le sous-ensemble de données de test (données que le modèle n'a pas rencontrées lors de l'entraînement). Cette métrique est un bon indicateur de la capacité du modèle à généraliser.

*   Aide à évaluer la généralisation du modèle.
*   Devrait diminuer initialement, mais si elle commence à augmenter alors que l'erreur d'entraînement diminue, cela indique un surapprentissage.
*   Souvent utilisée comme critère d'arrêt précoce pour éviter le surapprentissage.