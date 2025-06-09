# 🚀 Notre super modèle d'iA (oui, avec un petit 'i' pour l'humilité... ou pas) 🚀

Salut à toi, explorateur du code, curieux de l'IA (ou juste là pour le café) ! Bienvenue dans le saint des saints de la prédiction, là où les bits dansent et les neurones... bah, ils calculent. On a concocté un truc pas piqué des hannetons, alors attache ta ceinture !

---

#### 🛠️ Installation : Prêt à décoller ? 🛠️

Pour que notre fusée de l'IA ne se transforme pas en vulgaire caillou, il faut un minimum de préparation.

###### Environnement dockerisé
Le projet peut être lancé depuis un container. Pour cela il faut d'abord copier le fichier d'environnement exemple comme suit :   
```bash
cp .env.example .env
```
Il faut ensuite monter le container et le lancer :
```bash
docker compose build
docker compose up -d 
docker compose exec api bash
```

###### Le `.venv` (alias, notre petite bulle de sérénité)

Pour éviter que ce projet ne mette le bazar dans ton PC (et vice-versa), on utilise un environnement virtuel. C'est comme une petite bulle magique où toutes les dépendances du projet vivent en harmonie, loin des conflits extérieurs.

Si tu as Python (et un peu de chance) :

```bash
python -m venv .venv
```

Puis active-le (c'est le moment "abra cadabra") :

* **Windows (PowerShell) :**
    ```bash
    .\.venv\Scripts\Activate.ps1
    ```
* **Windows (CMD) :**
    ```bash
    .\.venv\Scripts\activate.bat
    ```
* **macOS / Linux :**
    ```bash
    source .venv/bin/activate
    ```
Félicitations ! Tu es dans notre bulle. Ne respire pas trop fort, l'air y est précieux.

###### Le `requirements.txt` (alias, la liste de courses pour les geeks)

Maintenant que tu es dans la bulle, il faut la meubler avec les outils nécessaires. Ce fichier contient tout ce qu'il faut pour que Python comprenne nos blagues (et nos calculs).

Assure-toi que ton `.venv` est activé, puis :

```bash
pip install -r requirements.txt
```

Ça va mouliner un peu. C'est normal. C'est la magie qui s'opère.

---

#### 🧠 Le Modèle : Notre cerveau artificiel (enfin, un bout) 🧠

Oublie les super-ordinateurs qui prennent toute une pièce. Notre bijou, c'est un **super Neural Network (NN)** ! Oui oui, un NN à l'état de l'art (pour nous, en tout cas). Il est si avancé qu'il a :

* **2 couches "dense"** : parce que "dense", c'est le futur. Plus c'est dense, mieux c'est, n'est-ce pas ? 😉
* **1 couche de prédiction** : c'est là que la magie se produit. Elle crache la réponse, et on espère qu'elle a raison.

Prépare-toi à être émerveillé (ou juste à voir des chiffres, c'est selon).

---

#### 🗺️ Architecture : Où va quoi dans notre petit monde ? 🗺️

Pour ne pas se perdre dans les méandres de notre génie, voici comment on a organisé notre projet. C'est un peu comme une carte au trésor, mais le trésor, c'est le code !

```
.
├── data/
│   ├── df_new.csv
│   └── df_old.csv
├── models/
│   ├── models.py
│   ├── model_2024_08.pkl
│   └── preprocessor.pkl
├── modules/
│   ├── evaluate.py
│   ├── preprocess.py
│   └── print_draw.py
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

###### `data/` (Le garde-manger du projet)
Ici, c'est là que nos précieuses données vivent.
* `df_new.csv` : Les données fraîches du jour, prêtes à être dévorées par notre IA.
* `df_old.csv` : Les classiques, les vétérans, ceux qui ont tout vu. On les garde par nostalgie (et pour la rétrospective).

###### `models/` (Le garage à cerveaux)
Ce dossier, c'est notre caverne d'Ali Baba des cerveaux artificiels.
* `models.py` : Les plans de nos futurs cyborgs... euh, de nos modèles. C'est ici que l'on définit l'architecture de nos NN et autres merveilles.
* `model_2024_08.pkl` : Une version sauvegardée de notre modèle. On l'a encapsulé pour qu'il ne s'échappe pas et ne domine pas le monde... pas encore.
* `preprocessor.pkl` : L'outil magique qui prépare les données avant de les donner à manger au modèle. Sans lui, c'est l'indigestion assurée !

###### `modules/` (La boîte à outils de MacGyver)
Ce sont nos couteaux suisses du code. Chaque fichier est un expert dans son domaine.
* `evaluate.py` : Le juge impitoyable qui dit si notre modèle est un génie ou un cancre.
* `preprocess.py` : Le chef cuisinier des données. Il les nettoie, les coupe, les assaisonne pour qu'elles soient parfaites pour notre IA.
* `print_draw.py` : L'artiste du groupe. Il transforme nos chiffres barbares en beaux graphiques pour que même ta grand-mère puisse comprendre (enfin, presque).

#### Les tests
Vous pouvez lancer les tests automatisés avec la commande :
```bash
pytest
```
---

On espère que cette petite virée dans notre projet t'a plu. N'hésite pas à jeter un œil au `main.py` pour lancer le grand spectacle !

*Fait avec amour, code et une bonne dose de caféine (et un peu de folie).*