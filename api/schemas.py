from pydantic import BaseModel
from typing import Literal

OuiNon = Literal['oui', 'non']
Region = Literal['Auvergne-Rhône-Alpes', 'Bretagne', 'Corse', 'Hauts-de-France',
                 'Île-de-France', 'Normandie', 'Occitanie', 'Provence-Alpes-Côte d\'Azur']
NiveauEtude = Literal['bac+2', 'bac', 'master', 'doctorat', 'aucun']
Sexe = Literal['H','F']

class Profile(BaseModel):
    age: int
    taille: float
    poids: float
    sexe: Sexe
    sport_licence: OuiNon
    niveau_etude: NiveauEtude
    region: Region
    smoker: OuiNon
    nationalité_francaise: OuiNon
    revenu_estime_mois: int