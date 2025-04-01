import numpy as np
from Script_AlphaZero import AlphaZero


def traiterPolitique(politique):
    return [["LEFT", "RIGHT", "UP", "DOWN"][indice] for indice in np.argsort(-politique)]

def jouerLaProchainneAction_AlphaZero(
    jeu, liste_mouvementsValides, nom_reseau):

        AZ = AlphaZero(jeu)
        AZ.importer(nom_reseau)
        valeur, politique = AZ.evaluer()
        politique_traitee = traiterPolitique(politique)
        i = 0
        while politique_traitee[i] not in liste_mouvementsValides:
            i += 1
        action = politique_traitee[i]
        jeu.jouerLaProchainneAction(action)

def jouer_avec_AlphaZero(jeu, nom_reseau):
    liste_mouvementsValides = jeu.obtenirLesMouvementsValides()
    nb_mouvementsValides = len(liste_mouvementsValides)
    score = jeu.obtenirScore(nb_mouvementsValides)
    
    while score != -1:
        jeu.afficher()
        jouerLaProchainneAction_AlphaZero(
            jeu, liste_mouvementsValides, nom_reseau)
        liste_mouvementsValides = jeu.obtenirLesMouvementsValides()
        nb_mouvementsValides = len(liste_mouvementsValides)
        score = jeu.obtenirScore(nb_mouvementsValides)

    jeu.afficher()
