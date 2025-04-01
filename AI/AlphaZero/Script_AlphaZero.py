import numpy as np
import random
from tqdm import tqdm
from Script_MCTS_SP import MCTS_SP as MCTS
from Script_ResNet import ResNet


def softmax(X):
    """ 
    Remarque : la fonction d'activation softmax est a utiliser en
    sortie de réseau de neurones avec la perte d'entropie croisée
    """
    if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
        Exp_X = np.exp(X)
        return Exp_X / np.sum(Exp_X)
    else:
        Exp_X = np.exp(X)
        return Exp_X / np.sum(Exp_X, axis=1, keepdims=True)


class AlphaZero:
    arguments = {
        "nombre_apprentissages": 5,
        "nombre_partiesContreSoitMeme": 100
    }

    arguments_reseau = {
        "nb_blocs_res": 4,
        "nb_filtres": 64
    }

    arguments_hyperparametres_reseau = {
        "motif": 3,
        "pas": 1e-3,
        "epoque": 10,
        "taille_paquet": 32
    }

    arguments_MCTS = {
        "C": 2,
        "D": 0,
        "nb_recherches": 50,
        "profondeur_max": np.inf
    }

    def __init__(self, jeu, nom=f"AlphaZero"):
        self.jeu = jeu
        self.nom = nom

    def construireReseau(self):
        ResNet.hyperparametres = AlphaZero.arguments_hyperparametres_reseau
        reseau = ResNet(self.jeu, nom=self.nom)
        nb_blocs_res = AlphaZero.arguments_reseau["nb_blocs_res"]
        nb_filtres = AlphaZero.arguments_reseau["nb_filtres"]
        reseau.construire(nb_blocs_res, nb_filtres)
        self.reseau = reseau
        self.nom = reseau.nom

    def evaluer(self):
        X = self.jeu.encoderEtatPourReseau()
        X = np.expand_dims(X, axis=0)
        valeur, politique = self.reseau.evaluer(X)

        valeur = valeur.item()

        liste_mouvementsValides = self.jeu.obtenirLesMouvementsValides()
        politique *= self.jeu.vectoriserMouvementsValides(liste_mouvementsValides)
        politique = softmax(politique)

        return valeur, politique

    def jouerContreSoitMeme(self):
        mini_BdD = []
        self.jeu.initialiser()
        MCTS.arguments = AlphaZero.arguments_MCTS
        arbre = MCTS(self.jeu, self.reseau)
        arbre.rechercheArborescente()
        MCTS.arguments["nb_recherches"] = 15

        while not arbre.racine.estTerminal():
            arbre.rechercheArborescente()
            valeur, distribution = arbre.analyserRacine()
            liste_mouvementsValides = arbre.racine.liste_branches_parcourable

            self.jeu.changerDeConfiguration(arbre.racine.etat, 
                arbre.racine.joueur, arbre.racine.info)
            X = self.jeu.encoderEtatPourReseau()
            politique = np.zeros(self.jeu.dim_sortie_reseau)
            for i, action in enumerate(liste_mouvementsValides):
                politique[self.jeu.dico_indexAction[action]] = distribution[i]
            mini_BdD.append((X, np.array([0]), politique))
            
            try:
                action = np.random.choice(liste_mouvementsValides, p=distribution)
            except ValueError:
                print("liste_mouvementsValides =", liste_mouvementsValides)
                print("politique =", politique)
                print("action =", action)
            arbre.parcourirRacine(action)

        score = arbre.racine.score
        N = len(mini_BdD)
        for i in range(N):
            mini_BdD[i][1][0] = score

        return mini_BdD

    def entrainerReseau(self, BdD):
        random.shuffle(BdD)
        BdD_X = np.stack([X for X, V, P in BdD])
        BdD_V = np.stack([V for X, V, P in BdD])
        BdD_P = np.stack([P for X, V, P in BdD])
        historique = self.reseau.entrainer(BdD_X, BdD_V, BdD_P)
        self.reseau.sauvegarder()
        precision_val, precision_politique, perte = self.reseau.analyserEntrainement(historique)
        # print(precision_val, precision_politique, perte)
        
    def apprendre(self):
        for i in range(AlphaZero.arguments["nombre_apprentissages"]):
            print(f"### Apprentissage AlphaZero n°{i + 1} ###")
            BdD = []
            for _ in tqdm(range(AlphaZero.arguments["nombre_partiesContreSoitMeme"]),
                desc="Création BdD", mininterval=1):
                    mini_BdD = self.jouerContreSoitMeme()
                    BdD.extend(mini_BdD)
            self.entrainerReseau(BdD)

    def importer(self, nom):
        reseau = ResNet.importer(self.jeu, nom)
        self.reseau = reseau
