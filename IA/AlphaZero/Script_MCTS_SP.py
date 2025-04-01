import math
import numpy as np
import random


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


import copy  # Add this at the top


class NoeudAlpha:
    def __init__(self, jeu, noeud_parent=None, branche_parente=None, profondeur=0, priorite=0):
        self.noeud_parent = noeud_parent
        self.branche_parente = branche_parente
        self.profondeur = profondeur

        # Use deep copy instead of np.copy to handle heterogeneous objects.
        self.etat = copy.deepcopy(jeu.etat)
        self.joueur = jeu.joueur
        self.info = copy.deepcopy(jeu.info)

        self.score_moyen = 0
        self.nb_visites = 0
        self.priorite = priorite
        self.somme_scores_carres = 0

        liste_mouvementsValides = jeu.obtenirLesMouvementsValides()
        random.shuffle(liste_mouvementsValides)
        self.liste_branches_parcourable = liste_mouvementsValides
        if branche_parente == None:
            self.score = 0
        else:
            self.score = jeu.obtenirScore(len(liste_mouvementsValides))
        self.liste_fils = []

    def estRacine(self):
        return self.noeud_parent == None and self.branche_parente == None
    
    def estFeuille(self):
        return len(self.liste_fils) == 0
    
    def estTerminal(self):
        return self.score == -1

    def calculerUCTScore(self): # Upper Confidence bounds applied to Tree Score
        if self.nb_visites == 0:
            return np.inf
        else:
            Q = self.score_moyen / self.nb_visites
            U = MCTS_SP.arguments["C"] * self.priorite * math.sqrt(math.log(
                self.noeud_parent.nb_visites) / self.nb_visites)
            T = math.sqrt((self.somme_scores_carres - self.nb_visites * Q**2 + MCTS_SP.arguments["D"]
                           ) / self.nb_visites)
            return Q + U + T
    
    def selectioner(self):
        meilleur_noeud_fils = None
        ucts_max = -np.inf
        
        for noeud_fils in self.liste_fils:
            ucts = noeud_fils.calculerUCTScore()
            if ucts == np.inf:
                return noeud_fils
            elif ucts_max < ucts:
                meilleur_noeud_fils = noeud_fils
                ucts_max = ucts
        
        return meilleur_noeud_fils
    
    def expandre(self, jeu, politique):
        for branche in self.liste_branches_parcourable:
            jeu.changerDeConfiguration(self.etat, self.joueur, self.info)
            jeu.jouerLaProchainneAction(branche)
            proba = politique[jeu.dico_indexAction[branche]]
            noeud_fils = NoeudAlpha(jeu, self, branche, self.profondeur+1, proba)
            self.liste_fils.append(noeud_fils)
    
    def simuler(self, jeu, reseau):
        jeu.changerDeConfiguration(self.etat, self.joueur, self.info)
        etat_encode = jeu.encoderEtatPourReseau()
        etat_encode = np.expand_dims(etat_encode, axis=0)
        Y = reseau.evaluer(etat_encode)
        if len(Y) == 2:
            valeur, politique = Y[0], Y[1]
        else:
            valeur, politique = Y[0,0], Y[0,1:]
        valeur = valeur.item()
        politique = softmax(politique)
        politique *= jeu.vectoriserMouvementsValides(
            self.liste_branches_parcourable)
        if sum(politique) != 0:
            politique /= sum(politique)
        else:
            print("ATTENTION : Division par zéro, Car politique nul !")
            pass
        return valeur, politique
    
    def retropropager(self, resultat):
        self.score_moyen += resultat
        self.nb_visites += 1
        self.somme_scores_carres += resultat**2
        
        if not self.estRacine():
            parent = self.noeud_parent
            parent.retropropager(resultat)


class MCTS_SP: # Monte Carlo Tree Search (Recherche Arborescente de Monte Carlo en français) pour le AlphaZero
    
    arguments = {"C": math.sqrt(2), "D": 0, "nb_recherches": 1000, 
        "profondeur_max": np.inf}
    
    def __init__(self, jeu, reseau):
        self.jeu = jeu
        self.racine = NoeudAlpha(jeu)
        self.reseau = reseau
    
    def rechercheArborescente(self):
        racine = self.racine
        for _ in range(MCTS_SP.arguments["nb_recherches"]):
            noeud = racine
            while not noeud.estFeuille():
                noeud = noeud.selectioner()
            valeur, politique = noeud.simuler(self.jeu, self.reseau)
            if noeud.profondeur < MCTS_SP.arguments["profondeur_max"] and \
                not noeud.estTerminal():
                noeud.expandre(self.jeu, politique)
            noeud.retropropager(valeur)

    def analyserRacine(self):
        valeur = self.racine.score_moyen / self.racine.nb_visites
        distribution = np.array([noeud_fils.nb_visites
            for noeud_fils in self.racine.liste_fils], dtype=np.float32)
        distribution /= self.racine.nb_visites -1
        return valeur, distribution

    def obtenirMeilleurFils(self):
        valeur, distribution = self.analyserRacine()
        i = np.argmax(distribution)
        noeud_fils =  self.racine.liste_fils[i]
        return noeud_fils

    def parcourirRacine(self, branche):
        i = self.racine.liste_branches_parcourable.index(branche)
        noeud_fils = self.racine.liste_fils[i]
        self.racine = noeud_fils

        self.racine.noeud_parent = None
        self.racine.branche_parente = None
        MCTS_SP.arguments["profondeur_max"] += self.racine.profondeur
