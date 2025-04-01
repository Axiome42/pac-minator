import os
import matplotlib.pyplot as plt

import tensorflow as tf

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(tf.config.list_physical_devices('CPU'))
print(tf.config.list_physical_devices('GPU'))


def smooth_curve(Y: list, factor=0.7) -> list:
    smoothed_Y = []
    for point in Y:
        if smoothed_Y:
            smoothed_Y.append(smoothed_Y[-1]*factor + point*(1-factor))
        else:
            smoothed_Y.append(point)
    return smoothed_Y

def bloc_residuel(x, nb_filtres):
    """
    Un bloc résiduel classique :
    - 2 convolutions successives (motif=3, padding='same')
    - chacune suivie d'une BatchNormalization et d'une ReLU
    - puis on ajoute l'entrée x à la sortie (connexion résiduelle ou "skip connection")
    """
    y = tf.keras.layers.Conv2D(
        nb_filtres, 
        ResNet.hyperparametres["motif"], 
        padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    y = tf.keras.layers.Conv2D(
        nb_filtres, 
        ResNet.hyperparametres["motif"], 
        padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    # Connexion résiduelle : addition de l'entrée x
    sortie = tf.keras.layers.Add()([x, y])
    sortie = tf.keras.layers.ReLU()(sortie)
    
    return sortie


class ResNet:

    hyperparametres = {
        "motif" : 3,
        "pas" : 1e-3, 
        "epoque" : 10, 
        "taille_paquet" : 32
    }

    def __init__(self, jeu, nom=f"ResNet", cree_nouveau_dossier=True):
        self.jeu = jeu

        # Crée un répertoire spécifique au réseau
        chemin_sauvegarde = os.path.join(os.path.dirname(__file__), 'Sauvegarde')
            
        if cree_nouveau_dossier:
            if os.path.exists(f"{chemin_sauvegarde}/{nom}"):
                i = 1
                while os.path.exists(f"{chemin_sauvegarde}/{nom}_{i}"):
                    i += 1
                self.nom = f"{nom}_{i}"
            else:
                self.nom = nom
            os.mkdir(f"{chemin_sauvegarde}/{self.nom}")

            # Sauvegarde les Paramètres
            motif = ResNet.hyperparametres["motif"]
            epoque = ResNet.hyperparametres["epoque"]
            taille_paquet = ResNet.hyperparametres["taille_paquet"]
            fichier = open(f"{chemin_sauvegarde}/{self.nom}/document.txt", 
                mode="w", encoding="utf-8")
            fichier.write("\n      ### Paramètres réseau ###\n\n" +
                f"motif         = {motif}\n" +
                f"epoque        = {epoque}\n" +
                f"taille_paquet = {taille_paquet}\n\n")
            fichier.close()
        
        else:
            assert os.path.exists(f"{chemin_sauvegarde}/{nom}"), "Le dossier réseau n'existe pas !"
            self.nom = nom

    def construire(self, nb_blocs_res=6, nb_filtres=128):
        """
        Paramètres entrées:
        - dim_entree   : dimensions de l'état en entrée (H x W x Canaux)
        - nb_actions   : nombre de coups/actions possible à prédire
        - nb_blocs_res : nombre de blocs résiduels
        - nb_filtres   : nombre de filtres (canaux) pour les convolutions

        Construit un réseau de type AlphaZero :
        - Tronc commun : Conv + BatchNormalization + ReLU, suivi de blocs résiduels
        - Tête de politique
        - Tête de valeur
        """

        # 0) Sauvegarde les Paramètres
        dim_entree = self.jeu.dim_entree_reseau
        nb_actions = self.jeu.dim_sortie_reseau
        chemin_sauvegarde = os.path.join(os.path.dirname(__file__), 'Sauvegarde')
        fichier = open(f"{chemin_sauvegarde}/{self.nom}/document.txt", 
            mode="a", encoding="utf-8")
        fichier.write(
            f"dim_entree    = {dim_entree}\n" +
            f"nb_actions    = {nb_actions}\n" +
            f"nb_blocs_res  = {nb_blocs_res}\n" +
            f"nb_filtres    = {nb_filtres}\n\n"
        )
        fichier.close()
        
        # 1) Couche d'entrée
        entrees = tf.keras.layers.Input(shape=dim_entree)
        
        # Convolution initiale
        x = tf.keras.layers.Conv2D(
            nb_filtres, 
            kernel_size=ResNet.hyperparametres["motif"], 
            padding='same')(entrees)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        # 2) Empilement de blocs résiduels
        for _ in range(nb_blocs_res):
            x = bloc_residuel(x, nb_filtres)
        
        # 3) Tête de politique (policy head)
        #    Convolution 1x1 pour réduire la profondeur, puis Dense → softmax
        p = tf.keras.layers.Conv2D(filters=2, kernel_size=1, padding='same')(x)
        p = tf.keras.layers.BatchNormalization()(p)
        p = tf.keras.layers.ReLU()(p)
        p = tf.keras.layers.Flatten()(p)
        p = tf.keras.layers.Dense(nb_actions)(p)  # Logits
        politique_sortie = tf.keras.layers.Softmax(name='politique')(p)
        
        # 4) Tête de valeur (value head)
        #    Convolution 1x1, puis Dense → tanh
        v = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(x)
        v = tf.keras.layers.BatchNormalization()(v)
        v = tf.keras.layers.ReLU()(v)
        v = tf.keras.layers.Flatten()(v)
        v = tf.keras.layers.Dense(256, activation='relu')(v)
        v = tf.keras.layers.Dense(1)(v)
        valeur_sortie = tf.keras.layers.Activation('tanh', name='valeur')(v)
        
        # 5) Construction du modèle
        reseau = tf.keras.Model(
            inputs=entrees, 
            outputs=[politique_sortie, valeur_sortie], 
            name='ResNetAlphaZero')
        
        # 6) Compilation du modèle
        reseau.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=ResNet.hyperparametres["pas"]),
            loss={
                'politique': 'categorical_crossentropy',
                'valeur': 'mean_squared_error'
            },
            metrics={
                'politique': 'accuracy',
                'valeur': 'mse'
            }
        )

        self.reseau = reseau

    def evaluer(self, X):
        politique, valeur = self.reseau.predict(X, verbose=0)
        valeur = valeur[0, 0]
        politique = politique[0] 
        return valeur, politique

    def entrainer(self, BdD_X, BdD_V, BdD_P):
        historique = self.reseau.fit(
            x=BdD_X,
            y={
                'politique': BdD_P,
                'valeur': BdD_V
            },
            batch_size=ResNet.hyperparametres["taille_paquet"],
            epochs=ResNet.hyperparametres["epoque"]
        )
        return historique

    def sauvegarder(self):
        chemin_sauvegarde = os.path.join(os.path.dirname(__file__), 'Sauvegarde')
        self.reseau.save(f"{chemin_sauvegarde}/{self.nom}/reseau.keras")

    def importer(jeu, nom):
        chemin_sauvegarde = os.path.join(os.path.dirname(__file__), 'Sauvegarde')
        assert os.path.exists(f"{chemin_sauvegarde}/{nom}/reseau.keras"), "Le réseau à importer n'existe pas !"
        reseau_svg = tf.keras.models.load_model(f"{chemin_sauvegarde}/{nom}/reseau.keras")

        nouveauResNet = ResNet(jeu, nom=nom, cree_nouveau_dossier=False)
        nouveauResNet.reseau = reseau_svg
        return nouveauResNet

    def analyserEntrainement(self, historique):
        """ Trace les courbes de la précision et de la perte du réseau en fonction
        des époques """
        
        chemin_sauvegarde = os.path.join(os.path.dirname(__file__), 'Sauvegarde')

        liste_precisionValeur = historique.history["valeur_mse"]
        liste_precisionPolitique = historique.history["politique_accuracy"]
        liste_perte = historique.history["loss"]
        precisionValeur = liste_precisionValeur[-1]
        precisionPolitique = liste_precisionPolitique[-1]
        perte = liste_perte[-1]
        nb_epoques = len(liste_precisionValeur)
        liste_epoque = [i for i in range(1, nb_epoques + 1)]
        
        i = sum([1 if "graphique_perte" in nom else 0 
            for nom in os.listdir(f"{chemin_sauvegarde}/{self.nom}")])

        plt.plot(liste_epoque, smooth_curve(liste_precisionValeur), "bo", 
            label=f"Valeur = {round(precisionValeur, 3)}")
        plt.title("Courbe de l'éxactitude de la valeur pendant l'entrainement")
        plt.legend()
        plt.axis([0, nb_epoques, 0, 1])
        plt.savefig(f"{chemin_sauvegarde}/{self.nom}/graphique_precision_valeur_{i}.jpg")

        plt.figure()
        plt.plot(liste_epoque, smooth_curve(liste_precisionPolitique), "bo", 
            label=f"Politique = {round(precisionPolitique, 3)}")
        plt.title("Courbe de l'éxactitude de la politique pendant l'entrainement")
        plt.legend()
        plt.axis([0, nb_epoques, 0, 1])
        plt.savefig(f"{chemin_sauvegarde}/{self.nom}/graphique_precision_politique_{i}.jpg")
        
        plt.figure()
        plt.plot(liste_epoque, smooth_curve(liste_perte),"bo", 
            label=f"Perte = {round(perte, 3)}")
        plt.title("Courbe de la perte pendant l'entrainement et la validation")
        plt.legend()
        plt.savefig(f"{chemin_sauvegarde}/{self.nom}/graphique_perte_{i}.jpg")
        
        return (precisionValeur, precisionPolitique, perte)
