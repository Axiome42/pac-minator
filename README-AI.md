# AlphaZero

## Installer l’environnement :

### Création de l'environnement
```
conda create -p "C:\Users\arthu\Desktop\Club_IA\Cours_3-CrossyRoad\CrossyRoad\env_conda"
```

### Activation de l'environnement
```
conda activate C:\Users\arthu\Desktop\Club_IA\Cours_3-CrossyRoad\CrossyRoad\env_conda
```

### Téléchargement des librairie nécessaire
```
python -m pip install --upgrade pip
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install tensorflow<2.11
pip install numpy<2
pip install matplotlib tqdm
```

### Teste l'instalation de l'environement
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
