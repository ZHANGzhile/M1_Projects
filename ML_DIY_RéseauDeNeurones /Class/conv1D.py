
import numpy as np
from module import Module



class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):

        super().__init__()
        self.k_size = k_size
        self.stride = stride
        # ici on initialise les paramètres directement en utilisant la taille du noyau, le nombre de canaux d'entrée et de sortie, pas dictionnaire, car pas de biais
        self._parameters = np.random.rand(k_size, chan_in, chan_out) * 0.01
        # on initialise le gradient à zéro toujours en type dictionnaire
        self._gradient = {"weight": np.zeros_like(self._parameters), "bias": None}

    def forward(self,X): # X est de taille (batch, d, canaux d'entrée)
        assert X.shape[2]==self._parameters.shape[1] # on verifie que le nombre de canaux d'entrée de X est égal au nombre de canaux d'entrée des paramètres
        
        """
        np.lib.stride_tricks.as_strided : Crée une vue sur une matrice en utilisant une nouvelle forme et un nouveau pas.

        On utilise np.lib.stride_tricks.as_strided pour créer une vue sur la matrice X en utilisant une nouvelle forme et un nouveau pas, 
        ce qui permet de générer des fenêtres de convolution à partir de l'entrée X.
        Cette sortie est la forme: 
        """
        new_ = np.lib.stride_tricks.as_strided(X,
                                                    shape=(X.shape[0], # Taille du batch
                                                           int((X.shape[1] - self.k_size) / self.stride + 1), # Longueur de sortie: calculée à partir de la taille du noyau, du pas (stride) et de la longueur de l'entrée
                                                           self._parameters.shape[0], # Nombre de canaux de sortie: correspond au nombre de noyaux de convolution, chaque noyau produisant un canal de sortie
                                                           self._parameters.shape[1], # Nombre de lignes par noyau: pour une convolution 1D, cela représente généralement le nombre de canaux d'entrée que chaque noyau couvre
                                                    ),  
                                                    strides=(X.strides[0],  # Pas pour la dimension du batch: le nombre d'octets pour passer à l'échantillon complet suivant
                                                             X.strides[1] * self.stride,  # Pas pour la dimension de longueur : ajusté selon le pas de convolution, saute 'stride' éléments à chaque fois
                                                             X.strides[1],  # Pas pour la dimension du noyau : correspond au pas pour passer d'un élément au suivant dans la même fenêtre de convolution
                                                             X.strides[2],  # Pas pour la dimension des canaux d'entrée : permet de passer d'un canal d'entrée au suivant au sein d'une même position du noyau
                                                    ),
                                                    writeable=False,) # Rend le tableau non modifiable pour éviter des modifications accidentelles qui pourraient affecter les données d'origine
        
        """
        np.einsum : Implémentation de la somme de produits tensoriels, utilisée pour effectuer la multiplication matricielle entre les données d'entrée et les poids des noyaux de convolution

        bwkc : représente les quatre dimensions de new_input :
            b - Taille du batch
            w - Indice de la fenêtre dans la carte de caractéristiques de sortie
            k - Indice du noyau de convolution
            c - Nombre de canaux d'entrée, utilisé pour multiplier par les poids correspondants de chaque noyau de convolution
        
        kco : représente les trois dimensions de self._parameters :
            k - pareil que ci-dessus
            c - pareil que ci-dessus
            o - Nombre de canaux de sortie, chaque noyau de convolution générant un canal de sortie

        bwo : représente les trois dimensions de la sortie de la convolution : (batch, (d-k_size)/stride +1, canaux de sortie)

        """
        return np.einsum('bwkc,kco->bwo', new_, self._parameters,)
    
    
    
    def backward_delta(self,X,delta):

        length = delta.shape[1]
        #print("NaN in delta:", np.isnan(delta).any())
        #print("NaN in parameters:", np.isnan(self._parameters).any())

        res = np.zeros_like(X)

        # on calcule la nouvelle forme de X pour correspondre à la forme de delta
        new_ = np.einsum('bwo,kco->bwkc', delta, self._parameters)
        new_flat = new_.flatten()


        # on calcule la position de départ de chaque fenêtre de pooling dans l'entrée originale en multipliant l'indice de la valeur maximale par le pas
        start_win = np.arange(length) * self.stride
        # on crée un index conv complet pour chaque fenêtre
        full_window_indices = start_win[:, None] + np.arange(self.k_size)

        expanded_indices = np.tile(full_window_indices.flatten(), X.shape[2])  #repete pour chaque canal
        batch_expanded_indices = np.tile(expanded_indices, X.shape[0])  # repete pour chaque batch
        indexes = batch_expanded_indices

        # Calcul des indices du batch. Répète chaque indice du batch un nombre de fois égal pour correspondre au nombre total d'indices de fenêtres de conv.
        # Cela garantit que chaque valeur de gradient est correctement associée à son batch correspondant.        
        batch_indices = np.repeat(np.arange(X.shape[0]), len(indexes) // X.shape[0]) 
        # Calcul des indices de canal. Répète chaque indice de canal un nombre de fois égal pour correspondre au nombre total d'indices de fenêtres de conv.
        channel_indices = np.tile(np.arange(X.shape[2]), len(indexes) // X.shape[2])


        # Utilisation de np.add.at pour accumuler les gradients dans le tableau de résultats res. Cette fonction met à jour directement le tableau res sur place,
        # en accumulant les valeurs de gradient de helper.flatten() aux positions spécifiées par batch_indices, indexes et channel_indices.
        # Cela permet de transférer les gradients calculés lors de la rétropropagation aux positions appropriées du tenseur d'entrée X, complétant ainsi le processus de propagation du gradient.
        np.add.at(res, (batch_indices, indexes, channel_indices), new_flat)

        return res
    
    def backward_update_gradient(self, X, delta):
       
        new_=np.lib.stride_tricks.as_strided(X,shape=(
            X.shape[0],
            int((X.shape[1]-self.k_size)/self.stride + 1),
            X.shape[2], # nombre de canaux d'entrée
            self.k_size), # taille du noyau        
            strides=(
            X.strides[0],
            X.strides[1]*self.stride,
            X.strides[2], # pas pour les canaux d'entrée
            X.strides[1])) # pas pour la taille du noyau
        
        gradient_update = np.einsum('bwo,bwck->kco', delta, new_)
        self._gradient['weight'] += gradient_update
    
    def update_parameters(self, learning_rate=1e-4):
        self._parameters -= learning_rate * self._gradient["weight"]
        
    def zero_grad(self):
        self._gradient = {"weight": np.zeros_like(self._parameters), "bias": None}





class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self.stride = stride
        self.k_size = k_size

    def forward(self, X):

        newX = np.lib.stride_tricks.as_strided(X,
                                            shape=( X.shape[0],
                                                    int((X.shape[1]-self.k_size)/self.stride + 1),
                                                    X.shape[2],
                                                    self.k_size),                 
                                            strides=(X.strides[0],
                                                     X.strides[1]*self.stride,
                                                     X.strides[2],
                                                     X.strides[1]))
        

        return np.max(newX,axis=3)
    
    def backward_delta(self,X,delta):
        # delta: (batch,(lenght-k_size)/stride+1,channels )


        batch_size, length, channels = X.shape
        output_length = int((length - self.k_size) / self.stride + 1)
        # on calcule la position de départ de chaque fenêtre de pooling dans l'entrée originale en multipliant l'indice de la valeur maximale par le pas
        start_pos = np.arange(output_length) * self.stride

        res = np.zeros_like(X) #(batch, lenght, channels)
        
        # on crée une vue sur la matrice en utilisant une nouvelle forme et un nouveau pas 
        newX = np.lib.stride_tricks.as_strided( X,
                                                shape=(batch_size, output_length, channels, self.k_size),
                                                strides=(X.strides[0], X.strides[1] * self.stride, X.strides[2], X.strides[1])
                                                )
        
        # on trouve l'indice de la valeur maximale de chaque fenêtre de pooling
        indx = np.argmax(newX, axis=3)
        
        # on calcule les coordonnées de départ de chaque fenêtre de pooling
        new_start_pos = np.lib.stride_tricks.as_strided(start_pos,
                                                        shape=indx.shape,
                                                        strides=(0, start_pos.strides[0], 0)
                                                        )
        
        # on ajoute les coordonnées de départ à l'indice pour obtenir l'indice global
        indx += new_start_pos
        
        # on ajoute les gradients aux positions des valeurs maximales
        for i in range(batch_size):
            for j in range(channels):
                # np.add.at permet d'ajouter les gradients aux positions des valeurs maximales
                np.add.at(res[i, :, j], indx[i, :, j], delta[i, :, j]) 
        
        return res
    
    def update_parameters(self, gradient_step):
        return None

    def backward_update_gradient(self, X, delta):
        pass

    def zero_grad(self):
        pass


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        batch, lenght, chan_in = X.shape
        return X.reshape(batch, lenght * chan_in) # (batch, lenght * chan_in)
    
    def backward_delta(self,X,delta):
        batch, lenght, chan_in = X.shape
        
        return delta.reshape(batch, lenght, chan_in)
    
    def update_parameters(self, gradient_step):
        return None

    def backward_update_gradient(self, X, delta):
        pass

    def zero_grad(self):
        pass