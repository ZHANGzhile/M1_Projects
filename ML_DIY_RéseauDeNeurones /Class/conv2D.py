import numpy as np
from module import Module

class Conv2D(Module):
    def __init__(self, v_size, h_size, chan_in, chan_out, v_stride, h_stride):
        super().__init__()
        self.v_size = v_size
        self.h_size = h_size
        self.v_stride = v_stride
        self.h_stride = h_stride
        self._parameters = np.random.rand(v_size , h_size, chan_in, chan_out) * 0.01
        self._gradient = {"weight": np.zeros_like(self._parameters), "bias": None}
        

    def forward(self,X):
        """
        param X: np.array shape (batch_size, hauteur, largeur, nb_canaux)

        cette fonction est presque identique a la fonction forward de la classe Conv1D, sauf qu'on a une dimension en plus,
        donc on a besoin de remplacer k_size par v_size * h_size et stride par v_stride, h_stride
        """
        
        new_ = np.lib.stride_tricks.as_strided(X,
                                                shape=( X.shape[0], 
                                                        int((X.shape[1] - self.v_size) / self.v_stride + 1), # hauteur de sortie
                                                        int((X.shape[2] - self.h_size) / self.h_stride + 1), # largeur de sortie                                         
                                                        self._parameters.shape[0], # hauteur du noyau
                                                        self._parameters.shape[1], # largeur du noyau
                                                        self._parameters.shape[2], # nombre de canaux d'entrée
                                                ),
                                                strides=(X.strides[0],
                                                        X.strides[1]*self.v_stride,
                                                        X.strides[2]*self.h_stride,
                                                        X.strides[1],
                                                        X.strides[2],
                                                        X.strides[3],
                                                        ),
                                                writeable=False, 
                                                )
        
        """
        b - Taille du batch
        y - Indice vertical dans le noyau de convolution
        x - Indice horizontal dans le noyau de convolution
        v - La position du noyau de convolution dans la direction de la vertical
        h - La position du noyau de convolution dans la direction de la horizontal
        c - Nombre de canaux d'entrée
        o - Nombre de canaux de sortie
        """
        
        return np.einsum(
                'byxvhc,vhco->byxo',
                new_,
                self._parameters,
            )
    
    
    def backward_delta(self,X,delta):
        """
        param X: np.array shape (batch_size, hauteur, largeur, nb_canaux)
        param delta: np.array shape (batch_size, hauteur, largeur, nb_canaux)

        cette fonction est presque identique a la fonction backward_delta de la classe Conv1D, sauf qu'on a une dimension en plus,
        donc on a besoin de deux indexes verticaux et horisontaux pour pouvoir faire la multiplication avec le delta
        
        """

        hauteur =delta.shape[1]
        largeur =delta.shape[2]

        res=np.zeros_like(X)
                
        new_ = np.einsum('byxo,vhco->byxvhc',delta,self._parameters)
        new_flat = new_.flatten()


        start_v_indices = np.arange(hauteur) * self.v_stride
        start_h_indices = np.arange(largeur) * self.h_stride

        # np.add.outer : une fonction qui fait la somme de chaque élément de deux vecteurs
        # on crée une grille de start_v_indices et np.arange(self.v_size) et on les somme 
        # on fait la même chose pour start_h_indices et np.arange(self.h_size)
        full_v_indices = np.add.outer(start_v_indices, np.arange(self.v_size)).flatten() # c'est un vecteur de taille v_size * hauteur
        full_h_indices = np.add.outer(start_h_indices, np.arange(self.h_size)).flatten() # c'est un vecteur de taille h_size * largeur

        # 扩展索引以适应所有批次和通道
        v_indexes = np.tile(full_v_indices, X.shape[3] * self.h_size * hauteur * X.shape[0])
        h_indexes = np.tile(full_h_indices, X.shape[3] * self.v_size * largeur * X.shape[0])
        
 
        batch_indices = np.repeat(np.arange(X.shape[0]), len(v_indexes) // X.shape[0])
        channel_indices = np.tile(np.arange(X.shape[3]), len(h_indexes) // X.shape[3])


        np.add.at(res, (batch_indices, v_indexes, h_indexes, channel_indices), new_flat)
        
        return res
    
    def backward_update_gradient(self, X, delta):
        """
        idem que la fonction backward_update_gradient de la classe Conv1D
        """
        new_=np.lib.stride_tricks.as_strided(X,
                                             shape=(X.shape[0],
                                                    int((X.shape[1]-self.v_size)/self.v_stride + 1),
                                                    int((X.shape[2]-self.h_size)/self.v_stride + 1),
                                                    X.shape[3],
                                                    self.v_size,
                                                    self.h_size
                                                    ),      
                                            strides=(X.strides[0],
                                                    X.strides[1]*self.v_stride,
                                                    X.strides[2]*self.h_stride,
                                                    X.strides[3],
                                                    X.strides[1],
                                                    X.strides[2]
                                                    )
                                            )
        gradient_update=np.einsum('byxo,byxcvh->vhco',delta,new_)
        self._gradient["weight"] += gradient_update
    
    def update_parameters(self, learning_rate=1e-4):
        self._parameters -= learning_rate * self._gradient["weight"]
        
    def zero_grad(self):
        self._gradient = {"weight": np.zeros_like(self._parameters), "bias": None}



class AvgPool2D(Module):
    def __init__(self, v_size, h_size, v_stride, h_stride):
        super().__init__()
        self.v_stride = v_stride
        self.h_stride = h_stride
        self.v_size = v_size
        self.h_size = h_size

    def forward(self, X:np.array):
        """
        idem que la fonction forward de la classe MaxPool1D sauf qu'on remplace np.max par np.mean
        """

        X=np.lib.stride_tricks.as_strided(X,
                                          shape=(X.shape[0],
                                                int((X.shape[1]-self.v_size)/self.v_stride + 1),
                                                int((X.shape[2]-self.h_size)/self.h_stride + 1),
                                                X.shape[3],
                                                self.v_size,
                                                self.h_size
                                                ),      
                                           strides=(X.strides[0],
                                                    X.strides[1]*self.v_stride,
                                                    X.strides[2]*self.h_stride,
                                                    X.strides[3],
                                                    X.strides[1],
                                                    X.strides[2]
                                                    )
                                            )
        
        return np.mean(X,axis=(4,5)) # on prend la moyenne sur les deux dernieres dimensions
    
    
    def backward_delta(self,X,delta):
                            
        expanded_delta = np.repeat(delta, self.v_size, axis=1)
        expanded_delta = np.repeat(expanded_delta, self.h_size, axis=2)

        # 因为前向传播是取平均，所以反向传播时将梯度均匀分布到每个元素
        expanded_delta /= (self.v_size * self.h_size)

        # 调整 expanded_delta 的尺寸以匹配输入 X 的形状
        # 注意：这里假设了步长和池化窗口大小导致的维度缩减，没有考虑填充和不完整窗口的情况
        # 如果池化操作中包含填充或步长导致的不完整窗口，需要调整下述代码以正确处理边界情况
        final_delta = expanded_delta[:, :X.shape[1], :X.shape[2], :]
        #print(final_delta.shape)
        return final_delta
    
    def update_parameters(self, gradient_step:float = 0.001):
        return None

    def backward_update_gradient(self, X:np.array, delta:np.array):
        pass

    def zero_grad(self):
        pass

class Flatten2D(Module):
    """
    Tout comme la classe Flatten1D, cette classe est utilisée pour transformer un tenseur 2D en un tenseur 1D
    """
    def __init__(self):
        super().__init__()

    def forward(self, X:np.array):
        batch, hauteur, largeur, chan_in = X.shape
        return X.reshape(batch, hauteur * largeur * chan_in )
    
    def backward_delta(self,X:np.array,delta:np.array):

        batch, hauteur, largeur, chan_in = X.shape
        return delta.reshape(batch, hauteur, largeur, chan_in)
    
    def update_parameters(self, gradient_step):
        return None

    def backward_update_gradient(self, X, delta):
        pass

    def zero_grad(self):
        pass