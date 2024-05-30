import numpy as np
import random
from matplotlib import pyplot as plt

def generateRandomInstances(n,k):
    if not type(n) is int or not type(k) is int:
        raise TypeError("Seulement des entiers sont permis") 
    if k <= 0 or k>5:
        raise Exception("Le nombre de robots doit être un entier compris entre 1 et 5") 
    if n<8:
        raise Exception("La taille du plateau doit être supérieure ou égale à 8") 
    if n%2!=0:
        raise Exception("La taille du plateau doit être paire") 
        
    cellules = np.zeros((n,n), dtype=int)
    verticaux = np.zeros((n,n-1), dtype=int)
    horizontaux = np.zeros((n-1,n), dtype=int)
    
    #on place des murs verticaux sur la première et dernière ligne (on en place n/4 aléatoirement)
    
    nb=n//4
    
    lv=random.sample(range(0,n-1), nb)
    for i in range(nb):
        verticaux[0,lv[i]]=1
        
    lv=random.sample(range(0,n-1), nb)  
    for i in range(nb):
        verticaux[n-1,lv[i]]=1 
        
    #on place des murs horizontaux sur la première et dernière colonne (on en place n/4 aléatoirement)
    
    lv=random.sample(range(0,n-1), nb)
    for i in range(nb):
        horizontaux[lv[i],0]=1
        
    lv=random.sample(range(0,n-1), nb)
    for i in range(nb):
        horizontaux[lv[i],n-1]=1
        
    
    #on place les doubles murs. On en place 1 pour 1 sous-ensemble de carré de 4 cellules sur deux (sans compter les lignes et colonnes du bord)
        
    #on parcourt les coins supérieurs gauches de chacun des carrés de 4 cellules
    
    l = [i for i in range(1,n-2) if (i-1)%4==0]
    
    for  i in l:
        for j in l:
            choixh=random.randint(0,1)
            choixv=random.randint(0,1)
            choixa=random.randint(0,1)
            if choixa==0:
                horizontaux[i+choixv-1,j+choixh]=1
                verticaux[i+choixv,j+choixh-1]=1
            else:
                horizontaux[i+choixv,j+choixh]=1
                verticaux[i+choixv,j+choixh]=1
                    
    #on ajoute un ilot central

    horizontaux[n//2-2,n//2-1]=1
    horizontaux[n//2-2,n//2]=1
    horizontaux[n//2,n//2-1]=1
    horizontaux[n//2,n//2]=1
    verticaux[n//2-1,n//2-2]=1
    verticaux[n//2,n//2-2]=1
    verticaux[n//2-1,n//2]=1
    verticaux[n//2,n//2]=1

    #on place k robots au hasard (noté 1,2,...,k), mais pas dans l'ilot central. Le robot 1 sera le robot devant atteindre la cible but.

    for i in range(1,k+1):
        posok=False
        while not posok:
            abs=random.randint(0,n-1)
            ord=random.randint(0,n-1)
            #pas dans l'ilot central
            if not((abs==(n//2-1) and ord==(n//2-1)) or (abs==(n//2-1) and ord==(n//2)) or (abs==(n//2) and ord==(n//2-1)) or (abs==(n//2) and ord==(n//2))):
                if (cellules[abs,ord]==0):
                    cellules[abs,ord]=i
                    posok=True
        
    #choix d'une cible but

    posok=False
    it=1
    while not posok:
        butx=random.randint(1,n-2)
        buty=random.randint(1,n-2)
        #pas dans l'ilot central
        if not((butx==(n//2-1) and buty==(n//2-1)) or (butx==(n//2-1) and buty==(n//2)) or (butx==(n//2) and buty==(n//2-1)) or (butx==(n//2) and buty==(n//2))):
            if cellules[butx,buty]==0:  #un robot n'est pas sur la cible but
                #on vérifie que l'on est dans un "coin"
                if (((horizontaux[butx-1,buty]==1) or (horizontaux[butx,buty]==1)) and ((verticaux[butx,buty-1]==1) or (verticaux[butx,buty]==1))):
                    cellules[butx,buty]=-1
                    posok=True
        
    return cellules,verticaux,horizontaux
    
def showgrid(n,cellules,verticaux,horizontaux):         
    plt.grid()    

    plt.xlim([0, n])
    plt.ylim([0, n])
   
    plt.xticks(range(n+1))
    plt.yticks(range(n+1))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    scolor=["blue","black","green","yellow","red"]
 
    for i in range(n):
        for j in range(n):
            s=""
            if cellules[i][j]>0:
                s=s+'r'+str(cellules[i][j]) 
                plt.text(j,n-i-1,s, fontsize=220//n,color=scolor[cellules[i,j]-1])
            elif cellules[i][j]==-1:                
                plt.text(j,n-i-1,'t', fontsize=220//n,color='blue')
    
    for i in range(n):
        for j in range(n-1):
            if verticaux[i][j]==1:
                p1=[j+1,j+1]
                p2=[n-i,n-i-1]
                plt.plot(p1,p2,color='black',linewidth=3)
                
    for i in range(n-1):
        for j in range(n):
            if horizontaux[i][j]==1:
                p1=[j,j+1]
                p2=[n-i-1,n-i-1]
                plt.plot(p1,p2,color='black',linewidth=3)
                
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
                
    plt.show()
    #plt.savefig('exemple.png')







