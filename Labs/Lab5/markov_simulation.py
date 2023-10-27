import numpy as np

def markov(rho, A, nmax, rng):
    '''
    Fonction qui génère une chaine de markonv à nmx itérations, à partir de la densité rho, de matrice de cheminement A et rng un génératuer de nombre aléatoire
    '''
    #checks
    assert A.shape[0]==A.shape[1],"Erreur: merci de renseigner une matrice carrée"
    assert rho.shape[0]==A.shape[1], "Erreur: les lignes de A doivent etre de meme taille que rho"
    assert np.sum(rho, axis=0)==1, "Erreur, la somme des éléments de rho doit faire 1"
    assert np.sum(np.sum(A, axis=0),axis=0)==rho.shape[0],"Erreur: les lignes de la matrice doivent sommer à N"

    #initialisation de la chaine:
    traj=np.zeros(nmax)
    states=np.arange(1,rho.shape[0]+1)
    traj[0] = int(rng.choice(states, p=rho))
    
    for i in range(1,nmax):
        traj[i]=rng.choice(states, p=A[int(traj[i-1])-1]) 
    return traj

def run_markov(args):
    return markov(*args)