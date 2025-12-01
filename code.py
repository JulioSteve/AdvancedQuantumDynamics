import numpy as np
import sympy as sp
import math

J,U = sp.symbols('J U')

def fill_sites(N,L):
    Sets = []
    # Error management
    if N<0:
        raise ValueError("N needs to be positive, it is the number of particles.")
    if L<=0:
        raise ValueError("L needs to be strictly positive, it is the number of available sites.")
    
    # Limit cases
    if N==0:
        Sets.append([0]*L)
        return Sets
    if L==1:
        Sets.append([N])
        return Sets
    
    # Recursion we put k particles in the first site, hence we treat the N-k on the L-1 other sites by recurrence
    for k in range(N+1):
        for others in fill_sites(N-k, L-1):
            Sets.append([k] + others)   

    return Sets

def c(Set, i):
    Set[i] += 1
    #return the coefficient and the set
    return np.sqrt(Set[i]), Set 

def a(Set, i):
    if Set[i]==0:
        print("WARNING: you have applied annihilation to vacuum state and killed it.")
        return 0,0
    coef = Set[i]
    Set[i] -= 1
    #return the coefficient and the set
    return np.sqrt(coef), Set 

def jump_H(state):
    pass

###########################################################################################################################
N = 3
L = N
###########################################################################################################################
print(f"\nThere are {math.comb(N+L-1,N)} possible states to distribute {N} particles in {L} sites.")
print(f"{fill_sites(N,L)}\n")