import numpy as np
import math
import matplotlib.pyplot as plt


## Prepare the distribution of N particles in L sites 

def distribute(N, L):
    ''' Distributes N indistiguishable atoms in L traps. Returns a list of the possible distributions. (A distribution is itself a list of L non-negative integers whith sum N.) '''
    if N == 0:
        return [[0,]*L]
    elif L == 1:
        return [[N]]
    else:
        return one_in_first(distribute(N-1, L)) + zero_in_first(distribute(N, L-1))

def one_in_first(l):
    ''' For all distributions in l, add one atom in the first trap. '''
    L = len(l[0])
    return [list(np.array([1] + [0]*(L-1)) + np.array(e)) for e in l]

def zero_in_first(l):
    ''' For all distributions in l, add a new empty trap. '''
    return [[0] + e for e in l]



## Dimension of Hilbert space
def dim_of_hilbert_space(N,L):
    ''' Returns the dimension of the Hilbert space for N particles in L traps '''
    return int(math.factorial(N+L-1)/(math.factorial(N)*math.factorial(L-1)))


## Prepare subfunctions needed for hops
def roundloop(k,L):
    ''' acts as a modulo function: k modulo L; if k==L, returns 0; etc '''
    if k<0:
        return k+L
    elif k>=L:
        return k-L
    else:
        return k

def lefthop(k, L):
    ''' Returns an array of dimension L, with -1 at position k, and +1 at position k-1 '''
    tab=np.zeros(L,dtype=int)
    tab[k]=-1
    tab[roundloop(k-1,L)]=+1
    return tab

def righthop(k,L):
    ''' Returns an array of dimension L, with -1 at position k, and +1 at position k+1 '''
    tab=np.zeros(L,dtype=int)
    tab[k]=-1
    tab[roundloop(k+1,L)]=+1
    return tab

def droppart(j,L):
    ''' Returns an array of dimension L, with -1 at position j '''
    tab=np.zeros(L,dtype=int)
    tab[j]=-1
    return tab

## Find the position of a column in a 2D array
def find_list_pos(arraylist,testlist):
    ''' returns the index k of arraylist for which arraylist[k]==testlist'''
    for k in range(len(arraylist)):
        if np.all(arraylist[k]==testlist):
            return k
            break

## Generate the Hamiltonian matrix
def generate_hmat(ju,N):
    ''' Returns the expression of the hamiltonian matrix, calculated on the basis geerated by distribute function, for a J/U=ju parameter, for N particles on N sites '''
    dim_hilbert=dim_of_hilbert_space(N,N)
    basis=np.array(distribute(N,N))
    hmat=np.zeros((dim_hilbert,dim_hilbert))
    for nbvec in range(dim_hilbert):        #take one vector of the Hilbert basis
        hmat[nbvec,nbvec]+=0.5*sum(basis[nbvec]*(basis[nbvec]-1)) #on-site interaction
        for k in range(N):                  #hopping interaction, from all sites k
            if basis[nbvec,k]>0:            #only if there is a particle
                newvecleft=basis[nbvec]+lefthop(k,N)    #hop towards the left: new hilbert vector
                newvecright=basis[nbvec]+righthop(k,N)  #hop towards the right: new hilbert vector
                jl=find_list_pos(basis,newvecleft)
                jr=find_list_pos(basis,newvecright)
                hmat[nbvec,jl]+= - ju * math.sqrt(basis[nbvec,k]) * math.sqrt(basis[nbvec,roundloop(k-1,N)] + 1)    #hopping conribution
                hmat[nbvec,jr]+= - ju * math.sqrt(basis[nbvec,k]) * math.sqrt(basis[nbvec,roundloop(k+1,N)] + 1)    #same
    return hmat

print(generate_hmat(1.,3))

def find_ground_state(ju,N):
    ''' Returns the energy and the (normalized) ground state decomposition '''
    evals, evecs = np.linalg.eigh(generate_hmat(ju,N))
    mineigen=np.argmin(evals)
    gs=evecs[:,mineigen]
    return evals[mineigen], gs


def calculate_density_matrix(gs,N,basis):
    ''' Returns the density matrix, calculated with ground state gs for a system with N particles, using basis as basis of Hilbert space '''
    densitymat=np.zeros((N,N))
    dim_hilbert=len(gs)
    for i in range(N):
        for j in range(N):
            coefs=np.zeros((dim_hilbert,dim_hilbert))
            for k in range(dim_hilbert):
                for l in range(dim_hilbert):
                    if np.all(basis[k]+droppart(j,N)==basis[l]+droppart(i,N)):
                        delta=1
                    else:
                        delta=0
                    coefs[k,l]=gs[k]*gs[l] * math.sqrt(basis[k,j]*basis[l,i])*delta
            densitymat[i,j]=np.sum(coefs)
    return densitymat


def calculate_dfluc(gs,N,basis):
    ''' Returns the density fluctuations, calculated with ground state gs for a system with N particles, using basis as basis of hilbert space '''
    dim_hilbert=len(gs)
    n1sqall=np.zeros(dim_hilbert)
    n1all=np.zeros(dim_hilbert)
    for k in range(dim_hilbert):
        n1sqall[k]=gs[k] * gs[k] * basis[k,0] * basis[k,0]
        n1all[k]=gs[k]*gs[k]*basis[k,0]
    n1sq=sum(n1sqall)
    n1=sum(n1all)
    return n1sq - n1 * n1



def calculate_observables(ju,N):
    ''' Returns the following observables: energy, condensed fraction and density fluctuations  '''
    energy,gs = find_ground_state(ju, N)
    basis = np.array(distribute(N, N))
    densitymat = calculate_density_matrix(gs, N, basis)
    valdens = np.linalg.eigvalsh(densitymat)
    dfluc = calculate_dfluc(gs, N, basis)
    fc= np.max(valdens)/N
    return energy,fc,dfluc

npoints=21
jutab = np.linspace(0,0.4,npoints,endpoint=True)
energytab = np.zeros(npoints)
fctab = np.zeros(npoints)
dfluctab = np.zeros(npoints)

for ndim in range(1,6):
    print(ndim)
    for k in range(len(jutab)):
        ju=jutab[k]
        energytab[k],fctab[k],dfluctab[k] = calculate_observables(ju, ndim)
    plt.figure(1)
    plt.plot(jutab,fctab, label='N = '+str(ndim))
    plt.figure(2)
    plt.plot(jutab,dfluctab, label='N = '+str(ndim))

plt.figure(1)
plt.legend(loc='best')
plt.xlabel('J/U', style='italic')
plt.ylabel('condensed fraction')


plt.figure(2)
plt.legend(loc='best')
plt.xlabel('J/U', style='italic')
plt.ylabel('density fluctuations')

plt.show()


#energy, fc , dfluc = calculate_observables(0.5,2)    
#print(energy)
#print(fc)
#print(dfluc)

