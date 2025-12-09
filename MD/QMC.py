import numpy as np
import numpy.random as npr
import math
import matplotlib.pyplot as plt



def roundloop(k,L):
    ''' acts as a modulo function: k modulo L; if k==L, returns 0; etc '''
    if k<0:
        return k+L
    elif k>=L:
        return k-L
    else:
        return k




def generate_random_vec(N):
    ''' pick up randomly one vector from the Hilbert space basis '''
    listcases=np.array(range(2*N-1))
    wallspos=np.zeros(0,dtype=int)
    for i in range(N-1):                # draw the position of N-1 walls
        randomnum=npr.randint(0,len(listcases))
        wallspos=np.append(wallspos,int(listcases[randomnum]))
        listcases=np.delete(listcases,randomnum)
    wallspos=np.sort(wallspos)          #sort the walls positions
    basisvec=np.zeros(N,dtype=int)
    basisvec[0]=wallspos[0]             #nb of particles before 1st wall
    for i in range(1,N-1):
        basisvec[i]=wallspos[i]-wallspos[i-1]-1 #nb of particles between 2 walls
    basisvec[N-1]=2*N-1-wallspos[-1]-1  #nb of particles after last wall
    return basisvec



def accept_prob(si,sf,gpar):
    ''' calculates the probability to go from configuration si to configuration sf, given a parameter g = gpar '''
    N=len(si)
    sumn2i=sum(si*si)
    sumn2f=sum(sf*sf)
    ratiofactorial=np.ones(N)
    for j in range(N):
        if si[j]>sf[j]:
            ratiofactorial[j]=np.product(range(sf[j]+1,si[j]+1))
        elif si[j]<sf[j]:
            temp=np.zeros(sf[j]-si[j])
            for k in range(si[j]+1,sf[j]+1):
                temp[k-si[j]-1]=1./k
            ratiofactorial[j]=np.product(temp)
        else:
            pass
    probaratio=math.exp(-2*gpar*(sumn2f-sumn2i))*np.product(ratiofactorial)
    return np.min([1,probaratio])



def localenergy(sigma,gpar,ju):
    ''' evaluates the local energy of configuration sigma, with a parameter g = gpar and J/U=ju '''
    N=len(sigma)
    energypersite=np.zeros(N)
    for p in range(N):
        energypersite[p]=0.5 * sigma[p] * (sigma[p] - 1) \
                - ju * sigma[p] * math.exp(-2*gpar + 2 * gpar * sigma[p]) \
                *(math.exp(-2*gpar*sigma[roundloop(p-1,N)]) + math.exp(-2*gpar*sigma[roundloop(p+1,N)]))
    return sum(energypersite)


def energyg(gpar,ju,N,Niter):
    ''' evaluates the energy of the system in psi_gpar state, at N dimensions, for J/U=ju, with Niter iterations
    returns the evaluated average energy and the standard deviation of 10 subsets
    '''
    sigma = generate_random_vec(N)
    evalenergy = np.zeros(Niter)
    sbins = int(math.floor(Niter/10))
    for counter in range(Niter):
        sigmanew = generate_random_vec(N)
        acceptrate = accept_prob(sigma, sigmanew, gpar)
        testaccept = npr.rand()
        if testaccept < acceptrate:
            sigma = sigmanew
        evalenergy[counter] = localenergy(sigma,gpar,ju)
    meanE = np.mean(evalenergy)/N
    #now evaluate the dispersion of the energies
    stdevbins = np.zeros(10)
    for i in range(10):
        stdevbins[i] = np.mean(evalenergy[sbins * i : sbins * (i+1)]/N)
    stdE = 2 * np.std(stdevbins)/math.sqrt(10)
    return meanE, stdE

def optimumg(ju,N):
    Niter=40000
    if N<10:
        Nitermax=50000
    else:
        Nitermax=160000
    gmin=0.
    gmax=2.
    gmid=(gmin+gmax)/2.
    gacc=0.01
    Egmin, stdgmin = energyg(gmin, ju, N, Niter)
    Egmax, stdgmax = energyg(gmax, ju, N, Niter)
    Egmid, stdgmid = energyg(gmid, ju, N, Niter)
    while (gmax-gmin)>= 2 * gacc:
        while ((stdgmin >= 2* abs(Egmid - Egmin) or stdgmid >= 2* min(abs(Egmid - Egmin), abs(Egmid - Egmax)) or stdgmax >=2* abs(Egmid - Egmax)) and Niter <= Nitermax/2):
            Niter = Niter * 2
            print('Niter = ' + str(Niter))
            Egmin, stdgmin = energyg(gmin, ju, N, Niter)
            Egmax, stdgmax = energyg(gmax, ju, N, Niter)
            Egmid, stdgmid = energyg(gmid, ju, N, Niter)
        g1 = (gmin + gmid)/2
        Eg1,stdg1 = energyg(g1, ju, N, Niter)
        if Eg1 > Egmid:
            gmin = g1
            Egmin=Eg1
        else:
            gmax=gmid
            Egmax=Egmid
            gmid = g1
            Egmid=Eg1
        g2 = (gmid + gmax)/2
        Eg2, stdg2 = energyg(g2, ju, N, Niter)
        if Eg2 > Egmid:
            gmax = g2
            Egmax=Eg2
        else:
            gmin=gmid
            Egmin=Egmid
            gmid = g2
            Egmid=Eg2
        #print('gmin, gmid, gmax = ' + str(gmin) + ', ' + str(gmid) + ', ' + str(gmax))
    return gmid, Egmid/N, Niter


def observables(gopt, ju, N, Niter):
    sigma = generate_random_vec(N)
    n1sq = 0
    rhoij = np.zeros([N,N])
    for counter in range(Niter):
        sigmanew = generate_random_vec(N)
        acceptrate = accept_prob(sigma, sigmanew, gopt)
        testaccept = npr.rand()
        if testaccept < acceptrate:
            sigma = sigmanew
        n1sq+=(float(sigma[0])**2)/Niter
        for i in range(N):
            for j in range(N):
                if i==j:
                    rhoij[i,j] += float(sigma[i])/Niter
                else:
                    rhoij[i,j] += sigma[i] * math.exp(- 2 * gopt * (1 + sigma[j] - sigma[i]))/Niter
    dfluc = n1sq-1
    valdens = np.linalg.eigvals(rhoij)
    fc = np.max(valdens)/N
    return dfluc,fc

def calculate_gopt_and_observables(ju, N):
    gopt, energy, Niterfin = optimumg(ju,N)
    print(gopt)
    dfluc, fc = observables(gopt, ju, N, Niterfin)
    return energy, fc, dfluc

#print(accept_prob(np.array([1,1,1]),np.array([0,0,3]),0.))
#print(localenergy(np.array([3,0,0]),0.1,1.))
#print(energyg(0.1,0.1,3,10000))
#print(optimumg(0.4,3))
#print(calculate_gopt_and_observables(0.4,6))

npoints=9
jutab = np.linspace(0,0.3,npoints,endpoint=True)
energytab = np.zeros(npoints)
fctab = np.zeros(npoints)
dfluctab = np.zeros(npoints)

for ndim in [2,5,6,7,8,9,10]:
    print('N particles = ' + str(ndim))
    for k in range(len(jutab)):
        ju=jutab[k]
        print('J/U = ' + str(ju))
        energytab[k],fctab[k],dfluctab[k] = calculate_gopt_and_observables(ju, ndim)
    print('N = '+ str(ndim)+', J/U = ' + str(jutab)+', E = '+str(energytab)+', fc = '+str(fctab)+', Dn2 = '+str(dfluctab))
    plt.plot(jutab,fctab, label='N = '+str(ndim))

plt.legend(loc='best')
plt.xlabel('J/U', style='italic')
plt.ylabel('condensed fraction')
plt.axis([-0.01,0.31,-0.1,1.1])
plt.savefig('fcvsjuQMC2.pdf')
plt.show()
