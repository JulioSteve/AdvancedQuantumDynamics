# %%
import numpy as np
import scipy as sp
from numpy import linalg as la
import matplotlib.pyplot as plt
from numpy import linalg as LA;
from scipy import linalg as LA2;
from numpy import random as rand
from scipy.sparse import diags
from scipy.sparse import csr_matrix

# %%
def tensorvect(a,b):
    return(np.tensordot(a,b,axes=0).flatten())

def tensorvectop(a,b):
    return np.kron(a,b)

def opchain(a,i,nspin):
    if i==1:
        return np.kron(a,np.identity(2**(nspin-1)))
    else:
        if i==nspin:
            return np.kron(np.identity(2**(nspin-1)),a)
        else:
            return np.kron(np.kron(np.identity(2**(i-1)),a),np.identity(2**(nspin-i)))
        
def opchain2(a,i,b,j,nspin):
    if i==1:
        if j==nspin:
            return np.kron(np.kron(a,np.identity(2**(nspin-2))),b)
        else:
            return np.kron(np.kron(a,np.identity(2**(j-2))),np.kron(b,np.identity(2**(nspin-j))))      
    else:
        if j==nspin:
            return np.kron(np.kron(np.identity(2**(i-1)),a),np.kron(np.identity(2**(nspin-(i+1))),b))
        else:
            return np.kron(np.kron(np.kron(np.identity(2**(i-1)),a),np.kron(np.identity(2**(j-(i+1))),b)),np.identity(2**(nspin-j)))
            
def buildstate(bin):
    v=[0. for i in range(2**len(bin))];
    v[int(bin,2)]=1.
    return np.array(v)


def diracrep(psi,nspin):
    state='';
    for i in range(2**nspin):
        if abs(psi[i])>10**(-6):
            state=state+'+'+str(psi[i])+'|'+format(i,'0'+str(nspin)+'b')+'>'
    return state

def binnum(n):
    l=['0','1'];
    if n==1:
        return l
    else:
        return ['0'+i for i in binnum(n-1)]+['1'+i for i in binnum(n-1)]
    
def densmat(psi,i,nspin):
    if i>1:
        listindex0=binnum(i-1)
        listindex0=[j+'0' for j in listindex0]
    else:
        listindex0=['0']
    if i<nspin:
        listcomp=binnum(nspin-i)
        listindex0=list(np.array([[j+k for k in listcomp] for j in listindex0]).flatten())
    if i>1:
        listindex1=binnum(i-1)
        listindex1=[j+'1' for j in listindex1]
    else:
        listindex1=['1']
    if i<nspin:
        listcomp=binnum(nspin-i)
        listindex1=list(np.array([[j+k for k in listcomp] for j in listindex1]).flatten())
    rho00=sum(psi[int(j,2)]*np.conjugate(psi[int(j,2)]) for j in listindex0)
    rho11=sum(psi[int(j,2)]*np.conjugate(psi[int(j,2)]) for j in listindex1)
    rho01=sum(psi[int(j,2)]*np.conjugate(psi[int(listindex1[listindex0.index(j)],2)]) for j in listindex0)
    return np.array([[rho00,rho01],[np.conjugate(rho01),rho11]])

def avdensmat(psi,nspin):
    rho=densmat(psi,1,nspin);
    if nspin>1:
        for i in range(2,nspin+1):
            rho=rho+densmat(psi,i,nspin)
    rho=rho/nspin;
    return rho

def purity(rho):
    rho2 = np.dot(rho,rho)
    tr = np.trace(rho2)
    return(tr)

def SvN(rho):
    vp=np.real(LA.eigvals(rho));
    S=0.;
    for i in range(len(vp)):
        if vp[i]>0.:
            S=S+vp[i]*np.log(vp[i])
    return -S

def entangl(psi,nspin):
    S=SvN(densmat(psi,1,nspin));
    if nspin>1:
        for i in range(2,nspin+1):
            S=S+SvN(densmat(psi,i,nspin))
    return S/nspin

def Disorder(psi,nspin):
    return SvN(avdensmat(psi,nspin))-entangl(psi,nspin)

# %%
sigX=np.array([[0.,1.],[1.,0.]]);
sigY=np.array([[0.,-1j],[1j,0.]]);
sigZ=np.array([[1.,0.],[0.,-1.]]);
sig1=np.array([[1.,0.],[0.,0.]]);
id2 =np.array([[1.,0.],[0.,1.]]);

NOT=sigX;
HAD1=np.array([[1./np.sqrt(2.),1./np.sqrt(2.)],[1./np.sqrt(2.),-1./np.sqrt(2.)]]);
CNOT=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]]);
HAD2=np.array([[0.5,0.5,0.5,0.5],[0.5,-0.5,0.5,-0.5],[0.5,0.5,-0.5,-0.5],[0.5,-0.5,-0.5,0.5]]);
SWAP=np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]);

# %% [markdown]
# # Student code starts here:
# 
# In this practical work we will focus on quantum systems: lattice spin systems. In the first section we will tackle the description of various Hamiltonian that depend on the desired system we want to simulate (micro configurations of spins). In a second part we will use some Hamiltonians to perform diagonalization algortihm called "power method algorithm" in order to study properties of para-ferro-antiferromagnetic systems. Finally we will simulate the dynamics, using time evolution operator, of some quantum spin chain systems and discuss the impact of some parameters.

# %% [markdown]
# # 2.1 Studied models

# %% [markdown]
# In the following cell we will code functions to build Hamiltonian in the desired shape:

# %%
def QuantumHamiltonian_chain(Jxyz, w, N, flag, force_real):
    h0 = np.array([[-w,0],[0,0]])
    H0 = np.zeros(shape=(2**N,2**N), dtype=np.complex128)
    Hint = np.copy(H0)
    for i in range(1,N+1):
        H0 += opchain(h0,i,N)
        
    for i in range(1,N):
        j = i+1

        Hint -= (Jxyz[0]*opchain2(0.5*sigX,i,0.5*sigX,j,N)+Jxyz[1]*opchain2(0.5*sigY,i,0.5*sigY,j,N)+Jxyz[2]*opchain2(0.5*sigZ,i,0.5*sigZ,j,N))
    if (flag=="close" and N>2):
        Hint -= (Jxyz[0]*opchain2(0.5*sigX,1,0.5*sigX,N,N)+Jxyz[1]*opchain2(0.5*sigY,1,0.5*sigY,N,N)+Jxyz[2]*opchain2(0.5*sigZ,1,0.5*sigZ,N,N))
    H = H0+Hint

    if force_real == True:
        H = H.real

    return H

def QuantumHamiltonian_general(J, w, N, flag, force_real):
    if type(w)==float:
        h0 = np.array([[-w,0],[0,0]])
    H0 = np.zeros(shape=(2**N,2**N), dtype=np.complex128)
    for i in range(1,N+1):
        if type(w)!=float:
            h0 = np.array([[-w[i-1],0],[0,0]])
        H0 += opchain(h0,i,N)
        
    for i in range(1,N):
        for j in range(i+1,N+1):
            H0 -= J[i-1,j-1,0]*opchain2(0.5*sigX,i,0.5*sigX,j,N)
            H0 -= J[i-1,j-1,1]*opchain2(0.5*sigY,i,0.5*sigY,j,N)
            H0 -= J[i-1,j-1,2]*opchain2(0.5*sigZ,i,0.5*sigZ,j,N)
    if (flag=="close" and N>2):
        H0 -= (J[0,N-1,0]*opchain2(0.5*sigX,1,0.5*sigX,N,N)+J[0,N-1,1]*opchain2(0.5*sigY,1,0.5*sigY,N,N)+J[0,N-1,2]*opchain2(0.5*sigZ,1,0.5*sigZ,N,N))

    if force_real == True:
        H0 = H0.real

    return H0

# %% [markdown]
# ### 2.1 Studied models: Hamiltonian tests for different lattices and different sizes (N=2,3,8):
# 
# please read the Python comments, at the top of cells, to see which Hamiltonian is considered in each cell.

# %%
# (1) open Ising-Z spin chain

H_oIZ_2 = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.5, N=2, flag="open", force_real=True)
H_oIZ_3 = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.5, N=3, flag="open", force_real=True)
H_oIZ_8 = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.5, N=8, flag="open", force_real=True)

print(f"N=2 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oIZ_2,3)}\n\n")
print(f"N=3 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oIZ_3,3)}\n\n")
print("N=8:")
print(H_oIZ_8)
print(csr_matrix(H_oIZ_8))

# %%
# (2) closed Ising-Z spin chain

H_cIZ_2 = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.5, N=2, flag="close", force_real=True)
H_cIZ_3 = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.5, N=3, flag="close", force_real=True)
H_cIZ_8 = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.5, N=8, flag="close", force_real=True)

print(f"N=2 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_cIZ_2,3)}\n\n")
print(f"N=3 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_cIZ_3,3)}\n\n")
print("N=8:")
print(H_cIZ_8)
print(csr_matrix(H_cIZ_8))

# %%
# (3) open Ising-X spin chain

H_oIX_2 = QuantumHamiltonian_chain(Jxyz=[1,0,0], w=0.5, N=2, flag="open", force_real=True)
H_oIX_3 = QuantumHamiltonian_chain(Jxyz=[1,0,0], w=0.5, N=3, flag="open", force_real=True)
H_oIX_8 = QuantumHamiltonian_chain(Jxyz=[1,0,0], w=0.5, N=8, flag="open", force_real=True)

print(f"N=2 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oIX_2,3)}\n\n")
print(f"N=3 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oIX_3,3)}\n\n")
print("N=8:")
print(H_oIX_8)
print(csr_matrix(H_oIX_8))

# %%
# (4) open Heisenberg-XXX spin chain

H_oHXXX_2 = QuantumHamiltonian_chain(Jxyz=[1,1,1], w=0.5, N=2, flag="open", force_real=True)
H_oHXXX_3 = QuantumHamiltonian_chain(Jxyz=[1,1,1], w=0.5, N=3, flag="open", force_real=True)
H_oHXXX_8 = QuantumHamiltonian_chain(Jxyz=[1,1,1], w=0.5, N=8, flag="open", force_real=True)

print(f"N=2 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oHXXX_2,3)}\n\n")
print(f"N=3 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oHXXX_3,3)}\n\n")
print("N=8:")
print(H_oHXXX_8)
print(csr_matrix(H_oHXXX_8))

# %%
# (5) open Heisenberg-XYZ spin chain

H_oHXYZ_2 = QuantumHamiltonian_chain(Jxyz=[0.5,1,1.5], w=0.5, N=2, flag="open", force_real=True)
H_oHXYZ_3 = QuantumHamiltonian_chain(Jxyz=[0.5,1,1.5], w=0.5, N=3, flag="open", force_real=True)
H_oHXYZ_8 = QuantumHamiltonian_chain(Jxyz=[0.5,1,1.5], w=0.5, N=8, flag="open", force_real=True)

print(f"N=2 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oHXYZ_2,3)}\n\n")
print(f"N=3 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_oHXYZ_3,3)}\n\n")
print("N=8:")
print(H_oHXYZ_8)
print(csr_matrix(H_oHXYZ_8))

# %%
# (6) open random Ising-X spin chain with random Larmor frequencies

w_rand = lambda N: np.random.uniform(0,0.5,N)
def build_J_X(N):
    J = np.zeros(shape=(N,N,3))
    for i in range(N-1):
            J[i,i+1,0] = np.random.uniform(-1,1)
    return J

H_orIX_2 = QuantumHamiltonian_general(build_J_X(2), w_rand(2), N=2, flag="open", force_real=True)
H_orIX_3 = QuantumHamiltonian_general(build_J_X(3), w_rand(3), N=3, flag="open", force_real=True)
H_orIX_8 = QuantumHamiltonian_general(build_J_X(8), w_rand(8), N=8, flag="open", force_real=True)

print(f"N=2 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_orIX_2,3)}\n\n")
print(f"N=3 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_orIX_3,3)}\n\n")
print("N=8:")
print(H_orIX_8)
print(csr_matrix(H_orIX_8))

# %%
# (7) open random Spin Glass with random Larmor frequencies

w_rand = lambda N: np.random.uniform(0,0.5,N)
def build_JXYZ(N):
    J = np.zeros(shape=(N,N,3))
    for i in range(N):
        for j in range(N):
            if j>i:
                J[i,j] = np.random.uniform(-1,1,3)
    return J

H_SP_2 = QuantumHamiltonian_general(build_JXYZ(2), w_rand(2), N=2, flag="close", force_real=True)
H_SP_3 = QuantumHamiltonian_general(build_JXYZ(3), w_rand(3), N=3, flag="close", force_real=True)
H_SP_8 = QuantumHamiltonian_general(build_JXYZ(8), w_rand(8), N=8, flag="close", force_real=True)

print(f"N=2 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_SP_2,3)}\n\n")
print(f"N=3 (we rounded random outputs to see a proper matrix on screen):\n{np.around(H_SP_3,3)}\n\n")
print("N=8:")
print(H_SP_8)
print(csr_matrix(H_SP_8))

# %% [markdown]
# # 2.2 Diagonalization algorithms - Power Method algorithm 
# 
# We called "pm_0" the ground-state power-method algorithm and "pm_1" its 1st excited equivalent.

# %%
def pm_0(Hinput, mu, eps, kmax):
    N = len(Hinput)
    ket0 = np.random.rand(N)+1j*np.random.rand(N)
    ket0 = ket0/np.linalg.norm(ket0)

    # print(np.conjugate(ket0)@ket0) #We normalized as expected

    H = np.copy(Hinput)
    H = H-mu*np.identity(N)
    k = 0
    w = H@ket0
    while np.linalg.norm(w-(np.conjugate(ket0)@w)*ket0)>eps and k<=kmax:
        ket0 = H@ket0
        ket0 = ket0/np.linalg.norm(ket0)
        k = k+1
        w = H@ket0
        if k==kmax:
            print("pm0 did not converge!")
    
    H = H+mu*np.identity(N)
    eigval = np.conjugate(ket0)@w

    return (eigval, ket0)

def pm_1(H, mu=100, eps=1e-10, kmax=int(1e3)):
    (eigval0, ket0)= pm_0(H, mu, eps, kmax)
    N = len(H)
    ket1 = np.random.rand(N)+1j*np.random.rand(N)
    ket1 = ket1 - (np.conj(ket0)@ket1)*ket0
    ket1 = ket1/np.linalg.norm(ket1)

    H = np.copy(H)
    H = H-mu*np.identity(N)
    k = 0
    while (np.linalg.norm(H@ket1-(np.conjugate(ket1)@H@ket1)*ket1)>eps and k<=kmax):
        ket1 = H@ket1
        ket1 = ket1 - (np.conj(ket0)@ket1)*ket0
        ket1 = ket1/np.linalg.norm(ket1)
        k += 1
        if k==kmax:
            print("pm1 did not converge!")
    
    H = H+mu*np.identity(N)
    eigval1 = np.conjugate(ket1)@H@ket1

    return (eigval0,eigval1, ket0, ket1)

# %% [markdown]
# This function is used to write kets in a friendly way:

# %%
def ket_to_string(ket):
    n = len(ket)
    N = int(np.log(n)/np.log(2))
    coef = []
    arg = []
    ket_string = []
    coef_final = []
    arg_final = []
    final_print = []
    for i,c in enumerate(ket):
        coef.append(np.abs(c))
        if coef[i]!=0:
            arg.append(np.angle(c/abs(c)))
        else:
            arg.append(0)

        ket_string.append(format(i, f"0{N}b"))

        if np.around(coef[i],3)!=0:
            final_print.append(ket_string[i])
            coef_final.append(coef[i])
            arg_final.append(arg[i])
    
    for i in range(len(final_print)):
        final_print[i] = f"{coef_final[i]:.3f} exp({arg_final[i]:.3f}) |"+final_print[i]+">"
        if i==0:
            str = final_print[i]
        if i != 0:
            str += " + "+final_print[i]

    return str

# %% [markdown]
# Now we can test it on some Hamiltonian systems and see if the Python-Scipy method is according to our results.

# %%
l0,l1,v0,v1 = pm_1(H_oIX_3, mu=100, eps=1e-10, kmax=1e5)
(eigvals, eigvecs) = LA.eigh(H_oIX_3)

print("Here N=3.")
print("Recalling the considered Hamiltonian for X-Ising chain of spin:")
print(H_oIX_3)

print(f"\nground eigenvalue: {np.around(l0,3)}")
print(f"excited eigenvalue: {np.around(l1,3)}\n")
print("raw eigenvectors (0,1):")
print(np.around(v0,3))
print(np.around(v1,3))
print(f"\nA better visualization is to print in the form |c|*exp(i*phi)|ket>:")
print(f"v0: {ket_to_string(v0)}")
print(f"v1: {ket_to_string(v1)}")

print(f"\nNumpy linalg.eigh method comparison:")
print(f"eigval0_numpy = {eigvals[0]:.3f}")
print(f"eigval1_numpy = {eigvals[1]:.3f}")
print(f"eigvec0_numpy: {np.around(eigvecs[:,0],3)}")
print(f"eigvec1_numpy: {np.around(eigvecs[:,1],3)}")



# %% [markdown]
# We see that 

# %%
l0,l1,v0,v1 = pm_1(H_oIX_8, mu=500, eps=1e-8, kmax=1e6)
(eigvals, eigvecs) = LA.eigh(H_oIX_8)

print("Here N=8.")
print("Recalling the considered Hamiltonian for X-Ising chain of spin:")
print(H_oIX_8)

print(f"\nground eigenvalue: {np.around(l0,3)}")
print(f"excited eigenvalue: {np.around(l1,3)}\n")
# print("raw eigenvectors (0,1):")
# print(np.around(v0,3))
# print(np.around(v1,3))
print(f"\nA better visualization is to print in the form |c|*exp(i*phi)|ket>:")
print(f"v0: {ket_to_string(v0)}")
print(f"v1: {ket_to_string(v1)}")

print(f"\nNumpy linalg.eigh method comparison:")
print(f"eigval0_numpy = {eigvals[0]:.3f}")
print(f"eigval1_numpy = {eigvals[1]:.3f}")
print(f"eigvec0_numpy: {np.around(eigvecs[:,0],3)}")
print(f"eigvec1_numpy: {np.around(eigvecs[:,1],3)}")



# %% [markdown]
# # 2.3 Properties of the ground state
# ## 2.3.1 Para- and ferromagnetic systems

# %%
def plot_231(H, N,title, mu, eps, kmax):
    plt.figure()
    v0 = pm_0(H, mu, eps, kmax)[1]
    n_up = np.zeros(N, dtype=np.complex64)
    n_down = np.zeros(N, dtype=np.complex64)
    S = np.zeros(N, dtype=np.complex64)
    coherence = np.zeros(N)
    E = entangl(v0,N)
    D = Disorder(v0, N)

    for i in range(1,N+1):
        j=i-1
        rho = densmat(v0,i,N)
        coherence[j] = np.abs(rho[0,1]) #the matrix is hermitian and we have |rho_01|=|rho_10|
        n_up[j] = rho[0,0]
        n_down[j] = rho[1,1]
        S[j] = SvN(rho)

    X = np.arange(1,N+1)
    
    markers = ["o","^","*","d"]
    colors = ["red", "blue", "green", "purple"]
    labels = [r"$\rho_{00}:\uparrow$ population",r"$\rho_{11}:\downarrow$ population",r"$S_{vN}(\langle\rho\rangle)$:von Neumann Entropy",r"$|\rho_{01}|=|\rho_{10}|$:Coherence (modulus)"]
    for i,L in enumerate([n_up,n_down,S,coherence]):
        L = L.real
        plt.plot(X,L, marker=markers[i], color=colors[i], ls="solid", lw=1, ms=10, label=labels[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    title = title + f"\nE={E:.3f}, D={D:.3f}"
    plt.title(title, fontsize=15)
    plt.xlabel("Spins")

# %%
# EXPECT AT LEAST 30 SECONDS TO RUN THIS CELL! 
# In the case it does not converge, relaunch this cell, the random vector might be wrong.

H1 = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.5, N=8, flag="close", force_real=False)
H2 = QuantumHamiltonian_chain(Jxyz=[1,0,0], w=0.5, N=8, flag="open", force_real=False)
H3 = QuantumHamiltonian_general(build_J_X(8), w_rand(8), N=8, flag="open", force_real=False)

plot_231(H1, 8,'case (2): closed Ising-Z chain - N=8', mu=100, eps=1e-8, kmax=1e6)
# print("(2) done")
plot_231(H2, 8,'case (3): open Ising-X chain - N=8', mu=100, eps=1e-8, kmax=1e6)
# print("(3) done")
plot_231(H3, 8,'case (6): open random Ising-X chain - N=8', mu=100, eps=1e-8, kmax=1e6)
# print("(6) done")

# %% [markdown]
# ### In the following we speak about the fundamental level (ground state) of every considered system:
# 
# We can see in the first case (3) the closed Ising-Z chain that the population are always in the same state for each spin, they are all in the up-state. This means that the state can be seen as a product state of only up-spins. This is thus expected that the entropy is null because we fell in a classical ordered micro/macro-system. This is confirmed by an Entanglement measurement (E) that is null as the disorder.
# 
# For the model (3) the open Ising-X chain, here the spins are not forced to be up, there is a non-zero probability for them to be down (the ones on the border are less probable to be down than up, this is attended because they have no interaction on one side). The entanglement is possible as the value of E indicates and macroscopically the system cannot be ordered a priori as the disorder (D) is non-zero. We now see a curve for the _von Neumann_ entropy $S_{vN}$ that seems to be roughly proportionnal to the down population probability (with respect to each spin).
# 
# Finally, for the most random model (6) which is an open Ising-X chain (using random interactions and Larmor frequencies) the population are randomly more likely to be up/down. The observation of the last case remains correct here, the _von Neumann_ entropy $S_{vN}$ seems to be roughly proportionnal to the down population. In this perticular case, the entanglement and disorder are likely to be higher because of the randomness of everything in the system. 
# 
# Interestingly, it seems that in all cases the systems have no coherences (modulus are close to 0). Or they might just be extremely small and negligible compared to other values. Physically, we could consider a complete decoherence. 
# 

# %% [markdown]
# ## 2.3.2 Antiferromagnetic systems

# %%
H_ferro_open = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.0, N=8, flag="open", force_real=False)
H_anti_open = QuantumHamiltonian_chain(Jxyz=[0,0,-1], w=0.0, N=8, flag="open", force_real=False)

plot_231(H_ferro_open, 8,'FERROMAGNETIC open Ising-Z chain - N=8', mu=100, eps=1e-8, kmax=1e6)
plot_231(H_anti_open, 8,'ANTI-FERROMAGNETIC open Ising-Z chain - N=8', mu=100, eps=1e-8, kmax=1e6)

# %% [markdown]
# If we compare the ground-states of ferromagnetic/anti-ferromagnetic systems, we see that the behavior of spins is different. In both cases, the entanglement is non-zero and higher than the previous models observed in 2.3.1. A first distinction is the disorder, as we can see for anti-ferromagnetic the disorder is non-zero while for the ferromagnetic the system seems microscopically ordered. 
# 
# For the populations, the ferromagnetic has equiprobability between each spins to be down/up but it is still stochastic (probability up ~70% while down ~30%) besides that, the phenomenon for anti-ferromagnetic is an alternation of high/low probability for each spin in the chain. If your neighbor is up, you are likely to be down etc. which is expected for anti-ferromagnetic because each spin up should be compensed by a spin down. The average probability per spin is 50% up/down. 
# 
# In both cases we have a high _von Neumann_ entropy that is constant with respect to spin. We can this way simulate the computation of a ferro/antiferro system and compare the entropy with reality.

# %%
H_ferro_close = QuantumHamiltonian_chain(Jxyz=[0,0,1], w=0.0, N=8, flag="close", force_real=False)
H_anti_close = QuantumHamiltonian_chain(Jxyz=[0,0,-1], w=0.0, N=8, flag="close", force_real=False)

plot_231(H_ferro_close, 8,'FERROMAGNETIC open Ising-Z chain - N=8', mu=100, eps=1e-8, kmax=1e6)
plot_231(H_anti_close, 8,'ANTI-FERROMAGNETIC open Ising-Z chain - N=8', mu=100, eps=1e-8, kmax=1e6)

# %% [markdown]
# For the same systems but closed cases, we see that the fundamental repartition of spins does not change, the only thing changed is that for the ferromagnetic case, the probability to be up/down is closer to 50% each and it lowered a bit the entropy of the system. This is logical, by interacting completely a physical stability is more likely to be reached.
# 
# The entanglement measurement is around the same values with a small increasing in ferromagnetic case and decreasing in anti-ferromagnetic case.

# %% [markdown]
# ### Now let's see the ground state expression directly:

# %%
v0_fo = pm_0(H_ferro_open, mu=100, eps=1e-8, kmax=1e6)[1]
v0_fc = pm_0(H_ferro_close, mu=100, eps=1e-8, kmax=1e6)[1]
v0_ao = pm_0(H_anti_open, mu=100, eps=1e-8, kmax=1e6)[1]
v0_ac = pm_0(H_anti_close, mu=100, eps=1e-8, kmax=1e6)[1]

print(f"Open ground-state ferromagnetic      : {ket_to_string(v0_fo)}")
print(f"Close ground-state ferromagnetic     : {ket_to_string(v0_fc)}")
print(f"Open ground-state anti-ferromagnetic : {ket_to_string(v0_ao)}")
print(f"Close ground-state anti-ferromagnetic: {ket_to_string(v0_ac)}")

# %% [markdown]
# As expected by looking at the population but in a clearer way, ferromagnetic implies a degeneracy for states all-down/all-up while the anti-ferromagnetic states are also a degeneracy but for states alternating up/down.
# 
# 
# Now to do the triangle spin-system we can chose the simplest one being a 3 spins triangle in a closed-chain:

# %%
H_triangle = QuantumHamiltonian_chain(Jxyz=[0,0,-1], w=0.0, N=3, flag="close", force_real=False)

plot_231(H_triangle, 3,'ANTI-FERROMAGNETIC closed Ising-Z chain - N=3', mu=100, eps=1e-8, kmax=1e6)

# %%
v0_triangle = pm_0(H_triangle, mu=100, eps=1e-8, kmax=1e6)[1]
print(f"Ground-state of the simplest triangle spin-system:\n{ket_to_string(v0_triangle)}")

# %% [markdown]
# In the case of an anti-ferromagnetic triangle it seems that the coherences are non-zero and we have a pretty strong entanglement measurement. In the case of a triangle it is obvious that obtaining a real perfect anti-ferromagnetic is not possible, either two spins down, either two up so we don't have a good balance. Thus, one of the spin is correlated with the two others. The "frustration" can be seen as a high entropy value for the system. We can notice that the highest entropy per spin is the one corresponding to the different spin (if two are up then the down one has the highest entropy). The frustration can be seen as necessarily putting the first spin up, the second spin down (to minimize energy of the system) and the third can be either up or down BUT this last spin will increase the energy. 
# 
# For the ground-state it is obvious as stated before that the kets |000> and |111> cannot describe this physical system (they would represent a ferromagnetic system) and so we see this alternation of up/down.

# %% [markdown]
# If we do the same kind of remarks for an even number of spins we understand that the degeneracy is way lower than with odd number of spins. On an even "triangular" spin system we have only a degeneracy that depends on the first spin orientation choice. For an odd system, each choice of a first spin orientation induces more freedom on the last one, so more total degeneracy. 

# %% [markdown]
# # 3. Dynamics of lattice spin systems
# 
# ## 3.1 Studied models
# 
# We consider:
# - (1): Heisenberg-XXX open chain - N=7, w=1, J=0.1
# - (2): Heisenberg-XXX closed chain - N=7, w=1, J=0.1
# - (3): inhomogeneous Heisenberg-XXX open chain - N=7, w=1, $J_{i,i+1}=\frac{0.5}{i}$

# %%
print("case (1)\n")
H_open = QuantumHamiltonian_chain(Jxyz=[0.1,0.1,0.1], w=1, N=7, flag="open", force_real=True)
print(H_open)
print(csr_matrix(H_open)) #I checked all values are strictly real
H_open = QuantumHamiltonian_chain(Jxyz=[0.1,0.1,0.1], w=1, N=7, flag="open", force_real=False)

# %%
print("case (2)\n")
H_closed2 = QuantumHamiltonian_chain(Jxyz=[0.1,0.1,0.1], w=1, N=7, flag="close", force_real=True)
print(H_closed2)
print(csr_matrix(H_closed2)) #I checked all values are strictly real
H_closed2 = QuantumHamiltonian_chain(Jxyz=[0.1,0.1,0.1], w=1, N=7, flag="close", force_real=False)

# %%
print("case (3)\n")

def build_J_XXX_inho(N,val):
    J = np.zeros(shape=(N,N,3))
    for i in range(N-1):
        for k in range(3):
            J[i,i+1,k] = val/(i+1)
    return J

H_closed3 = QuantumHamiltonian_general(build_J_XXX_inho(7,0.5), w=1.0, N=7, flag="close", force_real=True)
print(H_closed3)
print(csr_matrix(H_closed3))
H_closed3 = QuantumHamiltonian_general(build_J_XXX_inho(7,0.5), w=1.0, N=7, flag="close", force_real=False)

# %% [markdown]
# ## 3.2 Spectral integrator

# %% [markdown]
# We use scipy.linalg.expm that does an exponential of matrix in a smart, direct and optimized way:

# %%
def Dyn(H, t, psi0):
    U = LA2.expm(-1j*H*t)
    return U@psi0

# %% [markdown]
# ## 3.3 Dynamics

# %%
Ntime = 200
Nspins = 7
t = np.linspace(0,500,Ntime)
dt = t[1]-t[0]

psi0_mid = buildstate('0001000')
psi0_edge = buildstate('1000000')

def createpop(psi0, H):
    pop = np.zeros(shape=(Ntime, Nspins))
    for i in range(Ntime):
        if i==0:
            psi = psi0
        else:
            psi = Dyn(H, i*dt, psi0)

        for k in range(1, Nspins+1):
            rho = densmat(psi, k, Nspins)
            pop[i, k-1] = rho[1,1].real

    return pop

def graph_model(psi, title):
    fig, ax = plt.subplots(1,3, layout="constrained", sharex=True, sharey=True)
    HList=[H_open, H_closed2, H_closed3]
    Htitle = ["(1)\nH-XXX OPEN", "(2)\nH-XXX CLOSED", "(3) inhomogeneous\nH-XXX OPEN"]
    for k in range(3):
        ax[k].set_title(Htitle[k])
        pop = createpop(psi, HList[k])
        gp=ax[k].contourf(range(1,Nspins+1),[i*dt for i in range(Ntime)],pop,[i*0.02 for i in range(51)],cmap='hot',antialiased=False)
    fig.supxlabel('Spin n°')
    fig.supylabel('Time (s)')
    cbar = fig.colorbar(gp)
    cbar.set_label('Population density', rotation=270, labelpad=15)
    fig.suptitle(title)

# %%
graph_model(psi0_mid, r"$|\psi_0\rangle=|0001000\rangle$")

# %%
graph_model(psi0_edge, r"$|\psi_0\rangle=|1000000\rangle$")

# %% [markdown]
# ## Intepretation of the graphs depending on models and initial state:
# 
# We understand by construction of the initial states that we face two cases where only one spin is up on a chain ends and for the second case, the up one is right in the middle of the spin-chain. 
# 
# We can already expect a different behavior for the closed-open chain considering the initial state with up spin on the edge (second graph). Indeed, if the chain is closed it should roughly look like the middle-up-case by cyclicity. This is what we see on graph 1.1/1.2 and 2.2 with just an excited population on a different spin but the "shape" of what we observe is the same. This explains why 1.1 and 1.2 are so similar: structurally they are the same (in the closed/open case the extreme spins have a 0 interaction result because both down and/or chain is opened).
# 
# In the edge-case with opened chain, we see a swap of excitation from initially being on the 1st spin and swapping around 380s to the 8th spin (seen with yellowish-white color "dots").
# 
# In the closed cases, we have an oscillation between up-spin on initially excited spin and random reparition forth and back. Occasionally, the initial state is recovered when time increases but it is hard to estimate if it is periodic or chaotic with so few oscillations. 
# 
# 
# Now, about the inhomogeneous cases we observe faster oscillation with time of the excitation going from spin to spins and it seems that the maxima of excited population density stays around the initially excited spin. The farthest spins from the initially excited one, are less likely to be excited even when time increases. Farthest can be understand easily because we work on an oppened chain, this behavior is thus maximally visible on graph 2.3 where the 8th spin is almost always completely black.

# %%
def last_graph(psi, title):
    fig, ax = plt.subplots(3,3,layout="constrained", sharex=True, sharey=True)
    fig.suptitle(title)

    wL = [0.0,1.0,10.0]
    jL = [0.01, 0.1, 1.0]

    for i,ws in enumerate(wL):
        for j,js in enumerate(jL):
            ax[i,j].set_title(fr"$w={ws}$ $j={js}$")
            H = QuantumHamiltonian_chain(Jxyz=[js,js,js], w=ws, N=7, flag="open", force_real=True)
            pop = createpop(psi, H)
            gp=ax[i,j].contourf(range(1,Nspins+1),[i*dt for i in range(Ntime)],pop,[i*0.02 for i in range(51)],cmap='hot',antialiased=False)
    fig.supxlabel('Spin n°')
    fig.supylabel('Time (s)')
    cbar = fig.colorbar(gp, ax=ax.ravel().tolist(), shrink=0.8, aspect=30)
    cbar.set_label('Population density', rotation=270, labelpad=20)

# %%
last_graph(psi0_mid, r"$|\psi_0\rangle=|0001000\rangle$")

# %%
last_graph(psi0_edge, r"$|\psi_0\rangle=|1000000\rangle$")

# %% [markdown]
# ## Intepretation of the graphs where we let vary the Larmor frequency and the exchange integral
# 
# The most evident factor in both simulation shows that the dynamics are not depending on the Larmor frequency and only $j$ value changes the way systems react.
# 
# There is not much to say considering the behavior of systems between each other, we use an opened chain and the behavior is the same as seen in the last paragraph. 
# 
# There is still an important characteristic we can deduce from these simulations. The role of the exchange integral value seems to make faster oscillations in dynamics, the excitation seems to travel among the possibilities quicker for a set duration, but the maxima of excited population density is still situated on the same spots described in the last paragraph (initial value or other edge).


