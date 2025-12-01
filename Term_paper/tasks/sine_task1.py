# ============================
# Task 1 — Clean ED Convergence Plots
# 3 Plots, 12–16 data points each
# ============================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize':(6,4),'font.size':12})

# -----------------------------
# HO basis local operators
# -----------------------------
def local_ho_operators(n_max, omega=1.0):
    n = n_max
    rows, cols, data = [], [], []
    for i in range(n-1):
        rows.append(i); cols.append(i+1); data.append(np.sqrt(i+1))
    a = sp.csr_matrix((data,(rows,cols)),shape=(n,n),dtype=np.complex128)
    adag = a.getH()
    phi = (a+adag)/np.sqrt(2*omega)
    pi  = -1j*np.sqrt(omega/2)*(a-adag)
    phi = 0.5*(phi+phi.getH())
    pi  = 0.5*(pi+pi.getH())
    I   = sp.eye(n,format='csr')
    return a,adag,phi,pi,I

def kron_n(ops):
    out = sp.csr_matrix(1.0)
    for A in ops:
        out = sp.kron(out, A, format='csr')
    return out

# -----------------------------
# Hamiltonian builder SG
# -----------------------------
def build_H(N, n_max, alpha, beta, omega=1.0):
    a,adag,phi_local,pi_local,I_local = local_ho_operators(n_max,omega)
    Kin = 0.5*(pi_local @ pi_local)
    phi_dense = phi_local.toarray()
    Cos_dense = 0.5*(expm(1j*beta*phi_dense)+expm(-1j*beta*phi_dense)).real
    Cos_local = sp.csr_matrix(Cos_dense)
    Ids=[sp.eye(n_max,format='csr') for _ in range(N)]
    dim=n_max**N
    H=sp.csr_matrix((dim,dim),dtype=np.complex128)

    # onsite terms
    for j in range(N):
        ops=Ids.copy(); ops[j]=Kin;         H+=kron_n(ops)
        ops=Ids.copy(); ops[j]=alpha*(I_local - Cos_local); H+=kron_n(ops)

    # gradient part
    for j in range(N):
        jp=(j+1)%N
        ops=Ids.copy(); ops[j]=0.5*(phi_local@phi_local); H+=kron_n(ops)
        ops=Ids.copy(); ops[jp]=0.5*(phi_local@phi_local); H+=kron_n(ops)
        ops=Ids.copy(); ops[j]=phi_local; ops[jp]=phi_local; H+=-1*kron_n(ops)

    H=0.5*(H+H.getH())
    return H

# -----------------------------
# Diagonalization
# -----------------------------
def lowest_two(H):
    if H.shape[0] <= 2000:
        evals,vecs=np.linalg.eigh(H.toarray())
        idx=np.argsort(evals)
        return evals[idx][:2]
    vals,_=spla.eigsh(H,k=2,which='SA')
    vals=np.sort(vals.real)
    return vals[0],vals[1]

# ========================================================
# Generate 12–16 data points for three convergence plots
# ========================================================

alpha=0.05
beta=1.0
omega=1.0

# ---- Plot 1: gap vs n_max  (fix N=3)
N=3
n_list = list(range(4,20))     # 4..19 → 16 data points
gaps=[]
for n_max in n_list:
    H=build_H(N,n_max,alpha,beta,omega)
    E0,E1 = lowest_two(H)
    gaps.append(E1-E0)

plt.plot(n_list,gaps,'o-')
plt.xlabel("n_max")
plt.ylabel("gap = E1 - E0")
plt.title("Convergence: gap vs n_max (N=3)")
plt.grid(True)
plt.show()

# ---- Plot 2: gap vs 1/N  (fix n_max=6)
n_max=6
N_list = list(range(2,18))     # N = 2..17 → 16 points
gaps2=[]; invN=[]
for N in N_list:
    H=build_H(N,n_max,alpha,beta,omega)
    E0,E1 = lowest_two(H)
    gaps2.append(E1-E0)
    invN.append(1.0/N)

plt.plot(invN,gaps2,'o-')
plt.xlabel("1/N")
plt.ylabel("gap")
plt.title("Finite-size scaling: gap vs 1/N (n_max=6)")
plt.grid(True)
plt.show()

# ---- Plot 3: ground-state energy E0 vs n_max (N=3)
N=3
n_list2 = list(range(4,20))    # 16 values
E0_list=[]
for n_max in n_list2:
    H=build_H(N,n_max,alpha,beta,omega)
    E0,_ = lowest_two(H)
    E0_list.append(E0)

plt.plot(n_list2,E0_list,'o-')
plt.xlabel("n_max")
plt.ylabel("E0")
plt.title("Ground-state energy vs n_max (N=3)")
plt.grid(True)
plt.show()
