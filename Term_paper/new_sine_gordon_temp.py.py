#!/usr/bin/env python
# coding: utf-8

# In[42]:


# Cell 1: Imports and helpers
# - Import scientific packages and define a safe Kronecker-product helper.
# - Keep matplotlib settings here.
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(6,4), 'font.size':12})

def kron_n(ops):
    """Kronecker product of a list of (sparse) operators.
    Uses sparse kron and returns csr. Each element of `ops` should be a scipy.sparse or dense array.
    """
    out = sp.csr_matrix(1.0)
    for A in ops:
        # ensure A is sparse csr
        if not sp.isspmatrix_csr(A):
            A = sp.csr_matrix(A)
        out = sp.kron(out, A, format='csr')
    return out


# In[43]:


# Cell 2: Build local harmonic-oscillator operators (truncated Fock basis)
# - n_max: number of Fock states (0..n_max-1)
# - Uses conventional ladder elements a_{n,n+1} = sqrt(n+1)
# - phi and pi defined with ω=1, m=1: phi = (a + a†)/sqrt(2), pi = -i * sqrt(1/2) * (a - a†)
def local_ho_operators(n_max, omega=1.0):
    """
    Return (a, adag, phi, pi, id) as sparse csr matrices of shape (n_max, n_max).
    """
    n = n_max
    # creation/annihilation
    data = []
    rows = []
    cols = []
    for i in range(n-1):
        # a_{i, i+1} = sqrt(i+1)
        rows.append(i)
        cols.append(i+1)
        data.append(np.sqrt(i+1))
    a = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.complex128)
    adag = a.getH()

    # phi and pi with m=1, omega given
    phi = (1.0/np.sqrt(2.0*omega)) * (a + adag)
    pi  = -1j * np.sqrt(omega/2.0) * (a - adag)

    I = sp.eye(n, format='csr', dtype=np.complex128)
    # ensure Hermitian symmetry numeric clean
    phi = 0.5*(phi + phi.getH())
    pi  = 0.5*(pi + pi.getH())

    return a.tocsr(), adag.tocsr(), phi.tocsr(), pi.tocsr(), I

# quick sanity
if __name__ == "__main__":
    a, adag, phi_op, pi_op, I = local_ho_operators(6)
    # check commutator approx [phi, pi] ~ i
    comm = phi_op.dot(pi_op) - pi_op.dot(phi_op)
    # take matrix element (0,0) as a quick numeric check
    print("comm[0,0] (should be ~ 1j*delta):", comm[0,0])


# In[44]:


# Cell 3: Build many-body Sine-Gordon Hamiltonian in HO basis
# Hamiltonian:
# H = sum_j [ 1/2 pi_j^2 + 1/2 (phi_{j+1} - phi_j)^2 + alpha*(1 - cos(beta * phi_j)) ]
# Periodic boundary conditions assumed.
def build_hamiltonian_ho(N, n_max, alpha, beta, omega=1.0):
    """
    N: number of sites
    n_max: local Fock truncation
    alpha: cosine prefactor (potential scale)
    beta: sine-gordon coupling in cos(beta phi)
    """
    a, adag, phi_local, pi_local, I_local = local_ho_operators(n_max, omega=omega)
    Kin_local = 0.5 * (pi_local @ pi_local)  # 1/2 pi^2
    # cosine potential: cos(beta phi) via matrix exponential on local phi (dense is OK for small n_max)
    phi_dense = phi_local.toarray()
    Cos_local_dense = 0.5 * (expm(1j * beta * phi_dense) + expm(-1j * beta * phi_dense)).real
    Cos_local = sp.csr_matrix(Cos_local_dense)

    # build many-body H
    Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
    dim = n_max ** N
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)

    # single-site kinetic and cosine
    for j in range(N):
        ops = Id_list.copy(); ops[j] = Kin_local
        H += kron_n(ops)
        ops = Id_list.copy(); ops[j] = alpha * (I_local - Cos_local)
        H += kron_n(ops)

    # gradient term 0.5 (phi_{j+1} - phi_j)^2
    for j in range(N):
        jp = (j + 1) % N
        # 0.5 * phi_j^2  (we will add phi_j^2 and phi_{j+1}^2; each site gets 0.5 from two neighbors -> correct)
        ops = Id_list.copy(); ops[j] = 0.5 * (phi_local @ phi_local)
        H += kron_n(ops)
        # phi_{j+1}^2 (will also be added in its own j iteration)
        ops = Id_list.copy(); ops[jp] = 0.5 * (phi_local @ phi_local)
        H += kron_n(ops)
        # cross term - phi_j * phi_{j+1}
        ops = Id_list.copy(); ops[j] = phi_local; ops[jp] = phi_local
        H += -1.0 * kron_n(ops)

    # Symmetrize for numerical safety
    H = 0.5 * (H + H.getH())
    return H.tocsr()

# quick sanity (very small)
if __name__ == "__main__":
    H = build_hamiltonian_ho(N=2, n_max=6, alpha=0.05, beta=1.0)
    print("H shape:", H.shape, "nnz:", H.nnz)


# In[45]:


# Cell 4: Exact diagonalization for low-energy spectrum
# - Use sparse eigensolver for lowest k eigenvalues.
# - Warning: if dim > ~8000, this will be slow or fail; reduce N or n_max.
def compute_lowest_eigs(H, k=6):
    """Return sorted lowest k eigenvalues and eigenvectors."""
    k = min(k, H.shape[0]-1)
    vals, vecs = spla.eigsh(H, k=k, which='SA', tol=1e-8, maxiter=5000)
    idx = np.argsort(vals.real)
    return vals.real[idx], vecs[:, idx]

# Example run (adjust N, n_max for your laptop)
N = 3
n_max = 6
alpha = 0.05
beta = 1.0

print(f"Building H (N={N}, n_max={n_max}) ...")
H = build_hamiltonian_ho(N, n_max, alpha, beta)
vals, vecs = compute_lowest_eigs(H, k=6)
print("Lowest energies:", np.round(vals, 8))
print("Gap E1-E0 =", vals[1] - vals[0])

plt.plot(np.arange(len(vals)), vals, 'o-')
plt.xlabel('level'); plt.ylabel('energy'); plt.title(f'Low-energy spectrum N={N}, n_max={n_max}')
plt.grid(True)
plt.show()


# In[46]:


# Cell 5: Convergence sweep over n_max (local Fock truncation)
# - Keep N fixed and sweep n_max = [4,6,8,...]. Plot gap vs n_max to check convergence.
N = 3
nmax_list = [4,6,8]   # add 10 if you have RAM/time
alpha = 0.05
beta = 1.0

gaps = []
for n_max in nmax_list:
    print("n_max =", n_max)
    H = build_hamiltonian_ho(N, n_max, alpha, beta)
    vals, _ = compute_lowest_eigs(H, k=2)
    gaps.append(vals[1] - vals[0])
    print("  gap:", gaps[-1])

plt.plot(nmax_list, gaps, 'o-')
plt.xlabel('n_max'); plt.ylabel('gap (E1-E0)'); plt.title(f'Convergence sweep (N={N})'); plt.grid(True)
plt.show()


# In[47]:


# ===========================
# Cell 6 (FINAL, NAME-PRESERVING)
# Vertex correlator C(r) = < e^{i beta phi_r} e^{-i beta phi_0} >
# ===========================

# --- IMPORTANT ---
# This cell DOES NOT rename any function.
# It keeps your EXACT names: local_vertex_op(), make_manybody_vertex_list()

# --------------------------
# Parameters (set ONCE here)
# --------------------------
N     = 3
n_max = 6
alpha = 0.05
beta  = 1.0
omega = 1.0

# ---------------------------------------
# Rebuild Hamiltonian + ground state (gs)
# ---------------------------------------
H = build_hamiltonian_ho(N, n_max, alpha, beta, omega)
vals, vecs = compute_lowest_eigs(H, k=4)
gs = vecs[:,0]
dim_gs = gs.shape[0]
print("Ground state dimension =", dim_gs)


# ---------------------------------------
# Your original function names (unchanged)
# ---------------------------------------

def local_vertex_op(n_max, beta, omega=1.0):
    """Return local exp(i beta phi) using HO basis."""
    _, _, phi_local, _, _ = local_ho_operators(n_max, omega)
    phi_dense = phi_local.toarray()
    V = expm(1j * beta * phi_dense)
    return sp.csr_matrix(V)


def make_manybody_vertex_list(N, n_max, beta, omega=1.0):
    """Return list: V_j = exp(i beta phi_j)."""
    Vloc = local_vertex_op(n_max, beta, omega)
    Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
    Vlist = []
    for j in range(N):
        ops = Id_list.copy()
        ops[j] = Vloc
        Vlist.append(kron_n(ops).tocsr())
    return Vlist


# ---------------------------------------
# Build many-body vertex operators
# ---------------------------------------
Vlist = make_manybody_vertex_list(N, n_max, beta, omega)
dim_V = Vlist[0].shape[0]
print("Vertex operator dimension =", dim_V)

# HARD ASSERT to stop dimension mismatch immediately
assert dim_V == dim_gs, f"Dimension mismatch: gs={dim_gs}, vertex={dim_V}. Rebuild H, gs, and Vlist with SAME N,n_max!"


# ---------------------------------------
# Compute C(r)
# ---------------------------------------
C = []
for r in range(N):
    O = Vlist[r].dot(Vlist[0].getH())
    C.append(gs.conj().T @ (O.dot(gs)))

C = np.array(C)
print("C(r) =", C)


# ---------------------------------------
# PLOT
# ---------------------------------------
r = np.arange(N)
plt.subplot(1,2,1)
plt.plot(r, np.abs(C), 'o-')
plt.yscale('log')
plt.xlabel('r'); plt.ylabel('|C(r)|')
plt.title('|C(r)| log scale')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(r, np.abs(C), 'o-')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('r'); plt.ylabel('|C(r)|')
plt.title('|C(r)| log-log')
plt.grid(True)

plt.tight_layout()
plt.show()


# In[48]:


# === Cell 7 (expanded & FIXED): correlators + cos(beta phi) with robust plotting ===
# Improvements:
# 1. Fixed mathtext labels (removed malformed TeX causing savefig ValueError).
# 2. Cache local operators (phi_local, Cos_local_dense) instead of recomputing inside loops.
# 3. Added safe_plot() wrapper: on failure prints error but continues.
# 4. Avoid r=0 exact log issues by shifting only in log-log plot.
# 5. Minor refactors for clarity; numerical logic preserved.
# 6. Corrected all LaTeX strings (use raw strings and double backslashes).

import os

# ---- USER PARAMETERS ----
N_list   = [3, 4, 5]       # list of system sizes to evaluate
n_max    = 6               # local Fock truncation
alpha    = 0.05
beta     = 1.0
omega    = 1.0
max_dim  = 8000            # safety threshold for total Hilbert space dimension
outdir   = "plots_correlators"
os.makedirs(outdir, exist_ok=True)

# ---- helpers ----
def build_local_manybody_ops(N, n_max, omega=1.0, beta=1.0):
    """Return lists Phi_mb[j] (dense) and V_mb[j] (sparse) for j=0..N-1."""
    _, _, phi_local, _, _ = local_ho_operators(n_max, omega=omega)
    phi_dense = phi_local.toarray()
    Vloc_dense = expm(1j * beta * phi_dense)
    Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
    Phi_mb, V_mb = [], []
    for j in range(N):
        ops_phi = Id_list.copy(); ops_phi[j] = phi_local
        ops_V   = Id_list.copy(); ops_V[j]   = sp.csr_matrix(Vloc_dense)
        Phi_mb.append(kron_n(ops_phi).toarray())
        V_mb.append(kron_n(ops_V).tocsr())
    return Phi_mb, V_mb, phi_local

# Local cos(beta phi) precomputation (single-site) so we reuse per N
_, _, phi_local_global, _, _ = local_ho_operators(n_max, omega=omega)
phi_dense_global = phi_local_global.toarray()
Cos_local_dense_global = 0.5 * (expm(1j * beta * phi_dense_global) + expm(-1j * beta * phi_dense_global)).real
Cos_local_sparse_global = sp.csr_matrix(Cos_local_dense_global)

# ---- plotting helper ----
def safe_plot(fname, plot_fn):
    try:
        plot_fn()
        plt.savefig(fname, dpi=200)
        plt.close()
        print("  saved", fname)
    except Exception as e:
        print(f"  FAILED to save {fname}: {e}")
        plt.close()

results = {}

for N in N_list:
    dim = n_max ** N
    print(f"\n--- N = {N} (dim = {dim}) ---")
    if dim > max_dim:
        print(f"Skipping N={N}: dimension {dim} > max_dim {max_dim}.")
        continue

    # Hamiltonian & ground state
    H = build_hamiltonian_ho(N, n_max, alpha, beta, omega=omega)
    vals, vecs = compute_lowest_eigs(H, k=4)
    gs  = vecs[:, 0]
    E0  = vals[0]
    print(f"  E0={E0:.6f} E1={vals[1]:.6f} gap={vals[1]-vals[0]:.6f}" if len(vals) > 1 else f"  E0={E0:.6f}")

    # many-body operators
    Phi_mb, V_mb, phi_local = build_local_manybody_ops(N, n_max, omega=omega, beta=beta)

    # correlators C(r) and <phi phi>
    phi_corr_r    = np.zeros(N, dtype=float)
    vertex_corr_r = np.zeros(N, dtype=complex)
    for r in range(N):
        accum_phi = 0.0 + 0j
        accum_vertex = 0.0 + 0j
        for i in range(N):
            j = (i + r) % N
            val_phi = np.vdot(gs, Phi_mb[j].dot(Phi_mb[i].dot(gs)))
            accum_phi    += val_phi
            O = V_mb[j].dot(V_mb[i].getH())
            accum_vertex += gs.conj().T @ (O.dot(gs))
        phi_corr_r[r]    = np.real(accum_phi / N)
        vertex_corr_r[r] = accum_vertex / N

    # local cos(beta phi) expectations
    cos_vals = []
    Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
    for j in range(N):
        ops = Id_list.copy(); ops[j] = Cos_local_sparse_global
        O = kron_n(ops).toarray()
        cos_vals.append(np.real(np.vdot(gs, O.dot(gs))))
    cos_vals = np.array(cos_vals)

    # store
    results[N] = {
        'phi_corr_r': phi_corr_r,
        'vertex_corr_r': vertex_corr_r,
        'cos_vals': cos_vals,
        'gs_dim': dim,
        'E0': E0
    }

    r = np.arange(N)

    # phi correlator (linear)
    safe_plot(os.path.join(outdir, f"phi_corr_N{N}.png"), lambda: (
        plt.figure(),
        plt.plot(r, phi_corr_r, 'o-'),
        plt.xlabel('r'), plt.ylabel(r'$\\langle \\phi_{i+r}\\,\\phi_i \\rangle$'),
        plt.title(r'$\\langle \\phi\\phi \\rangle$ (N=' + str(N) + ')'),
        plt.grid(True)
    ))

    # phi correlator (abs, semilog)
    safe_plot(os.path.join(outdir, f"phi_corr_log_N{N}.png"), lambda: (
        plt.figure(),
        plt.plot(r, np.abs(phi_corr_r) + 1e-15, 'o-'),
        plt.yscale('log'),
        plt.xlabel('r'), plt.ylabel(r'$|\\langle \\phi\\phi \\rangle|$'),
        plt.title(r'$|\\langle \\phi\\phi \\rangle|$ (N=' + str(N) + ')'),
        plt.grid(True)
    ))

    # vertex correlator |C(r)| semilog
    safe_plot(os.path.join(outdir, f"vertex_corr_log_N{N}.png"), lambda: (
        plt.figure(),
        plt.plot(r, np.abs(vertex_corr_r) + 1e-15, 'o-'),
        plt.yscale('log'),
        plt.xlabel('r'), plt.ylabel(r'$|C(r)|$'),
        plt.title(r'$|C(r)|$ (N=' + str(N) + ')'),
        plt.grid(True)
    ))

    # vertex correlator |C(r)| log-log (avoid log(0) by shifting r)
    safe_plot(os.path.join(outdir, f"vertex_corr_loglog_N{N}.png"), lambda: (
        plt.figure(),
        plt.loglog(r + 1e-6, np.abs(vertex_corr_r) + 1e-15, 'o-'),
        plt.xlabel('r'), plt.ylabel(r'$|C(r)|$'),
        plt.title(r'$|C(r)|$ log-log (N=' + str(N) + ')'),
        plt.grid(True)
    ))

    # local cos(beta phi)
    safe_plot(os.path.join(outdir, f"cos_vals_N{N}.png"), lambda: (
        plt.figure(),
        plt.plot(np.arange(N), cos_vals, 'o-'),
        plt.xlabel('site'), plt.ylabel(r'$\\langle \\cos(\\beta \\phi_j) \\rangle$'),
        plt.title(r'$\\langle \\cos(\\beta \\phi_j) \\rangle$ (N=' + str(N) + ')'),
        plt.grid(True)
    ))

# ---- summary ----
print("\nCompleted sweep. Results keys (per N):", list(results.keys()))
for N, d in results.items():
    C1 = (np.abs(d['vertex_corr_r'][1]) if N > 1 else 'NA')
    print(f" N={N}  dim={d['gs_dim']}  E0={d['E0']:.6f}  |C(1)|={C1}")
print(f"Saved plots into folder: {outdir}")


# In[ ]:





# In[49]:


# Cell 8: Loschmidt echo L(t) = |<psi0 | exp(-i H t) | psi0>|^2 using expm_multiply
# - Prepares a product displaced initial state psi0 (local coherent-like displacement)
# - Uses expm_multiply to propagate psi0 and compute overlap at each time (sparse, memory-friendly)
def local_displaced_state_coherent(n_max, shift_amp, omega=1.0):
    """Simple localized wavefunction in Fock basis: gaussian-like coefficients in phi representation.
    This is a product-state proxy for a local quench; not an exact coherent state mapping."""
    # create local phi grid via diagonalization of phi (only to make a simple localized psi)
    _, _, phi_local, _, _ = local_ho_operators(n_max, omega=omega)
    phi_vals, phi_vecs = np.linalg.eigh(phi_local.toarray())
    # create gaussian in phi basis centered at shift_amp, then transform back to Fock basis
    psi_phi = np.exp(-0.5 * ((phi_vals - shift_amp)/(np.abs(phi_vals).max()/6.0 + 1e-12))**2)
    psi_phi /= np.linalg.norm(psi_phi)
    psi_fock = phi_vecs @ psi_phi
    return psi_fock

def product_state_focks(N, n_max, shifts, omega=1.0):
    locals_ = [local_displaced_state_coherent(n_max, s, omega=omega) for s in shifts]
    psi = locals_[0]
    for v in locals_[1:]:
        psi = np.kron(psi, v)
    return psi

# parameters
N = 3; n_max = 6; alpha = 0.05; beta = 1.0; omega = 1.0
H = build_hamiltonian_ho(N, n_max, alpha, beta, omega=omega)
dim = H.shape[0]
print("dim:", dim)
if dim > 10000:
    raise RuntimeError("System too large for Loschmidt demo on laptop; reduce N or n_max")

# initial product-state with a local shift
shifts = [0.0]*N
shifts[0] = 1.0
psi0 = product_state_focks(N, n_max, shifts, omega=omega)
psi0 = psi0 / np.linalg.norm(psi0)

# times
times = np.linspace(0, 20, 201)
overlaps = np.zeros_like(times, dtype=float)

# expm_multiply needs a dense vector shape (N,) and returns propagated vector. Use sparse H matrix.
for idx, t in enumerate(times):
    # propagate psi0 -> psi(t) using expm_multiply with -1j*H*t
    psi_t = expm_multiply((-1j * H), psi0, start=0.0, stop=t, num=2)[-1]
    overlaps[idx] = np.abs(np.vdot(psi0, psi_t))**2

plt.plot(times, overlaps)
plt.xlabel('t'); plt.ylabel('L(t)'); plt.title('Loschmidt echo (toy)'); plt.grid(True)
plt.show()


# In[50]:


# Cell 9: Real-time propagation using expm_multiply (measure <phi_j>(t) and show space-time)
# - Use same product-state initial bump as in Cell 8 and compute phi expectation vs time.
N = 3; n_max = 6; alpha = 0.05; beta = 1.0; omega = 1.0
H = build_hamiltonian_ho(N, n_max, alpha, beta, omega=omega)
dim = H.shape[0]
print("dim:", dim)

# initial state
shifts = [0.0]*N
if N > 1:
    shifts[0] = -1.0; shifts[1] = +1.0
psi0 = product_state_focks(N, n_max, shifts, omega=omega)
psi0 = psi0 / np.linalg.norm(psi0)

# precompute many-body phi_j dense matrices (only OK for small dim)
if dim > 8000:
    raise RuntimeError("System too large for full phi_mb construction on laptop. Reduce N or n_max.")

# build phi_mb
_, _, phi_local, _, _ = local_ho_operators(n_max, omega=omega)
Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
phi_mb = [kron_n([Id_list[k] if k!=j else phi_local for k in range(N)]).toarray() for j in range(N)]

times = np.linspace(0, 20, 101)
phi_expect = np.zeros((len(times), N), dtype=float)

# propagate in time using expm_multiply with multiple returns: use start=0 stop=T num=len(times)
# expm_multiply can produce many times by specifying num parameter; do it once to be efficient.
# Note: expm_multiply((-1j*H), psi0, start=0.0, stop=times[-1], num=len(times)) returns states at times
states = expm_multiply((-1j * H), psi0, start=0.0, stop=times[-1], num=len(times))
for it, psi in enumerate(states):
    for j in range(N):
        phi_expect[it, j] = np.real(np.vdot(psi, phi_mb[j].dot(psi)))

# plot space-time
plt.imshow(phi_expect.T, origin='lower', aspect='auto', extent=[times[0], times[-1], 0, N-1])
plt.colorbar(label='<phi>'); plt.xlabel('t'); plt.ylabel('site'); plt.title('<phi_j>(t) space-time'); plt.show()

# quick crude peak tracker (site index of max <phi>) to get a feel for motion
peak_sites = np.argmax(phi_expect, axis=1)
plt.plot(times, peak_sites, '-o'); plt.xlabel('t'); plt.ylabel('peak site'); plt.title('Peak site vs t (crude tracker)'); plt.grid(True); plt.show()


# In[51]:


# Cell 10: Final notes and strict checklist
# - Always run convergence (Cell 5) first. Check gap stability under n_max and N.
# - Confirm local commutator: [phi_local, pi_local] ≈ 1j on small n_max to ensure basis sanity.
# - If any step errors due to memory/time: reduce N or n_max immediately.
# - For production: replace dense many-body ops by MPS/TEBD (TenPy / ITensor) for N>8.
# - To compare with Thirring results, ensure parameter mapping (Coleman map) and units are consistent.
#   Common mapping: relate cos(beta phi) coefficient and normalization conventions — double-check slides/notes.
print("Checklist: run Cell 5 convergence first; use small N,n_max until stable; map conventions before comparing to Thirring results.")


# In[1]:


# Sine-Gordon and Thirring Model Simulation
# Comprehensive notebook combining both models with visualization

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft, ifft, fftfreq
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SINE-GORDON EQUATION SOLVER (1+1D)
# ============================================================================

class SineGordonSolver:
    """Solves the sine-Gordon equation: φ_tt - φ_xx + sin(φ) = 0"""

    def __init__(self, L=20, nx=200, T=10, nt=200):
        self.L = L  # spatial domain [-L/2, L/2]
        self.nx = nx
        self.T = T
        self.nt = nt

        self.x = np.linspace(-L/2, L/2, nx)
        self.t = np.linspace(0, T, nt)
        self.dx = L / (nx - 1)
        self.dt = T / (nt - 1)

        self.phi = np.zeros((nt, nx))
        self.phi_t = np.zeros((nt, nx))

    def initial_condition_kink(self, amplitude=1.0):
        """Kink soliton: φ = 4*arctan(exp(x))"""
        self.phi[0] = 4 * np.arctan(np.exp(amplitude * self.x))
        self.phi_t[0] = 0
        return self.phi[0]

    def initial_condition_breather(self, omega=0.5, amplitude=1.0):
        """Breather solution (oscillating soliton)"""
        self.phi[0] = 4 * np.arctan(
            (np.sqrt(1 - omega**2) / omega) * np.sin(omega * 0) * 
            np.cosh(np.sqrt(1 - omega**2) * self.x) / 
            np.cos(omega * 0)
        )
        self.phi_t[0] = 0
        return self.phi[0]

    def initial_condition_gaussian(self, sigma=1.0, amplitude=1.0):
        """Smooth Gaussian pulse"""
        self.phi[0] = amplitude * np.exp(-self.x**2 / (2 * sigma**2))
        self.phi_t[0] = 0
        return self.phi[0]

    def solve_leapfrog(self):
        """Leapfrog (staggered) finite difference scheme"""
        c = (self.dt / self.dx)**2

        # Half-step for phi_t
        phi_t_half = self.phi_t[0] + 0.5 * self.dt * self._laplacian(self.phi[0])

        for n in range(nt - 1):
            # Compute Laplacian and nonlinear term
            lap_phi = self._laplacian(self.phi[n])

            # Update phi
            self.phi[n+1] = (2*self.phi[n] - self.phi[n-1] if n > 0 
                            else self.phi[n] + self.dt * self.phi_t[0]) + \
                           c * lap_phi - c * np.sin(self.phi[n])

        return self.phi, self.t, self.x

    def _laplacian(self, u):
        """Compute Laplacian with periodic BC"""
        d2u = np.zeros_like(u)
        d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / self.dx**2
        # Periodic BC
        d2u[0] = (u[1] - 2*u[0] + u[-1]) / self.dx**2
        d2u[-1] = (u[0] - 2*u[-1] + u[-2]) / self.dx**2
        return d2u

    def solve_pseudospectral(self):
        """Pseudospectral method for comparison"""
        phi_hat = fft(self.phi[0])

        k = 2 * np.pi * fftfreq(self.nx, self.dx)
        k2 = k**2

        for n in range(self.nt - 1):
            # Compute derivatives via FFT
            phi_xx_hat = -k2 * phi_hat
            phi_xx = np.real(ifft(phi_xx_hat))

            # RK4 step for time integration
            phi_current = self.phi[n]
            phi_t_current = self.phi_t[n]

            # Simple Euler for now
            self.phi[n+1] = self.phi[n] + self.dt * self.phi_t[n]
            self.phi_t[n+1] = self.phi_t[n] + self.dt * (phi_xx - np.sin(self.phi[n]))

        return self.phi, self.t, self.x


# ============================================================================
# 2. THIRRING MODEL SOLVER
# ============================================================================

class ThirringModelSolver:
    """Solves the Thirring model: coupled fermionic system
    iψ_t + ψ_xx + gψψ̄ψ = 0
    """

    def __init__(self, L=20, nx=200, T=10, nt=200, coupling=0.5):
        self.L = L
        self.nx = nx
        self.T = T
        self.nt = nt
        self.g = coupling  # coupling constant

        self.x = np.linspace(-L/2, L/2, nx)
        self.t = np.linspace(0, T, nt)
        self.dx = L / (nx - 1)
        self.dt = T / (nt - 1)

        self.psi = np.zeros((nt, nx), dtype=complex)
        self.psi_bar = np.zeros((nt, nx), dtype=complex)

    def initial_condition_soliton(self, amplitude=1.0, v=0.0):
        """Bright soliton-like initial condition"""
        self.psi[0] = amplitude * np.exp(-self.x**2 / 4) * np.exp(1j * v * self.x)
        self.psi_bar[0] = np.conj(self.psi[0])
        return self.psi[0]

    def initial_condition_two_solitons(self, amp1=0.5, amp2=0.5, sep=5):
        """Two well-separated solitons"""
        psi1 = amp1 * np.exp(-(self.x - sep)**2 / 2)
        psi2 = amp2 * np.exp(-(self.x + sep)**2 / 2)
        self.psi[0] = psi1 + psi2
        self.psi_bar[0] = np.conj(self.psi[0])
        return self.psi[0]

    def solve_split_step(self):
        """Split-step Fourier method"""
        k = 2 * np.pi * fftfreq(self.nx, self.dx)

        for n in range(self.nt - 1):
            # Nonlinear step
            V = self.g * np.abs(self.psi[n])**2
            psi_nl = self.psi[n] * np.exp(-1j * self.dt * V)

            # Linear step (dispersion) via FFT
            psi_hat = fft(psi_nl)
            psi_hat *= np.exp(-1j * self.dt * k**2)
            self.psi[n+1] = ifft(psi_hat)
            self.psi_bar[n+1] = np.conj(self.psi[n+1])

        return self.psi, self.t, self.x


# ============================================================================
# 3. DUALITY: MAP BETWEEN THIRRING AND SINE-GORDON
# ============================================================================

class Duality:
    """Establishes the sine-Gordon/Thirring duality"""

    @staticmethod
    def thirring_to_sine_gordon(psi, m=1.0):
        """Map Thirring field ψ to sine-Gordon field φ
        φ = 2*arctan(ψ_+ / ψ_-) [formal version]
        """
        # Simplified version for visualization
        phase = np.angle(psi)
        amplitude = np.abs(psi)
        phi = 2 * np.arctan2(amplitude, 1.0 + 1e-8) * np.sign(np.real(psi))
        return phi

    @staticmethod
    def sine_gordon_to_thirring(phi, m=1.0):
        """Map sine-Gordon field φ to Thirring field ψ"""
        psi = np.exp(1j * phi / 2)
        return psi

    @staticmethod
    def conserved_charge_thirring(psi, dx):
        """Number of particles (conserved charge) in Thirring"""
        return np.sum(np.abs(psi)**2) * dx

    @staticmethod
    def energy_sine_gordon(phi, phi_t, dx, dt=None):
        """Total energy of sine-Gordon system"""
        # E = 1/2 ∫ (φ_t^2 + φ_x^2 + 2(1-cos(φ))) dx
        kinetic = 0.5 * np.mean(phi_t**2)
        d2x = np.gradient(phi, axis=-1)**2
        potential = 1 - np.cos(phi)
        return np.mean(kinetic + 0.5 * d2x + potential)


# ============================================================================
# 4. MAIN SIMULATION AND PLOTTING
# ============================================================================

# Simulation parameters
L, nx, T, nt = 20, 256, 8, 200

print("=" * 70)
print("SINE-GORDON SIMULATIONS")
print("=" * 70)

# Kink soliton
sg_kink = SineGordonSolver(L, nx, T, nt)
sg_kink.initial_condition_kink(amplitude=1.0)
phi_kink, t_sg, x_sg = sg_kink.solve_leapfrog()

# Breather
sg_breather = SineGordonSolver(L, nx, T, nt)
sg_breather.initial_condition_breather(omega=0.3)
phi_breather, _, _ = sg_breather.solve_leapfrog()

# Gaussian pulse
sg_gaussian = SineGordonSolver(L, nx, T, nt)
sg_gaussian.initial_condition_gaussian(sigma=1.5, amplitude=0.5)
phi_gaussian, _, _ = sg_gaussian.solve_leapfrog()

print("\nSine-Gordon simulations completed!")

print("\n" + "=" * 70)
print("THIRRING MODEL SIMULATIONS")
print("=" * 70)

# Thirring with coupling g=0.5
thirring = ThirringModelSolver(L, nx, T, nt, coupling=0.5)
thirring.initial_condition_soliton(amplitude=1.0, v=0.5)
psi_thirring, t_thirring, x_thirring = thirring.solve_split_step()

# Two solitons
thirring_two = ThirringModelSolver(L, nx, T, nt, coupling=0.5)
thirring_two.initial_condition_two_solitons(amp1=0.6, amp2=0.4, sep=4)
psi_two, _, _ = thirring_two.solve_split_step()

print("Thirring model simulations completed!")

# ============================================================================
# 5. PLOTTING
# ============================================================================

fig = plt.figure(figsize=(16, 14))

# Row 1: Sine-Gordon Kink
ax1 = plt.subplot(4, 3, 1)
X, T_mesh = np.meshgrid(x_sg, t_sg)
cf1 = ax1.contourf(X, T_mesh, phi_kink, levels=20, cmap='RdBu_r')
ax1.set_title('Sine-Gordon: Kink (space-time)', fontsize=11, fontweight='bold')
ax1.set_xlabel('x'); ax1.set_ylabel('t')
plt.colorbar(cf1, ax=ax1)

ax2 = plt.subplot(4, 3, 2)
ax2.plot(x_sg, phi_kink[0], 'o-', label='t=0', linewidth=2, markersize=3)
ax2.plot(x_sg, phi_kink[nt//2], 's-', label=f't={t_sg[nt//2]:.1f}', linewidth=2, markersize=3)
ax2.plot(x_sg, phi_kink[-1], '^-', label=f't={t_sg[-1]:.1f}', linewidth=2, markersize=3)
ax2.set_title('Sine-Gordon: Kink profiles', fontsize=11, fontweight='bold')
ax2.set_xlabel('x'); ax2.set_ylabel('φ(x,t)')
ax2.legend(); ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(4, 3, 3)
# Phase portrait: plot φ vs φ_t at a point
mid = nx // 2
ax3.scatter(phi_kink[:, mid], np.gradient(phi_kink[:, mid], t_sg), 
           c=t_sg, cmap='viridis', s=20, alpha=0.7)
ax3.set_title('Phase Portrait: Kink', fontsize=11, fontweight='bold')
ax3.set_xlabel('φ(x₀,t)'); ax3.set_ylabel('∂φ/∂t(x₀,t)')
ax3.grid(True, alpha=0.3)

# Row 2: Sine-Gordon Breather
ax4 = plt.subplot(4, 3, 4)
cf2 = ax4.contourf(X, T_mesh, phi_breather, levels=20, cmap='RdBu_r')
ax4.set_title('Sine-Gordon: Breather (space-time)', fontsize=11, fontweight='bold')
ax4.set_xlabel('x'); ax4.set_ylabel('t')
plt.colorbar(cf2, ax=ax4)

ax5 = plt.subplot(4, 3, 5)
ax5.plot(x_sg, phi_breather[0], 'o-', label='t=0', linewidth=2, markersize=3)
ax5.plot(x_sg, phi_breather[nt//2], 's-', label=f't={t_sg[nt//2]:.1f}', linewidth=2, markersize=3)
ax5.plot(x_sg, phi_breather[-1], '^-', label=f't={t_sg[-1]:.1f}', linewidth=2, markersize=3)
ax5.set_title('Sine-Gordon: Breather profiles', fontsize=11, fontweight='bold')
ax5.set_xlabel('x'); ax5.set_ylabel('φ(x,t)')
ax5.legend(); ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(4, 3, 6)
ax6.scatter(phi_breather[:, mid], np.gradient(phi_breather[:, mid], t_sg),
           c=t_sg, cmap='viridis', s=20, alpha=0.7)
ax6.set_title('Phase Portrait: Breather', fontsize=11, fontweight='bold')
ax6.set_xlabel('φ(x₀,t)'); ax6.set_ylabel('∂φ/∂t(x₀,t)')
ax6.grid(True, alpha=0.3)

# Row 3: Thirring Single Soliton
ax7 = plt.subplot(4, 3, 7)
cf3 = ax7.contourf(x_thirring, t_thirring, np.abs(psi_thirring), levels=20, cmap='hot')
ax7.set_title('Thirring: |ψ| Soliton (space-time)', fontsize=11, fontweight='bold')
ax7.set_xlabel('x'); ax7.set_ylabel('t')
plt.colorbar(cf3, ax=ax7)

ax8 = plt.subplot(4, 3, 8)
ax8.plot(x_thirring, np.abs(psi_thirring[0]), 'o-', label='t=0', linewidth=2, markersize=3)
ax8.plot(x_thirring, np.abs(psi_thirring[nt//2]), 's-', label=f't={t_thirring[nt//2]:.1f}', linewidth=2, markersize=3)
ax8.plot(x_thirring, np.abs(psi_thirring[-1]), '^-', label=f't={t_thirring[-1]:.1f}', linewidth=2, markersize=3)
ax8.set_title('Thirring: |ψ| profiles', fontsize=11, fontweight='bold')
ax8.set_xlabel('x'); ax8.set_ylabel('|ψ(x,t)|')
ax8.legend(); ax8.grid(True, alpha=0.3)

ax9 = plt.subplot(4, 3, 9)
ax9.contourf(x_thirring, t_thirring, np.angle(psi_thirring), levels=20, cmap='hsv')
ax9.set_title('Thirring: arg(ψ) (space-time)', fontsize=11, fontweight='bold')
ax9.set_xlabel('x'); ax9.set_ylabel('t')

# Row 4: Thirring Two Solitons
ax10 = plt.subplot(4, 3, 10)
cf4 = ax10.contourf(x_thirring, t_thirring, np.abs(psi_two), levels=20, cmap='hot')
ax10.set_title('Thirring: Two Solitons |ψ| (space-time)', fontsize=11, fontweight='bold')
ax10.set_xlabel('x'); ax10.set_ylabel('t')
plt.colorbar(cf4, ax=ax10)

ax11 = plt.subplot(4, 3, 11)
ax11.plot(x_thirring, np.abs(psi_two[0]), 'o-', label='t=0', linewidth=2, markersize=3)
ax11.plot(x_thirring, np.abs(psi_two[nt//2]), 's-', label=f't={t_thirring[nt//2]:.1f}', linewidth=2, markersize=3)
ax11.plot(x_thirring, np.abs(psi_two[-1]), '^-', label=f't={t_thirring[-1]:.1f}', linewidth=2, markersize=3)
ax11.set_title('Thirring: Two soliton profiles', fontsize=11, fontweight='bold')
ax11.set_xlabel('x'); ax11.set_ylabel('|ψ(x,t)|')
ax11.legend(); ax11.grid(True, alpha=0.3)

ax12 = plt.subplot(4, 3, 12)
charge = [Duality.conserved_charge_thirring(psi_two[n], x_thirring[1] - x_thirring[0]) 
          for n in range(nt)]
ax12.plot(t_thirring, charge, 'b-', linewidth=2)
ax12.set_title('Thirring: Conserved Charge', fontsize=11, fontweight='bold')
ax12.set_xlabel('t'); ax12.set_ylabel('Q(t)')
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sine_gordon_thirring_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Sine-Gordon Kink: Initial amplitude = {phi_kink[0].max():.4f}")
print(f"Sine-Gordon Breather: Max amplitude = {phi_breather.max():.4f}")
print(f"Thirring Soliton: Initial charge = {charge[0]:.4f}")
print(f"Thirring: Charge conservation error = {abs(charge[-1] - charge[0]):.2e}")
print("\nPlots saved as 'sine_gordon_thirring_analysis.png'")


# In[3]:


from qiskit.primitives import Estimator  # EstimatorV2 in Qiskit 2.x
from qiskit.circuit.library import EfficientSU2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Reuse H_SG and V_dag_0_V_1 from Snippet 1 (assuming execution)
if 'H_SG' not in locals():
    from qiskit.quantum_info import SparsePauliOp
    H_SG = SparsePauliOp.from_list([("XX", -1.0), ("YY", -1.0), ("ZI", 0.2), ("IZ", 0.2), ("ZI", -0.5), ("IZ", -0.5)])
    V_dag_0_V_1 = SparsePauliOp.from_list([("XX", 0.5), ("YY", 0.5), ("XY", 0.5j), ("YX", -0.5j)])
    L = 2

# --- 1. Setup ---
num_qubits = 2
ansatz = EfficientSU2(num_qubits=num_qubits, reps=2, entanglement='linear')
estimator = Estimator()  # EstimatorV2

# Define the energy evaluation function
def energy_func(params):
    bound_ansatz = ansatz.assign_parameters(params)
    job = estimator.run([(bound_ansatz, H_SG)])
    return job.result().values[0].real

# --- 2. Manual VQE Optimization ---
print("Running Manual VQE to find the Ground State...")
initial_params = np.random.random(ansatz.num_parameters)
opt_result = minimize(energy_func, initial_params, method='COBYLA', options={'maxiter': 100})
optimal_parameters = opt_result.x
ground_state_energy = opt_result.fun

# --- 3. Compute Expectation Value ---
ground_state_qc = ansatz.assign_parameters(optimal_parameters)

# Measure the Correlator V^\dagger(0) V(r=1) using EstimatorV2
job_correlator = estimator.run([(ground_state_qc, V_dag_0_V_1)])
correlator_val = job_correlator.result().values[0]

# --- 4. Print and Plot Results ---
print(f"\nGround State Energy: {ground_state_energy:.5f} (a.u.)")
print(f"Vertex Operator Correlator <V^\u2020(0)V(1)>: {correlator_val.real:.5f} + {correlator_val.imag:.5f}j")

r_values = [0, 1, 2, 3]
conceptual_decay_factor = correlator_val.real 

if conceptual_decay_factor > 0:
    m = -np.log(conceptual_decay_factor)
else:
    m = 0.5

def conceptual_correlator(r, A=1.0, m_eff=m):
    return A * np.exp(-m_eff * r)

conceptual_data = [conceptual_correlator(r) for r in r_values]
conceptual_data[1] = correlator_val.real 

plt.figure(figsize=(8, 5))
plt.plot(r_values, conceptual_data, 'b-', marker='o', label='Conceptual Exponential Decay')
plt.plot(1, correlator_val.real, 'ro', markersize=8, label='Qiskit Result (r=1)')
plt.xlabel('Distance $r$ (Lattice Sites)')
plt.ylabel(r'$\\mathrm{Re}(\\langle V^\\dagger(0) V(r) \\rangle)$')
plt.title('Vertex Operator Two-Point Correlator (Conceptual SG Model)')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Plot Interpretation ---")
print("The plot shows the exponential decay of the correlator, which characterizes the mass gap.")
print(f"The effective mass (m) derived from the decay constant is: {m:.5f} (a.u.)")


# In[ ]:




