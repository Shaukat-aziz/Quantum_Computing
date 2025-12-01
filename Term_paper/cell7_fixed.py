# === Cell 7 (FIXED & EXPANDED): correlators + cos(beta phi) with robust plotting ===
# Improvements:
# 1. Fixed vertex correlator computation: C(r) = <gs| V†(i) V(i+r) |gs> averaged over i
# 2. Corrected operator ordering in the correlation function
# 3. Cache local operators (phi_local, Cos_local_dense) instead of recomputing inside loops
# 4. Added safe_plot() wrapper: on failure prints error but continues
# 5. Avoid r=0 exact log issues by shifting only in log-log plot
# 6. Corrected all LaTeX strings (use raw strings and double backslashes)

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
    # FIXED: Proper translation-invariant averaging
    phi_corr_r    = np.zeros(N, dtype=float)
    vertex_corr_r = np.zeros(N, dtype=complex)
    
    for r in range(N):
        accum_phi = 0.0 + 0j
        accum_vertex = 0.0 + 0j
        for i in range(N):
            j = (i + r) % N
            # Phi correlator: <phi_i phi_j>
            val_phi = np.vdot(gs, Phi_mb[i].dot(Phi_mb[j].dot(gs)))
            accum_phi += val_phi
            
            # Vertex correlator: <V†(i) V(j)> = <gs| V†(i) V(j) |gs>
            # FIXED: Correct operator ordering
            V_dag_i = V_mb[i].getH()  # V†(i)
            V_j = V_mb[j]              # V(j)
            O = V_dag_i.dot(V_j)       # V†(i) V(j)
            accum_vertex += np.vdot(gs, O.dot(gs))
            
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
        plt.xlabel('r'), plt.ylabel(r'$\langle \phi_{i+r}\,\phi_i \rangle$'),
        plt.title(r'$\langle \phi\phi \rangle$ (N=' + str(N) + ')'),
        plt.grid(True)
    ))

    # phi correlator (abs, semilog)
    safe_plot(os.path.join(outdir, f"phi_corr_log_N{N}.png"), lambda: (
        plt.figure(),
        plt.plot(r, np.abs(phi_corr_r) + 1e-15, 'o-'),
        plt.yscale('log'),
        plt.xlabel('r'), plt.ylabel(r'$|\langle \phi\phi \rangle|$'),
        plt.title(r'$|\langle \phi\phi \rangle|$ (N=' + str(N) + ')'),
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
        plt.xlabel('site'), plt.ylabel(r'$\langle \cos(\beta \phi_j) \rangle$'),
        plt.title(r'$\langle \cos(\beta \phi_j) \rangle$ (N=' + str(N) + ')'),
        plt.grid(True)
    ))

# ---- summary ----
print("\nCompleted sweep. Results keys (per N):", list(results.keys()))
for N, d in results.items():
    C1 = (np.abs(d['vertex_corr_r'][1]) if N > 1 else 'NA')
    print(f" N={N}  dim={d['gs_dim']}  E0={d['E0']:.6f}  |C(1)|={C1}")
print(f"Saved plots into folder: {outdir}")
