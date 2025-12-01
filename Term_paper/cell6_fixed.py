7 to 9# ===========================
# Cell 6 (FIXED VERSION)
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

print(f"=== Building Sine-Gordon System: N={N}, n_max={n_max}, alpha={alpha}, beta={beta} ===\n")

# ---------------------------------------
# Rebuild Hamiltonian + ground state (gs)
# ---------------------------------------
print("Step 1: Building Hamiltonian...")
H = build_hamiltonian_ho(N, n_max, alpha, beta, omega)
print(f"  Hamiltonian shape: {H.shape}, nnz: {H.nnz}")

print("\nStep 2: Computing ground state...")
vals, vecs = compute_lowest_eigs(H, k=4)
gs = vecs[:,0]
dim_gs = gs.shape[0]
print(f"  Ground state dimension: {dim_gs}")
print(f"  Ground state norm: {np.linalg.norm(gs):.6f}")
print(f"  Ground state energy E0: {vals[0]:.6f}")
if len(vals) > 1:
    print(f"  First excited state E1: {vals[1]:.6f}")
    print(f"  Energy gap (E1-E0): {vals[1]-vals[0]:.6f}")

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
print("\nStep 3: Building vertex operators...")
Vlist = make_manybody_vertex_list(N, n_max, beta, omega)
dim_V = Vlist[0].shape[0]
print(f"  Vertex operator dimension: {dim_V}")
print(f"  Number of vertex operators: {len(Vlist)}")

# Dimension validation
if dim_V != dim_gs:
    raise ValueError(f"DIMENSION MISMATCH: gs has dimension {dim_gs}, but vertex operators have dimension {dim_V}. "
                     f"Ensure H, gs, and Vlist are built with the SAME N={N} and n_max={n_max}!")

print("  ✓ Dimension check passed!")

# ---------------------------------------
# Compute C(r) = <gs| V†(0) V(r) |gs>
# ---------------------------------------
print("\nStep 4: Computing vertex operator correlator C(r)...")
C = []
for r in range(N):
    # Correct formula: C(r) = <gs| V†(0) V(r) |gs>
    # This is equivalent to: <gs| V(r)† V(0) |gs> due to Hermiticity
    # We compute: V†(0) V(r) = Vlist[0].getH() @ Vlist[r]
    V_dag_0 = Vlist[0].getH()  # V†(0)
    V_r = Vlist[r]              # V(r)
    
    # Operator product: O = V†(0) V(r)
    O = V_dag_0.dot(V_r)
    
    # Expectation value: <gs| O |gs>
    # For real ground state: gs.conj() = gs, but we keep it general
    expectation = np.vdot(gs, O.dot(gs))
    C.append(expectation)
    
    print(f"  C({r}) = {expectation.real:.6e} + {expectation.imag:.6e}j, |C({r})| = {np.abs(expectation):.6e}")

C = np.array(C)
print(f"\nCorrelator values C(r):\n{C}")

# ---------------------------------------
# PLOT
# ---------------------------------------
print("\nStep 5: Generating plots...")
r = np.arange(N)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Semi-log scale
ax1.plot(r, np.abs(C), 'o-', linewidth=2, markersize=8, color='blue')
ax1.set_yscale('log')
ax1.set_xlabel('r (distance)', fontsize=12)
ax1.set_ylabel('|C(r)|', fontsize=12)
ax1.set_title(f'Vertex Correlator |C(r)| (N={N}, n_max={n_max})\nSemi-log scale', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(r)

# Plot 2: Log-log scale (shift r to avoid log(0))
r_shifted = r + 0.1  # Small shift to avoid log(0)
ax2.plot(r_shifted, np.abs(C), 'o-', linewidth=2, markersize=8, color='red')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('r (distance)', fontsize=12)
ax2.set_ylabel('|C(r)|', fontsize=12)
ax2.set_title(f'Vertex Correlator |C(r)| (N={N}, n_max={n_max})\nLog-log scale', fontsize=11)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

print("\n=== Vertex Operator Correlation Function Computation Complete ===")
