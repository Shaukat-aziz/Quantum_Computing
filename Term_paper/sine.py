#!/usr/bin/env python3
"""
sg_analysis.py

Single-file Sine-Gordon analysis utilities for small-lattice Exact Diagonalization (ED).
Implements:

1) Entanglement entropy (ED)
2) Vertex correlator and two-point correlator + fits
3) Loschmidt echo and DQPT detector
4) Kink-sector preparation via imaginary-time relaxation and soliton mass extraction
5) Scattering pipeline (toy wavepackets) with crude peak-tracking and time-delay -> δ(θ)
6) Convergence sweep helpers and fit utilities (power-law and exponential)
7) Coleman mapping helper (Thirring <-> Sine-Gordon) - see citation below

References:
- Coleman S., "Quantum sine-Gordon equation as the massive Thirring model", Phys. Rev. D 11, 2088 (1975).
  (Mapping relations used below follow Coleman and subsequent reviews.) See: https://link.aps.org/doi/10.1103/PhysRevD.11.2088
- Delepine et al., and review notes on the mapping (see Delepine 1998 or Benfatto 2007).
  (Search keywords: "4π/β^2 = 1 + g/π", "Coleman mapping".)

WARNING / USAGE NOTES:
- This is NOT MPS/TEBD code. It uses full many-body matrices; total dim = n_max**N.
  If n_max**N > ~8000 you may run out of RAM or be extremely slow.
- Run small test cases first: N=2..3, n_max=4..8.
- The Coleman mapping has a cutoff-dependent prefactor for alpha; we expose that as 'z' in the mapping helper.
"""

# Standard imports
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import warnings
from functools import partial

plt.rcParams.update({'figure.figsize':(6,4), 'font.size':12})


# ------------------------------
# Basic HO-basis local operators
# ------------------------------
def local_ho_operators(n_max, omega=1.0):
    """Return (a, adag, phi, pi, id) as sparse csr matrices in truncated Fock basis.

    Conventions: m=1, omega parameter (default 1).
    phi = (a + a†)/sqrt(2 omega)
    pi  = -i sqrt(omega/2) (a - a†)
    """
    n = n_max
    rows, cols, data = [], [], []
    for i in range(n-1):
        rows.append(i)
        cols.append(i+1)
        data.append(np.sqrt(i+1))
    a = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.complex128)
    adag = a.getH()

    phi = (1.0 / np.sqrt(2.0 * omega)) * (a + adag)
    pi  = -1j * np.sqrt(omega / 2.0) * (a - adag)

    # numeric Hermitian cleanup
    phi = 0.5 * (phi + phi.getH())
    pi  = 0.5 * (pi + pi.getH())

    I = sp.eye(n, format='csr', dtype=np.complex128)
    return a.tocsr(), adag.tocsr(), phi.tocsr(), pi.tocsr(), I


def kron_n(ops):
    """Compute many-body Kronecker product of operator list ops (sparse)."""
    out = sp.csr_matrix(1.0, dtype=np.complex128)
    for A in ops:
        if not sp.isspmatrix_csr(A):
            A = sp.csr_matrix(A)
        out = sp.kron(out, A, format='csr')
    return out


# ------------------------------
# Build full SG Hamiltonian
# ------------------------------
def build_hamiltonian_ho(N, n_max, alpha, beta, omega=1.0):
    """Assemble many-body Sine-Gordon Hamiltonian on N sites with HO truncation n_max.

    H = sum_j [ 1/2 pi_j^2 + 1/2 (phi_{j+1}-phi_j)^2 + alpha * (1 - cos(beta phi_j)) ]
    Periodic boundary conditions assumed.
    """
    a, adag, phi_local, pi_local, I_local = local_ho_operators(n_max, omega=omega)
    Kin_local = 0.5 * (pi_local @ pi_local)

    # local cosine via dense exponentiation of phi
    phi_dense = phi_local.toarray()
    Cos_local_dense = 0.5 * (expm(1j * beta * phi_dense) + expm(-1j * beta * phi_dense)).real
    Cos_local = sp.csr_matrix(Cos_local_dense)

    Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
    H = sp.csr_matrix((n_max**N, n_max**N), dtype=np.complex128)

    # single-site kinetic and cosine
    for j in range(N):
        ops = Id_list.copy(); ops[j] = Kin_local
        H += kron_n(ops)
        ops = Id_list.copy(); ops[j] = alpha * (I_local - Cos_local)
        H += kron_n(ops)

    # gradient term 0.5*(phi_{j+1} - phi_j)^2
    for j in range(N):
        jp = (j + 1) % N
        ops = Id_list.copy(); ops[j] = 0.5 * (phi_local @ phi_local)
        H += kron_n(ops)
        ops = Id_list.copy(); ops[jp] = 0.5 * (phi_local @ phi_local)
        H += kron_n(ops)
        ops = Id_list.copy(); ops[j] = phi_local; ops[jp] = phi_local
        H += -1.0 * kron_n(ops)

    H = 0.5 * (H + H.getH())
    return H.tocsr()


# ------------------------------
# ED helpers: lowest eigenpairs
# ------------------------------
def compute_lowest_eigs(H, k=6):
    """Return sorted lowest k eigenvalues and eigenvectors (sparse eigsh)."""
    k = min(k, H.shape[0] - 1)
    if H.shape[0] <= 2000:
        # when dim small, do dense diagonalization for robustness
        evals, evecs = np.linalg.eigh(H.toarray())
        idx = np.argsort(evals.real)[:k]
        return evals[idx], evecs[:, idx]
    else:
        vals, vecs = spla.eigsh(H, k=k, which='SA', tol=1e-8, maxiter=5000)
        idx = np.argsort(vals.real)
        return vals.real[idx], vecs[:, idx]


# ------------------------------
# Entanglement entropy (ED)
# ------------------------------
def bipartite_entropy_from_state(psi, N, n_max, cut):
    """Compute bipartite von Neumann entropy S = -Tr rho_A log rho_A for cut after 'cut' sites.

    psi: many-body state vector (1D array) of length n_max**N
    cut: integer in 1..N-1 indicating subsystem A size
    Returns S (float) and eigenvalues of reduced density matrix.
    """
    dimA = n_max**cut
    dimB = n_max**(N - cut)
    if psi.size != dimA * dimB:
        raise ValueError("psi dimension mismatch for provided N and n_max.")

    # reshape to matrix of shape (dimA, dimB)
    rho_psi = psi.reshape((dimA, dimB))
    # reduced density matrix rho_A = rho_psi @ rho_psi^dagger
    rhoA = rho_psi @ rho_psi.conj().T
    # eigenvalues of rhoA
    evals = np.linalg.eigvalsh(rhoA)
    # numerical floor
    evals = np.real(evals)
    evals[evals < 0] = 0.0
    # von Neumann entropy
    nonzero = evals[evals > 1e-12]
    S = -np.sum(nonzero * np.log(nonzero))
    return S, evals


# ------------------------------
# Correlators and fits
# ------------------------------
def build_manybody_local_ops(N, n_max, local_op):
    """Return list of many-body operators with given local_op placed on each site."""
    Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
    ops_list = []
    for j in range(N):
        ops = Id_list.copy()
        ops[j] = local_op
        ops_list.append(kron_n(ops).tocsr())
    return ops_list


def vertex_correlator(gs, N, n_max, beta, omega=1.0):
    """Compute C(r) = < e^{i beta phi_r} e^{-i beta phi_0} > for r=0..N-1, using ground state gs."""
    # build local vertex op
    _, _, phi_local, _, _ = local_ho_operators(n_max, omega)
    Vloc = sp.csr_matrix(expm(1j * beta * phi_local.toarray()))
    Vlist = build_manybody_local_ops(N, n_max, Vloc)
    dim = gs.shape[0]
    if any(V.shape[0] != dim for V in Vlist):
        raise ValueError("Dimension mismatch between GS and vertex operators.")
    C = np.zeros(N, dtype=complex)
    for r in range(N):
        O = Vlist[r].dot(Vlist[0].getH())
        C[r] = gs.conj().T @ (O.dot(gs))
    return C


def phi_two_point(gs, N, n_max, omega=1.0):
    """Compute <phi_j phi_0> for j=0..N-1."""
    _, _, phi_local, _, _ = local_ho_operators(n_max, omega)
    Philist = build_manybody_local_ops(N, n_max, phi_local)
    dim = gs.shape[0]
    if any(P.shape[0] != dim for P in Philist):
        raise ValueError("Dimension mismatch between GS and phi operators.")
    corr = np.zeros(N)
    for j in range(N):
        corr[j] = np.real(np.vdot(gs, Philist[j].dot(Philist[0].dot(gs))))
    return corr


# Fit functions
def power_law(r, a, b):
    return a * (r ** (-b))


def exponential(r, A, xi):
    return A * np.exp(-r / xi)


def fit_correlator_abs(rvals, cabs):
    """Fit absolute correlator to power law and exponential, return fit params and covariances."""
    # fit power law for r>=1 to avoid r=0 divergence
    valid = rvals >= 1
    try:
        popt_pl, pcov_pl = curve_fit(power_law, rvals[valid], cabs[valid], p0=[cabs[1], 0.5], maxfev=5000)
    except Exception as e:
        popt_pl, pcov_pl = None, None
    try:
        popt_exp, pcov_exp = curve_fit(exponential, rvals, cabs, p0=[cabs[0], 2.0], maxfev=5000)
    except Exception as e:
        popt_exp, pcov_exp = None, None
    return (popt_pl, pcov_pl), (popt_exp, pcov_exp)


# ------------------------------
# Loschmidt echo and DQPT detector
# ------------------------------
def loschmidt_echo(H, psi0, times):
    """Compute L(t) = |<psi0|exp(-i H t)|psi0>|^2 using expm_multiply for each time.
    Returns array L(t) for times (1D array).
    """
    dim = H.shape[0]
    if dim > 20000:
        raise RuntimeError("System too large for Loschmidt on laptop; reduce N or n_max.")
    overlaps = np.zeros(len(times), dtype=float)
    for i, t in enumerate(times):
        psi_t = expm_multiply((-1j * H), psi0, start=0.0, stop=t, num=2)[-1]
        overlaps[i] = np.abs(np.vdot(psi0, psi_t))**2
    return overlaps


def loschmidt_rate(Lt, N):
    """Compute rate lambda(t) = - (1/N) log L(t)."""
    # clip tiny values to avoid -inf
    Lt = np.clip(Lt, 1e-300, 1.0)
    return - (1.0 / N) * np.log(Lt)


def detect_dqpts(times, rate):
    """Detect candidate DQPT times by locating cusp-like features.
    We use numerical derivative and find peaks in |dλ/dt|.
    Returns times where candidate cusps occur (coarse detector).
    """
    dt = times[1] - times[0]
    d_rate = np.gradient(rate, dt)
    # absolute derivative peaks above a threshold (mean + 2 sigma)
    thresh = np.mean(np.abs(d_rate)) + 2.0 * np.std(np.abs(d_rate))
    peaks = np.where(np.abs(d_rate) > thresh)[0]
    # convert to times and unique
    times_peaks = np.unique(np.round(times[peaks], 6))
    return times_peaks


# ------------------------------
# Kink-sector prep (imaginary-time relaxation)
# ------------------------------
def prepare_kink_initial_product_state(N, n_max, phi_profile, omega=1.0):
    """Create a product-state many-body vector from a list phi_profile (length N) giving preferred phi expectation per site.

    phi_profile: array-like length N with phi shifts (floats).
    Returns normalized many-body vector psi0.
    """
    _, _, phi_local, _, _ = local_ho_operators(n_max, omega)
    # diagonalize local phi to get basis and construct gaussian in phi basis
    vals, vecs = np.linalg.eigh(phi_local.toarray())
    local_states = []
    for shift in phi_profile:
        psi_phi = np.exp(-0.5 * ((vals - shift) / (np.abs(vals).max() / 6.0 + 1e-12))**2)
        psi_phi /= np.linalg.norm(psi_phi)
        psi_fock = vecs @ psi_phi
        local_states.append(psi_fock)
    psi = local_states[0]
    for v in local_states[1:]:
        psi = np.kron(psi, v)
    psi = psi / np.linalg.norm(psi)
    return psi


def imaginary_time_relax(H, psi0, beta_imag=10.0, nsteps=200):
    """Simple imaginary-time propagation using small step propagation:
    psi_{k+1} = exp(-dτ H) psi_k (renormalized).
    beta_imag: total imaginary time
    nsteps: number of steps
    Returns relaxed state psi and approximate energy E = <H>.
    """
    dtau = beta_imag / nsteps
    # use expm_multiply stepping; for small dims can use dense expm
    psi = psi0.copy().astype(np.complex128)
    for _ in range(nsteps):
        psi = expm_multiply((-dtau * H), psi, start=0.0, stop=dtau, num=2)[-1]  # note sign for imaginary time
        # renormalize
        psi /= np.linalg.norm(psi)
    E = np.real(np.vdot(psi, H.dot(psi)))
    return psi, E


def compute_kink_mass(H_ground, psi_kink, E_kink=None):
    """Estimate kink mass: difference between kink energy and ground-state energy.
    H_ground: Hamiltonian with same N/n_max, ideally in topologically trivial sector.
    psi_kink: kink state vector (relaxed)
    If E_kink is None, compute expectation value.
    """
    if E_kink is None:
        E_kink = np.real(np.vdot(psi_kink, H_ground.dot(psi_kink)))
    # compute ground state energy
    vals, vecs = compute_lowest_eigs(H_ground, k=1)
    E0 = vals[0]
    return E_kink - E0, E0, E_kink


# ------------------------------
# Scattering pipeline (toy)
# ------------------------------
def gaussian_peak_fit(x, y):
    """Fit a gaussian to (x,y) around the maximum. Returns center, amplitude, width."""
    # coarse center at max
    imax = np.argmax(y)
    # select small window
    left = max(0, imax - 3)
    right = min(len(x) - 1, imax + 3)
    xs = x[left:right+1]
    ys = y[left:right+1]
    # avoid failures: fit parabola if few points
    if len(xs) < 4:
        # quadratic interpolation for sub-pixel peak
        if len(xs) >= 3:
            coeffs = np.polyfit(xs, ys, 2)
            a, b, c = coeffs
            center = -b / (2*a) if a != 0 else xs[np.argmax(ys)]
            amp = np.max(ys)
            width = np.sqrt(-amp / a) if a != 0 and -amp/a > 0 else 1.0
            return center, amp, width
        else:
            return xs[np.argmax(ys)], np.max(ys), 1.0
    try:
        logy = np.log(np.clip(ys, 1e-12, None))
        coeffs = np.polyfit(xs, logy, 2)  # log gaussian -> quadratic
        A = coeffs[2]
        B = coeffs[1]
        C = coeffs[0]
        center = -B / (2*C)
        amp = np.exp(A - (B**2) / (4*C))
        width = np.sqrt(-1 / (2*C)) if C < 0 else 1.0
        return center, amp, width
    except Exception:
        return xs[np.argmax(ys)], np.max(ys), 1.0


def scattering_time_delay(H, psi0, times, N, n_max, phi_mb=None):
    """Run time evolution of psi0 over 'times' using expm_multiply (returns states),
    compute <phi_j>(t), track peak centers (for single bump per side), and estimate time-delay.
    This is a heuristic toy implementation for small systems only.
    Returns: times, phi_expect array (len(times) x N), peak_centers, crude delta_t estimate.
    """
    dim = H.shape[0]
    if dim > 15000:
        raise RuntimeError("System too large for scattering demo on laptop; reduce N or n_max.")
    # get states at times
    states = expm_multiply((-1j * H), psi0, start=0.0, stop=times[-1], num=len(times))
    # build phi many-body operators if not provided
    if phi_mb is None:
        _, _, phi_local, _, _ = local_ho_operators(n_max)
        Id_list = [sp.eye(n_max, format='csr') for _ in range(N)]
        phi_mb = [kron_n([Id_list[k] if k != j else phi_local for k in range(N)]).toarray() for j in range(N)]
    phi_expect = np.zeros((len(times), N))
    for it, psi in enumerate(states):
        for j in range(N):
            phi_expect[it, j] = np.real(np.vdot(psi, phi_mb[j].dot(psi)))
    # track peaks (crude): for each time find center in site+subpixel space using gaussian fit over site index
    x_sites = np.arange(N)
    peak_centers = np.zeros(len(times))
    for it in range(len(times)):
        y = phi_expect[it, :]
        center, _, _ = gaussian_peak_fit(x_sites, y)
        peak_centers[it] = center
    # estimate incoming slope vs time and outgoing slope by fitting first and last quarters
    n = len(times)
    q = max(3, n // 6)
    # fit linear to first q and last q points
    p_in = np.polyfit(times[:q], peak_centers[:q], 1)
    p_out = np.polyfit(times[-q:], peak_centers[-q:], 1)
    # naive delta_t estimate: intersect lines to get time shift
    # p_in: y = m1 t + b1; p_out: y = m2 t + b2. For same y, t_intersect = (b2 - b1)/(m1 - m2)
    if abs(p_in[0] - p_out[0]) < 1e-12:
        delta_t = 0.0
    else:
        delta_t = (p_out[1] - p_in[1]) / (p_in[0] - p_out[0])
    return times, phi_expect, peak_centers, delta_t


# ------------------------------
# Convergence helpers
# ------------------------------
def convergence_sweep_gap(N, nmax_list, alpha, beta, omega=1.0):
    """Compute ground-state gap versus n_max list for fixed N."""
    gaps = []
    for n in nmax_list:
        H = build_hamiltonian_ho(N, n, alpha, beta, omega=omega)
        vals, _ = compute_lowest_eigs(H, k=2)
        gaps.append(vals[1] - vals[0])
    return gaps


# ------------------------------
# Coleman mapping helper
# ------------------------------
def coleman_map(thirring_g, thirring_m, z_cutoff=1.0):
    """
    Map (g, m) of (massive) Thirring model to (beta, alpha) of Sine-Gordon using Coleman relation.

    Coleman relation (standard form):
        4*pi / beta^2  = 1 + g / pi
    (see Coleman 1975 and subsequent reviews)

    Thus:
        beta^2 = 4*pi / (1 + g/pi)

    The mapping of mass terms is cutoff dependent:
        im z psi_bar psi  <->  - alpha0 / lambda^2 cos(lambda phi)
    One often writes (schematically):
        alpha = const * m * z_cutoff    (cutoff-dependent)
    Here we return beta and alpha up to an overall multiplicative constant 'z_cutoff' that depends on UV regularization.
    The user must calibrate z_cutoff by matching a physical observable (mass gap) between the two lattice models.

    Returns: beta, alpha_estimate (float)
    """
    g = float(thirring_g)
    if (1.0 + g / math.pi) <= 0:
        warnings.warn("Coleman mapping gives non-positive denominator; check g. Returning nan beta.")
        beta = float('nan')
    else:
        beta = math.sqrt(4.0 * math.pi / (1.0 + g / math.pi))
    # rough alpha estimate: linear in mass times cutoff constant
    alpha = z_cutoff * float(thirring_m)
    return beta, alpha


# ------------------------------
# Simple command-line like driver examples
# ------------------------------
def demo_small_run():
    """Run a short demo showing the main pieces for a tiny system (N=2, n_max=6)."""
    N = 2
    n_max = 6
    alpha = 0.05
    beta = 1.0
    omega = 1.0
    print("Building H ...")
    H = build_hamiltonian_ho(N, n_max, alpha, beta, omega=omega)
    print("Diagonalizing ...")
    vals, vecs = compute_lowest_eigs(H, k=4)
    print("Lowest energies:", vals)
    gs = vecs[:, 0]
    # Entropy across cut 1
    S, evals = bipartite_entropy_from_state(gs, N, n_max, cut=1)
    print("Bipartite entropy (cut=1):", S)
    # correlators
    C = vertex_correlator(gs, N, n_max, beta)
    print("Vertex correlator C(r):", C)
    phi_corr = phi_two_point(gs, N, n_max)
    print("<phi_j phi_0>:", phi_corr)
    # Loschmidt test: small local shift
    psi0 = prepare_kink_initial_product_state(N, n_max, phi_profile=[-1.0, 1.0], omega=omega)
    times = np.linspace(0, 10, 51)
    L = loschmidt_echo(H, psi0, times)
    rate = loschmidt_rate(L, N)
    print("Loschmidt rate sample:", rate[:5])
    # kink prep demo (imag time)
    psi_init = prepare_kink_initial_product_state(N, n_max, phi_profile=np.linspace(-1.5, 1.5, N))
    psi_kink, E_kink = imaginary_time_relax(H, psi_init, beta_imag=5.0, nsteps=80)
    kink_mass, E0, Ek = compute_kink_mass(H, psi_kink, E_kink)
    print("kink mass (approx):", kink_mass)
    # plot Loschmidt
    plt.plot(times, rate); plt.xlabel('t'); plt.ylabel('rate'); plt.title('Loschmidt rate (demo)'); plt.grid(True); plt.show()
    # brief scattering (toy)
    psi_scatter_init = prepare_kink_initial_product_state(N, n_max, phi_profile=[-1.5, 1.5])
    tarr = np.linspace(0, 10, 51)
    times, phi_expect, peaks, dt = scattering_time_delay(H, psi_scatter_init, tarr, N, n_max)
    print("crude delta_t:", dt)
    plt.imshow(phi_expect.T, origin='lower', aspect='auto', extent=[times[0], times[-1], 0, N-1]); plt.colorbar(); plt.title('<phi>(t)'); plt.show()


# If run as script
if __name__ == "__main__":
    print("Sine-Gordon analysis demo (small). Read header warnings before scaling up.")
    demo_small_run()
