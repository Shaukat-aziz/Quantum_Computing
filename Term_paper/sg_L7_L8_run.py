#!/usr/bin/env python3
"""Run Sine-Gordon toy mapping for L=7 and L=8 (2 qubits/site).
This script mirrors the notebook cell added to `final.ipynb` and is intended
for command-line execution and verification.
"""
import os
import numpy as np
from qiskit.quantum_info import SparsePauliOp
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

outdir = 'plots_sg_L7_L8'
os.makedirs(outdir, exist_ok=True)


def pauli_str(op_positions, ops, n_qubits):
    s = ['I'] * n_qubits
    for p, o in zip(op_positions, ops):
        s[p] = o
    return ''.join(s)


def single_pauli_term(qubit_idx, pauli, n_qubits, coeff=1.0):
    return (pauli_str([qubit_idx], [pauli], n_qubits), coeff)


def two_site_term(qi, qj, pauli, n_qubits, coeff=1.0):
    return (pauli_str([qi, qj], [pauli, pauli], n_qubits), coeff)


def build_sg_hamiltonian(L, J=1.0, h=0.2, alpha=0.5):
    n_qubits = 2 * L
    terms = []
    for j in range(L):
        qj = 2*j
        qjp = 2*((j+1)%L)
        terms.append(two_site_term(qj, qjp, 'X', n_qubits, -0.5*J))
        terms.append(two_site_term(qj, qjp, 'Y', n_qubits, -0.5*J))
    for j in range(L):
        qj = 2*j
        terms.append(single_pauli_term(qj, 'Z', n_qubits, h))
        terms.append(single_pauli_term(qj, 'Z', n_qubits, -alpha))
    H = SparsePauliOp.from_list(terms)
    return H


def build_vertex_ops(L):
    n_qubits = 2*L
    V_list = []
    for j in range(L):
        qj = 2*j
        terms = [ (pauli_str([qj], ['X'], n_qubits), 0.5), (pauli_str([qj], ['Y'], n_qubits), 0.5j) ]
        V_list.append(SparsePauliOp.from_list(terms))
    return V_list


def analyze_L(L, do_loschmidt=True):
    print(f"\n=== L = {L} (n_qubits = {2*L}) ===")
    H = build_sg_hamiltonian(L)
    n_qubits = 2*L
    dim = 2**n_qubits
    print("Hilbert space dim:", dim)
    H_sp = H.to_spmatrix()
    k = 2
    try:
        vals, vecs = spla.eigsh(H_sp, k=k, which='SA', tol=1e-6, maxiter=5000)
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
    except Exception as e:
        print('eigsh failed, falling back to dense diagonalization:', e)
        H_dense = H.to_matrix()
        evals, evecs = np.linalg.eigh(H_dense)
        idx = np.argsort(evals)[:k]
        vals = evals[idx]
        vecs = evecs[:, idx]
    E0 = vals[0].real
    gap = (vals[1]-vals[0]).real if len(vals)>1 else np.nan
    psi_gs = vecs[:,0]
    print(f"E0={E0:.6f}, gap={gap:.6f}")
    V_list = build_vertex_ops(L)
    V_sp_list = [V.to_spmatrix() for V in V_list]
    C = np.zeros(L, dtype=complex)
    for r in range(L):
        accum = 0+0j
        for j in range(L):
            jp = (j + r) % L
            O = V_sp_list[jp].dot(V_sp_list[j].conj().T)
            accum += psi_gs.conj().T @ (O.dot(psi_gs))
        C[r] = accum / L
    cos_vals = np.zeros(L, dtype=float)
    for j in range(L):
        qj = 2*j
        term = SparsePauliOp.from_list([(pauli_str([qj], ['Z'], 2*L), 1.0)])
        val = psi_gs.conj().T @ (term.to_spmatrix().dot(psi_gs))
        cos_vals[j] = val.real
    r = np.arange(L)
    plt.figure(); plt.plot(r, np.abs(C), 'o-'); plt.yscale('log'); plt.xlabel('r'); plt.ylabel('|C(r)|'); plt.title(f'|C(r)| (L={L})'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(outdir, f'VertexCorr_L{L}.png')); plt.close()
    plt.figure(); plt.plot(r, np.abs(C), 'o-'); plt.xlabel('r'); plt.ylabel('|C(r)|'); plt.title(f'|C(r)| linear (L={L})'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(outdir, f'VertexCorr_linear_L{L}.png')); plt.close()
    plt.figure(); plt.plot(np.arange(L), cos_vals, 'o-'); plt.xlabel('site'); plt.ylabel('<cos(beta phi)> (Z proxy)'); plt.title(f'<cos(beta phi)> (L={L})'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(outdir, f'cos_vals_L{L}.png')); plt.close()
    print('Saved correlator and cos plots for L=', L)
    if do_loschmidt and dim <= 30000:
        print('Computing Loschmidt echo (local quench) ...')
        basis = np.zeros(dim, dtype=complex)
        pos = 2*0
        idx = 1 << (n_qubits-1 - pos)
        basis[idx] = 1.0
        psi0 = basis
        times = np.linspace(0.0, 10.0, 201)
        try:
            states = expm_multiply((-1j * H_sp), psi0, start=0.0, stop=times[-1], num=len(times))
            overlaps = np.abs(np.array([np.vdot(psi0, states[i]) for i in range(len(states))]))**2
            plt.figure(); plt.plot(times, overlaps); plt.xlabel('t'); plt.ylabel('Loschmidt echo L(t)'); plt.title(f'Loschmidt echo (L={L})'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(outdir, f'Loschmidt_L{L}.png')); plt.close()
            print('Saved Loschmidt echo for L=', L)
        except Exception as e:
            print('Loschmidt computation failed:', e)
    else:
        print('Skipping Loschmidt for L=', L, ' (dim too large or disabled)')
    return {'L': L, 'E0': E0, 'gap': gap, 'C': C, 'cos_vals': cos_vals}


if __name__ == '__main__':
    results = {}
    for L in [7, 8]:
        do_l = True if L==7 else False
        try:
            results[L] = analyze_L(L, do_loschmidt=do_l)
        except Exception as e:
            print(f'Analysis for L={L} failed: {e}')
    print('\nDone. Plots saved into:', outdir)
