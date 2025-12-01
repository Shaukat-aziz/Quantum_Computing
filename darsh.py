import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os # Added for directory creation

# --- helpers for fits ---
def power_law(r, A, eta):
    """Power-law function: A * r^(-eta)"""
    # Handle log(0) case for safety, though typically r > 0 in usage
    return A * np.power(r, -eta) 

def expo(r, B, xi):
    """Exponential function: B * exp(-r/xi)"""
    return B * np.exp(-r/xi)

def linear(x, a, b):
    """Linear function: a * x + b"""
    return a * x + b

# --- Correlator plotting: Czz vs r (log-log + overlay fits) ---
def plot_corr_Czz(r, Czz, out_pdf="figs/corr_Czz_vs_r.pdf"):
    r = np.array(r)
    Czz = np.array(Czz)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    
    # pick fit window (avoid r=0 and extreme edges)
    r_nonzero = r[r > 0]
    if r_nonzero.size < 4: # Need at least a few points for a decent fit
        print(f"Warning: Insufficient non-zero data points ({r_nonzero.size}) for robust fitting. Skipping fits.")
        popt, pexp = None, None
        mask = r > 0
    else:
        r_min_fit = max(3, r_nonzero.min())
        r_max_fit = r_nonzero.max() / 2
        mask = (r >= r_min_fit) & (r <= r_max_fit)
        
        if np.sum(mask) < 2:
            print("Warning: Not enough data points in the fit window [3, max(r)/2]. Using all points > 0.")
            mask = r > 0
            if np.sum(mask) < 2:
                 popt, pexp = None, None
            
        try:
            # Power-law fit
            popt, _ = curve_fit(power_law, r[mask], np.abs(Czz[mask]), p0=[1.0, 0.5], maxfev=5000)
        except RuntimeError:
            print("Warning: Power-law fit failed.")
            popt = None
            
        # Exponential fit (optional)
        pexp = None
        try:
            pexp, _ = curve_fit(expo, r[mask], np.abs(Czz[mask]), p0=[1.0, 10.0], maxfev=5000)
        except RuntimeError:
            print("Warning: Exponential fit failed.")
            pexp = None

    plt.figure(figsize=(7.2, 4.0))
    plt.loglog(r[r>0], np.abs(Czz[r>0]), 'o', markersize=4, label='$|C_{zz}(r)|$')
    
    # Plot Power-Law Fit
    if popt is not None and np.sum(mask) >= 2:
        rfit = np.linspace(r[mask].min(), r[mask].max(), 200)
        # Use plain f-string (raw f-string caused syntax issue with TeX backslashes)
        plt.loglog(rfit, power_law(rfit, *popt), '-', 
                   label=f"power-law fit: $A r^{{-\\eta}}$, $\\eta={popt[1]:.3f}$") 
    
    # Plot Exponential Fit
    if pexp is not None and np.sum(mask) >= 2:
        # Check if rfit was created by power-law fit; if not, create it
        if popt is None:
            rfit = np.linspace(r[mask].min(), r[mask].max(), 200)
        plt.loglog(rfit, expo(rfit, *pexp), '--', label=f"exp fit: $\\xi={pexp[1]:.2f}$")
        
    plt.xlabel('Distance $r$')
    plt.ylabel('$|C_{zz}(r)|$')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print("Saved:", out_pdf)

# --- Entanglement entropy vs log(L) ---
def plot_entropy_vs_logL(ell, S, out_pdf="figs/entropy_vs_logL.pdf", logbase=np.e):
    ell = np.array(ell)
    S = np.array(S)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    
    if logbase == np.e:
        x = np.log(ell)
        xlabel = 'ln($\\ell$)'
    else:
        x = np.log10(ell)
        xlabel = 'log10($\\ell$)'
        
    try:
        popt, _ = curve_fit(linear, x, S)
    except RuntimeError:
        print("Warning: Linear fit for entropy failed.")
        return
        
    slope = popt[0]
    c_est = 6 * slope  # for open chain slope = c/6
    
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(x, S, 'o', markersize=4, label='data')
    plt.plot(x, linear(x, *popt), '-', label=fr'linear fit (slope={slope:.4f}), $c \approx {c_est:.3f}$')
    plt.xlabel(xlabel)
    plt.ylabel('Entanglement entropy $S(\\ell)$')
    plt.grid(True, ls=':', alpha=0.5)
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print("Saved:", out_pdf)


# --- Example Data and Execution Block ---

if __name__ == "__main__":
    # --- 1. Define sample data for Czz plot (Correlator vs Distance) ---
    print("Generating sample data for Czz plot...")
    L_max = 50
    # Create distance array (r), starting from 1
    r_array = np.arange(1, L_max // 2) 
    
    # Create sample Correlator data (power-law decay with some noise)
    A_true, eta_true = 10, 0.4
    Czz_ideal = power_law(r_array, A_true, eta_true)
    Czz_array = Czz_ideal * (1 + 0.1 * np.random.randn(len(r_array))) # Add noise
    
    # --- 2. Define sample data for Entropy plot (Entropy vs log(L)) ---
    print("Generating sample data for Entropy plot...")
    # System block sizes ell
    ell_array = np.array([4, 8, 16, 32, 64, 128])
    
    # Create sample Entropy data (logarithmic growth, S ~ c/6 * ln(ell))
    c_true = 1.0 # Central charge c=1 (for critical systems)
    S_ideal = (c_true / 6) * np.log(ell_array)
    S_array = S_ideal + 0.05 * np.random.randn(len(ell_array)) # Add noise
    
    # --- 3. Call the plotting functions with the sample data ---
    print("\nCalling plotting functions...")
    
    plot_corr_Czz(r_array, Czz_array, out_pdf="figs/corr_Czz_vs_r_sample.pdf")
    
    plot_entropy_vs_logL(ell_array, S_array, out_pdf="figs/entropy_vs_logL_sample.pdf")

    print("\nExecution complete. Check the 'figs/' directory for the generated PDF files.")