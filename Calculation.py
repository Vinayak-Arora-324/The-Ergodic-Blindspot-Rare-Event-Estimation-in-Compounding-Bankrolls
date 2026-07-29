import numpy as np
from scipy import integrate
import yfinance as yf
from scipy.stats import scoreatpercentile, levy_stable

def martingale_condition(alpha, beta, gamma,T, r, q):
    ##     Enforces the martingale condition, even if assuming that prices are not independent, mostly for Arbitrage Prevention reasons. Quiet possibly to eliminate a free option.     
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if not (-1 <= beta <= 1):
        raise ValueError("beta must be in [-1, 1]")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    
    base_drift = (r - q) * T
    
    if np.isclose(alpha, 2.0):
        sigma = gamma * np.sqrt(2)
        return base_drift - 0.5 * sigma**2 * T
    elif np.isclose(alpha, 1.0):
        return base_drift - (2/np.pi) * gamma * beta * T
    else:
        if abs(alpha - 1.0) < 1e-10:
            return base_drift - (2/np.pi) * gamma * beta * T
        term = gamma**alpha / np.cos(np.pi * alpha / 2)
        return base_drift - term * np.tan(np.pi * alpha / 2) * beta * T

def option_itm_probability(S0, K, T, r, q, alpha, beta, gamma, option_type='call'):
    mu = martingale_condition(alpha, beta, gamma, T, r, q)
    scale = gamma * (T ** (1/alpha))
    log_threshold = np.log(K / S0)
    
    if option_type == 'call':
        # Use survival function for better numerical stability
        return levy_stable.sf(log_threshold, alpha, beta, loc=mu, scale=scale)
    else:
        return levy_stable.cdf(log_threshold, alpha, beta, loc=mu, scale=scale)

def expected_itm_payoff(S0, K, T, r, q, alpha, beta, gamma, option_type):
    mu = martingale_condition(alpha, beta, gamma, T, r, q)
    scale = gamma * (T ** (1/alpha))
    log_threshold = np.log(K / S0)
    
    if option_type == 'call':
        def integrand(x):
            log_payoff = np.log(S0) + x - np.log(S0 * np.exp(x) - K + 1e-100)
            pdf_val = levy_stable.pdf(x, alpha, beta, loc=mu, scale=scale)
            return np.exp(log_payoff) * pdf_val if np.isfinite(np.exp(log_payoff)) else 0
        
        # Use adaptive integration with bounds to avoid extreme values
        expected_payoff, _ = integrate.quad(integrand, log_threshold, log_threshold + 10, 
                                          limit=1000, epsabs=1e-6, epsrel=1e-6)
        return max(0, expected_payoff)
        
    else:  # put
        # Puts don't have the overflow issue since x is bounded above
        def integrand(x):
            payoff = K - S0 * np.exp(x)
            return payoff * levy_stable.pdf(x, alpha, beta, loc=mu, scale=scale)
        
        expected_payoff, _ = integrate.quad(integrand, -np.inf, log_threshold, 
                                          limit=1000, epsabs=1e-6, epsrel=1e-6)
        return max(0, expected_payoff)

def expected_option_profit(S0, K, T, r, q, alpha, beta, gamma, option_type='call'):
    itm_prob = option_itm_probability(S0, K, T, r, q, alpha, beta, gamma, option_type)
    expected_payoff = expected_itm_payoff(S0, K, T, r, q, alpha, beta, gamma, option_type)
    
    return {
        'probability_itm': itm_prob,
        'expected_payoff': expected_payoff,
        'discounted_profit': expected_payoff * np.exp(-r * T)
    }

def analyze_option_profitability(S0, K, T, r, q, alpha, beta, gamma, option_type='call'):
    print(f"Analyzing {option_type.upper()} option:")
    print(f"S0: ${S0}, K: ${K}, T: {T} years")
    print(f"α: {alpha}, β: {beta}, γ: {gamma}")
    print(f"r: {r}, q: {q}")
    
    is_otm = (option_type == 'call' and S0 < K) or (option_type == 'put' and S0 > K)
    
    if is_otm:
        print(f"✓ Option is OUT-OF-THE-MONEY")
        
        try:
            results = expected_option_profit(S0, K, T, r, q, alpha, beta, gamma, option_type)
            
            print(f"\nResults:")
            print(f"Probability of becoming ITM: {results['probability_itm']:.4f} ({results['probability_itm']*100:.1f}%)")
            print(f"Expected payoff if ITM: ${results['expected_payoff']:.2f}")
            print(f"Discounted expected profit: ${results['discounted_profit']:.2f}")
            
            return results
            
        except Exception as e:
            print(f"Numerical computation failed: {e}")
            print("Try adjusting integration bounds or using different numerical methods")
            return None
    else:
        status = "at-the-money" if S0 == K else "in-the-money"
        print(f"Option is {status.upper().replace('-', ' ')}")
        return None

# Alternative: Use characteristic function methods for better stability
def expected_payoff_characteristic(S0, K, T, r, q, alpha, beta, gamma, option_type):
    """Alternative method using characteristic function inversion."""
    # This would be more numerically stable but more complex to implement
    # For now, we'll use the improved quadrature method
    return expected_itm_payoff(S0, K, T, r, q, alpha, beta, gamma, option_type)


def quantile_estimation(returns):
    
    x_05 = scoreatpercentile(returns, 5)
    x_25 = scoreatpercentile(returns, 25)
    x_50 = scoreatpercentile(returns, 50)  # median
    x_75 = scoreatpercentile(returns, 75)
    x_95 = scoreatpercentile(returns, 95)
    
    phi_1 = (x_95 - x_05) / (x_75 - x_25)
    phi_2 = (x_95 + x_05 - 2*x_50) / (x_95 - x_05)
    
    if phi_1 <= 2.4390:
      alpha_est = 0.10 + (2.439 - phi_1) / 0.8
    else:
      alpha_est = 2.0 - (phi_1 - 2.439) / 0.5
    
    alpha_est = np.clip(alpha_est, 1.1, 2.0)  # Constrain to reasonable range
    
    if alpha_est < 1.7:
        beta_est = 5.5 * (phi_2 + 0.25) * (alpha_est - 1.7) + phi_2
    else:
        beta_est = phi_2
    
    beta_est = np.clip(beta_est, -1.0, 1.0)  # Constrain to [-1, 1]
    
    if alpha_est < 1.7:
        gamma_est = (x_75 - x_25) / (1.654 + 0.288*alpha_est - 0.640*(alpha_est-1.7))
    else:
        gamma_est = (x_75 - x_25) / (1.654 + 0.288*alpha_est)
    
    delta_est = x_50 + gamma_est * beta_est * np.tan(np.pi * alpha_est / 2)
    
    return {
        'alpha': alpha_est,
        'beta': beta_est,
        'gamma': gamma_est,
        'delta': delta_est
    }



"""
    Parameters:
    S0: initial stock price
    K: strike price
    T: time to maturity
    r: risk-free rate
    q: dividend yield
    alpha: stability parameter (0 < α ≤ 2)
    beta: skewness parameter (-1 ≤ β ≤ 1)
    gamma: scale parameter (γ > 0)
    option_type: 'call' or 'put'
    method: 'quadrature' (numerical integration) or 'characteristic' (FFT-based)
    Use quantile for proper estimation
    Returns:
    probability: P(S_T > K) for call, P(S_T < K) for put
"""
    

'''
Use yf function for downloading the data and put the variables through the quantile function for final usage
Use the quantile function for proper alpha, beta and gamma variables
This error can be safely ignored 
'''
if __name__ == "__main__":
    # Your parameters
    # S0, K, T, r, q = 100, 110, 0.25, 0.05, 0.02
    S0, K, T, r, q = 4500, 4600, 0.25, 0.05, 0.015
    # alpha, beta, gamma = 1.7, 0.3, 0.2
    alpha, beta, gamma = 1.1, -0.2, 0.3
    
    print("CALL OPTION ANALYSIS:")
    results_call = analyze_option_profitability(S0, K, T, r, q, alpha, beta, gamma, 'call')
    
    print("\n" + "="*50)
    print("PUT OPTION ANALYSIS:")
    results_put = analyze_option_profitability(S0, 90, T, r, q, alpha, beta, gamma, 'put')
