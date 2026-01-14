"""
Quick diagnostic script to check if allocations are becoming more diverse
Run this after training to analyze the results
"""

import numpy as np

def analyze_allocations(final_weights, ticker_list):
    """
    Analyze portfolio allocations to check if they're meaningfully diverse
    
    Parameters:
    -----------
    final_weights: array of portfolio weights
    ticker_list: list of ticker symbols
    """
    n_assets = len(final_weights)
    uniform = 1.0 / n_assets
    
    print("=" * 60)
    print("ALLOCATION ANALYSIS")
    print("=" * 60)
    
    # 1. Range analysis
    print(f"\n1. Range Analysis:")
    print(f"   Min allocation: {final_weights.min()*100:.2f}%")
    print(f"   Max allocation: {final_weights.max()*100:.2f}%")
    print(f"   Range: {(final_weights.max() - final_weights.min())*100:.2f}%")
    print(f"   Uniform would be: {uniform*100:.2f}%")
    
    # 2. Diversity score
    deviation_from_uniform = np.sum(np.abs(final_weights - uniform))
    diversity_score = deviation_from_uniform / (2 * (n_assets - 1) / n_assets) * 100
    
    print(f"\n2. Diversity Score: {diversity_score:.1f}%")
    print(f"   0% = Perfectly uniform (bad)")
    print(f"   100% = Maximally concentrated (one asset 100%)")
    print(f"   Target: >30% for good diversification")
    
    # 3. Entropy analysis
    entropy = -np.sum(final_weights * np.log(final_weights + 1e-10))
    max_entropy = np.log(n_assets)  # Uniform distribution
    relative_entropy = (1 - entropy / max_entropy) * 100
    
    print(f"\n3. Concentration (Relative Entropy): {relative_entropy:.1f}%")
    print(f"   0% = Uniform distribution")
    print(f"   100% = Single asset")
    print(f"   Target: >20% for meaningful concentration")
    
    # 4. Top holdings
    sorted_idx = np.argsort(final_weights)[::-1]
    print(f"\n4. Top 5 Holdings:")
    for i in range(min(5, n_assets)):
        idx = sorted_idx[i]
        print(f"   {ticker_list[idx]}: {final_weights[idx]*100:.2f}%")
    
    # 5. Bottom holdings
    print(f"\n5. Bottom 5 Holdings:")
    for i in range(max(0, n_assets-5), n_assets):
        idx = sorted_idx[i]
        print(f"   {ticker_list[idx]}: {final_weights[idx]*100:.2f}%")
    
    # 6. Verdict
    print(f"\n6. Verdict:")
    if diversity_score < 10:
        print("   ❌ UNIFORM - Agent hasn't learned to differentiate")
        print("   → Try: Remove entropy, add concentration bonus, train longer")
    elif diversity_score < 30:
        print("   ⚠️  SLIGHTLY DIVERSE - Some learning but not enough")
        print("   → Try: Increase training epochs, reduce learning rate")
    elif diversity_score < 60:
        print("   ✅ GOOD DIVERSITY - Agent has learned meaningful allocations")
    else:
        print("   ⚡ HIGH CONCENTRATION - Agent strongly prefers certain assets")
        print("   → Check if over-concentrated (might be risky)")
    
    print("=" * 60)
    
    return {
        'diversity_score': diversity_score,
        'relative_entropy': relative_entropy,
        'range': final_weights.max() - final_weights.min(),
        'top_5': sorted_idx[:5],
        'bottom_5': sorted_idx[-5:]
    }


# Example usage:
if __name__ == "__main__":
    # Example 1: Nearly uniform (BAD)
    print("Example 1: Uniform allocation (what you're getting now)")
    uniform_weights = np.array([0.0588] * 17)
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'JNJ', 'UNH', 
               'PFE', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'WMT', 'PG', 'BA', 'CAT']
    analyze_allocations(uniform_weights, tickers)
    
    print("\n\n")
    
    # Example 2: Diverse (GOOD)
    print("Example 2: Diverse allocation (what you WANT)")
    diverse_weights = np.array([
        0.15, 0.12, 0.10, 0.08, 0.09,  # Tech: concentrated
        0.07, 0.08, 0.04,               # Healthcare
        0.06, 0.04, 0.05,               # Finance
        0.03, 0.02,                     # Energy: underweight
        0.04, 0.02,                     # Consumer
        0.01, 0.00                      # Industrial: minimal
    ])
    analyze_allocations(diverse_weights, tickers)
