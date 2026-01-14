# Why Portfolio Allocations Were Uniform & How to Fix It

## The Problem

Your A2C agent was outputting nearly uniform allocations:

```
Before (7 tech stocks):
AAPL: 14.73% ≈ 1/7 = 14.29%
MSFT: 14.56%
GOOGL: 13.98%
...all ≈ 14%

After (17 diverse stocks):
AAPL: 6.05% ≈ 1/17 = 5.88%
MSFT: 5.56%
NVDA: 5.80%
...all ≈ 6%
```

**This means the agent hasn't learned to differentiate between assets!**

---

## Root Causes

### 1. Entropy Regularization

```python
# In your training loop:
entropy_loss = -entropy  # Maximize entropy
loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
                                                     ↑
                                          This pushes toward uniformity!
```

**Entropy** measures randomness/uniformity:
- Maximum entropy = uniform distribution = [1/N, 1/N, ..., 1/N]
- Minimum entropy = concentrated = [1.0, 0, 0, ..., 0]

**By maximizing entropy**, you're explicitly encouraging uniform allocations!

**Why we use entropy:**
- Prevents premature convergence
- Encourages exploration
- Avoids getting stuck in local optima

**But:** Too much entropy → permanent uniformity

### 2. Dirichlet Distribution Symmetry

```python
# Actor outputs:
concentrations = softplus(logits) + 1e-3
# Example: [5.2, 5.1, 5.3, 5.0, 5.2, 5.1, 5.3, ...]

# Dirichlet sampling:
weights ~ Dirichlet(concentrations)
# When α₁ ≈ α₂ ≈ ... ≈ αₙ → weights ≈ [1/n, 1/n, ..., 1/n]
```

**The Dirichlet distribution is symmetric**: if all concentrations are similar, it naturally outputs near-uniform weights.

**What happened:**
1. Network initialization → similar outputs
2. Entropy bonus → keeps them similar
3. Weak reward signal → no strong reason to diverge
4. Result → stuck at uniform

### 3. Weak Reward Differentiation

```python
# Your reward:
reward = portfolio_return - 0.1 * volatility

# Problem: With similar assets or short horizons,
# differences between allocations are tiny:
Allocation A: reward = 0.0215
Allocation B: reward = 0.0217  # Only 0.0002 difference!
```

**Gradient signal too weak** to overcome entropy bias toward uniformity.

### 4. Insufficient Training

With 17 stocks and ~900 trading days:
- State space dimension: (11, 17) = 187 features
- Action space: 17-dimensional simplex (infinite possibilities)
- Episodes: 300 × 900 steps = 270,000 steps

**For such complexity, 300 epochs might not be enough!**

---

## Solutions Applied

### Fix 1: Remove Entropy Bonus

```python
# OLD:
entropy_coef=0.01  # or 0.005

# NEW:
entropy_coef=0.0  # REMOVED COMPLETELY
```

**Effect:** Allow agent to learn concentrated positions without penalty.

**Trade-off:**
- ✅ Enables diverse allocations
- ❌ Might converge faster to local optima
- Solution: Use longer training to explore before converging

### Fix 2: Add Concentration Reward

```python
# Bonus for being different from uniform
uniform_weight = 1.0 / stock_dim  # e.g., 1/17 = 5.88%
weight_deviation = sum(|weights - uniform|)
concentration_reward = 0.05 * weight_deviation

# Total reward:
reward = portfolio_return - risk_penalty + concentration_reward
```

**Example:**
```
Allocation A (uniform): [5.88%, 5.88%, ..., 5.88%]
→ deviation = 0 → concentration_reward = 0

Allocation B (diverse): [15%, 10%, 8%, 5%, ..., 2%]
→ deviation = 0.45 → concentration_reward = 0.0225 ✅

Agent learns: "Being different is good!"
```

### Fix 3: Increase Training Duration

```python
# OLD:
epochs=300

# NEW:
epochs=500
```

**Effect:** More time to explore and find better allocations.

**With 17 stocks:**
- 500 epochs × ~900 steps = 450,000 gradient updates
- More chances to discover which stocks to favor

### Fix 4: Reduce Learning Rate

```python
# OLD:
lr=3e-2  # 0.03 - quite high!

# NEW:
lr=1e-3  # 0.001 - more stable
```

**Why:**
- High LR → big updates → oscillations → hard to converge
- Low LR → small updates → stable → better fine-tuning

**With 17 stocks, we need fine-grained adjustments!**

### Fix 5: Diversified Stock Universe

```python
# OLD: All tech stocks (high correlation)
['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']

# NEW: Multiple sectors (low correlation)
Technology: AAPL, MSFT, NVDA, GOOGL, META
Healthcare: JNJ, UNH, PFE
Finance: JPM, BAC, GS
Energy: XOM, CVX
Consumer: WMT, PG
Industrial: BA, CAT
```

**Correlation comparison:**
```
Tech-only portfolio:
Avg correlation = 0.75 (very high)
→ Agent can't differentiate

Diversified portfolio:
Tech-Tech: 0.70
Tech-Healthcare: 0.40
Tech-Energy: 0.20
Energy-Finance: -0.10 (negative! hedge!)
→ Agent can learn meaningful differences
```

---

## Expected Results After Fixes

### Before Fixes:
```
All stocks ≈ 5-6% (uniform)
No clear preferences
Sharpe ratio: ~1.0
```

### After Fixes (Expected):
```
Technology (high growth):
  NVDA: 12% ✅ (semiconductors booming)
  AAPL: 8%
  MSFT: 7%

Healthcare (defensive):
  UNH: 9% ✅ (stable
, low correlation with tech)
  JNJ: 6%

Energy (hedge):
  XOM: 10% ✅ (negative correlation, inflation hedge)
  CVX: 7%

Finance:
  JPM: 8%
  GS: 5%

Others:
  Lower allocations to boring/low-return stocks
  
Total: More varied (3-12% range vs 5-6%)
Sharpe ratio: ~1.3-1.5 (improvement!)
```

---

## Additional Techniques (If Still Not Working)

### Option 1: Initialization Bias

Add a bias to prefer certain sectors initially:

```python
class Actor(nn.Module):
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        self.net = nn.Sequential(...)
        
        # Initialize final layer with slight preferences
        # Example: favor first 5 stocks (tech)
        with torch.no_grad():
            self.net[-1].bias[:5] += 0.5  # Boost tech
            self.net[-1].bias[5:8] -= 0.2  # Reduce healthcare
```

### Option 2: Two-Stage Training

```python
# Stage 1: High entropy, explore
train_a2c(env, epochs=200, entropy_coef=0.01)

# Stage 2: Low entropy, exploit
train_a2c(env, epochs=300, entropy_coef=0.0, actor=stage1_actor)
```

### Option 3: Curriculum Learning

Start with fewer stocks, gradually add more:

```python
# Week 1: Train on 5 stocks
# Week 2: Add 5 more stocks, continue training
# Week 3: Add final 7 stocks
```

### Option 4: Stronger Reward Signal

Use Sharpe-optimized reward instead of mean-variance:

```python
# Instead of:
reward = return - 0.1 * var

# Use:
sharpe = (return - risk_free) / std
reward = return * (1 + sharpe)  # Amplify good allocations
```

---

## Debugging Checklist

After running with these fixes, check:

### 1. **Are concentrations diverse?**
```python
# Add to training loop:
if epoch % 50 == 0:
    sample_conc = concentrations.squeeze(0).detach().cpu().numpy()
    print(f"Concentrations: min={sample_conc.min():.2f}, "
          f"max={sample_conc.max():.2f}, "
          f"std={sample_conc.std():.2f}")
```

**Good:** std > 2.0, max/min ratio > 3
**Bad:** std < 0.5, all values similar

### 2. **Is the agent learning?**
```python
# Check if rewards are improving
print(f"First 50 episodes avg reward: {np.mean(rewards[:50])}")
print(f"Last 50 episodes avg reward: {np.mean(rewards[-50:])}")
```

**Good:** Last > First by >20%
**Bad:** No improvement or getting worse

### 3. **Are allocations changing over time?**
```python
# Look at actions_history from different episodes
early_actions = actions_history[:100]  # First 100 steps
late_actions = actions_history[-100:]  # Last 100 steps

early_entropy = -np.sum(np.mean(early_actions, axis=0) * 
                        np.log(np.mean(early_actions, axis=0) + 1e-8))
late_entropy = -np.sum(np.mean(late_actions, axis=0) * 
                       np.log(np.mean(late_actions, axis=0) + 1e-8))

print(f"Entropy early: {early_entropy:.2f}")
print(f"Entropy late: {late_entropy:.2f}")
```

**Good:** Late entropy < Early entropy (becoming more concentrated)
**Bad:** Both high and similar

---

## Summary of Changes

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| Stocks | 7 tech | 17 diversified | Enable differentiation |
| Entropy coef | 0.01 | **0.0** | Allow concentration |
| Epochs | 300 | **500** | More learning time |
| Learning rate | 3e-2 | **1e-3** | Stable convergence |
| Reward | return - 0.1*var | **+ concentration bonus** | Encourage diversity |

---

## What to Expect

**Training time:** ~10-20 minutes (500 epochs × 17 stocks)

**Expected final allocations:**
- **Range:** 3-15% (not 5-6%)
- **Top holdings:** 10-15% (clear favorites)
- **Bottom holdings:** 2-5% (clear dislikes)
- **Sharpe ratio:** Improvement of 15-30%

**If still uniform after this:**
1. The reward signal might be too weak for your data
2. Consider switching to Sharpe-based rewards
3. Try deterministic policy (softmax) instead of Dirichlet
4. Check if market data has enough variation (bull market = less differentiation)

---

## The Fundamental Issue

A2C + Dirichlet + Entropy regularization is **designed for exploration**, not exploitation. For portfolio optimization, you want:

**Less exploration, more exploitation!**

The fixes above shift the balance:
- ❌ Entropy → promotes exploration → uniformity
- ✅ Concentration bonus → promotes exploitation → diversity
- ✅ More epochs → time to find optimal
- ✅ Lower LR → precision in convergence

This should finally give you **meaningful, differentiated allocations**! 🎯
