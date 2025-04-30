"""
Crude Monte Carlo estimator for E[X^2] when X ~ Exp(1).
-------------------------------------------------------
Run from the command line:

    python cmc_exp_square.py
"""

import numpy as np


def crude_monte_carlo(n: int, seed: int, a: int) -> tuple[float, float]:
    """
    Parameters
    ----------
    n     : int
        Number of Monte-Carlo samples.
    seed  : int | None
        Random-number–generator seed (makes runs repeatable).

    Returns
    -------
    mu_hat : float
        Point estimate of E[X^2].
    se_hat : float
        Estimated standard error of mu_hat.
    """
    rng = np.random.default_rng(seed)

    C = np.array([
        [4, -1],
        [-1, 4]
    ])

    mean = np.array ([0, 0])
    # 1. SAMPLE .....................................  X_i ~ Exp(1)
    x = rng.multivariate_normal(mean=mean, cov=C, size=n)

    # 2. EVALUATE ..................................  g_i = f(X_i) = X_i^2
    # g = x**2
    g = x > a
    insideA = np.logical_and(g[:, 0], g[:, 1])
    Y = np.count_nonzero(insideA)

    # 3. AVERAGE ...................................  mu_hat = (1/n) Σ g_i
    mu_hat = insideA.mean()

    # -------- diagnostics ----------
    # Unbiased estimate of Var[g]  (ddof=1 --> sample var)
    var_hat = insideA.var(ddof=1)
    se_hat = np.sqrt(var_hat / n)   # SE( μ̂ ) = √(Var[g] / n )

    return mu_hat, se_hat

def show_results(a: int):
    n_draws = 5_000

    mu, se = crude_monte_carlo(n_draws, seed=42, a = a)
    
    ci_low  = mu - 1.96 * se
    ci_high = mu + 1.96 * se

    print(f"n         : {n_draws:,}")
    print(f"estimate  : {mu:.6f}")
    print(f"Std. error: {se:.6f}")
    print(f"95% CI    : [{ci_low:.6f}, {ci_high:.6f}]")
    print()


if __name__ == "__main__":
    n_draws = 5_000

    show_results(1)
    show_results(3)
    show_results(10)
    # mu, se = crude_monte_carlo(n_draws, seed=42, 1=)
    # mu, se = crude_monte_carlo(n_draws, seed=42, a = 3)
    # mu, se = crude_monte_carlo(n_draws, seed=42, a = 10)


