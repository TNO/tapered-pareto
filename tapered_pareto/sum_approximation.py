import numpy as np
import scipy.stats as st
import scipy.special as sp
from .rv_scipy import tapered_pareto


def normal_quantile(q, n, beta, M_T, M_C):
    """
    Calculate the normal approximation of the quantiles of the sum of tapered Pareto seismic moment distribution.

    Parameters
    ----------
    q : float
        Quantile.
    n : int
        Number of earthquakes.
    beta : float
        Pareto index.
    M_T : float
        The lower truncation seismic moment (e.g., moment of completeness).
    M_C : float
        Corner seismic moment.

    Returns
    -------
    float
        The normal approximation quantile.

    """

    z_q = st.norm.ppf(q)
    tp = tapered_pareto(beta, M_T, M_C)
    mu = tp.mean()
    var = tp.var()
    approx = n * (mu + z_q * np.sqrt(var / n))

    return approx


def stable_quantile(q, n, beta, M_T):
    """
    Calculate the stable distribution approximation of the quantiles of the sum of tapered Pareto seismic moment distribution.

    Parameters
    ----------
    q : float
        Quantile.
    n : int
        Number of earthquakes.
    beta : float
        Pareto index.
    M_T : float
        The lower truncation seismic moment (e.g., moment of completeness).
    M_C : float
        Corner seismic moment.

    Returns
    -------
    float
        The stable distribution approximation quantile.

    References
    ----------
    .. [1] Zaliapin, Kagan and Schoenberg (2005). Approximating the distribution of Pareto sums. Pure appl. geophys. 162, 1187-1228.

    """
    skewness_index = 1
    index = beta  # note the possible confusion of parameter names beta->alpha
    s_qb = st.levy_stable.ppf(q, alpha=index, beta=skewness_index)

    # eq. 24 in Zaliapin et al. Note that b_n = 0 according to eq. 21.
    approx = (
        M_T
        * s_qb
        * (n * sp.gamma(1 - index) * np.cos(np.pi * index / 2)) ** (1 / index)
    )

    return approx


def quantile_approximator(q, count, beta, M_T, M_C):
    """
    Calculate approximation of the quantiles of the sum of tapered Pareto distributed variables.
    This approximation is symmetric around the expectation value in log-space.

    Parameters
    ----------
    q : float or array_like of float
        Quantile nodes.
    count : int
        Number of earthquakes.
    beta : float
        Pareto index.
    M_T : float
        The lower truncation seismic moment (e.g., moment of completeness).
    M_C : float
        Corner seismic moment.

    Returns
    -------
    float
        Quantile approximations.
    """
    # symmetric approximator
    expect = tapered_pareto(beta, M_T, M_C).moment(1)
    norm = normal_quantile(q, count, beta, M_T, M_C)
    stab = stable_quantile(q, count, beta, M_T)
    inv_norm = normal_quantile(1 - q, count, beta, M_T, M_C)
    reciprocal = (count * expect) ** 2 / inv_norm
    return np.where(
        q > 0.5,
        np.minimum(norm, stab),
        np.minimum(reciprocal, stab),
    )


def quantile_approximator_zaliapin(q, count, beta, M_T, M_C, c=2):
    """
    Calculate approximation of the quantiles of the sum of tapered Pareto distributed variables.
    This approximation is the form originally propose by Ilya, but not finalized.

    Parameters
    ----------
    q : float or array_like of float
        Quantile nodes.
    count : int
        Number of earthquakes.
    beta : float
        Pareto index.
    M_T : float
        The lower truncation seismic moment (e.g., moment of completeness).
    M_C : float
        Corner seismic moment.
    c : float
        Indicator value available to tune the approximation.

    Returns
    -------
    float
        Quantile approximations.
    """
    q_indicator = 0.5
    c_indicator = c
    expect = tapered_pareto(beta, M_T, M_C).moment(1)
    norm = normal_quantile(q, count, beta, M_T, M_C)
    stab = stable_quantile(q, count, beta, M_T)
    inv_norm = normal_quantile(1 - q, count, beta, M_T, M_C)
    reciprocal = (count * expect) ** 2 / inv_norm
    norm_indicator = normal_quantile(q_indicator, count, beta, M_T, M_C)
    stab_indicator = stable_quantile(q_indicator, count, beta, M_T, M_C)
    return np.where(
        q > 0.5,
        np.minimum(norm, stab),
        np.where(
            c_indicator * norm_indicator > stab_indicator,
            stab,
            reciprocal,
        ),
    )
