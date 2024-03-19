import scipy.special as sp
import numpy as np


def sample_tapered_pareto(index_pareto, scale_pareto, scale_exponential, rng, **kwargs):
    """
    Sample from a tapered pareto distribution.

    Parameters
    ----------
    index_pareto : array_like
        Pareto index.
    scale_pareto : array_like
        Scale parameter of the pareto distribution.
    scale_exponential : array_like
        Scale parameter of the exponential distribution.
    rng : numpy.random.Generator, dask.random.Generator or similar
        Random number generator that supports the pareto and exponential distributions.
    **kwargs : dict
        Additional keyword arguments to pass to the random number generator.

    Returns
    -------
    ndarray
        A sample from the tapered pareto distribution.

    Notes
    -----
    This function provides a convenient way to simulate tapered pareto variables
    such that we can defer the random generation to off-the-shelf random number generators,
    e.g., from numpy dask.
    """
    # note that numpy pareto is really lomax, or pareto II, which starts at 0 so we have to add 1
    pareto_samples = scale_pareto * (1 + rng.pareto(index_pareto, **kwargs))

    exponential_samples = scale_pareto + rng.exponential(scale_exponential, **kwargs)

    return np.minimum(pareto_samples, exponential_samples)


def raw_moment(index_pareto, scale_pareto, scale_exponential, k):
    """
    Calculate the raw moments of the tapered Pareto distribution.

    Parameters
    ----------
    index_pareto : array_like
        Pareto index, often denoted as beta
    scale_pareto : array_like
        Scale parameter of the pareto distribution.
    scale_exponential : array_like
        Scale parameter of the exponential distribution.
    k : int
        The order of the statistical moment.

    Returns
    -------
    ndarray
        The raw moment.

    References
    ----------
    .. [1] Kagan, Y. Y., & Schoenberg, F. P. (2001). Estimation of the upper cutoff parameter for the tapered Pareto distribution. J. Appl. Probab. 38A, 158-175.
    """
    if k == 0:
        expectation = 1
    else:
        x = scale_pareto / scale_exponential
        s = k - index_pareto
        inc_upper_gamma = sp.gamma(s) * sp.gammaincc(s, x)

        expectation = scale_exponential**k * (
            (x**k) + k * (x**index_pareto) * np.exp(x) * inc_upper_gamma
        )

    return expectation


def central_moment(index_pareto, scale_pareto, scale_exponential, k):
    """
    Calculate the central moment of the tapered Pareto seismic moment distribution.

    Parameters
    ----------
    index_pareto : array_like
        Pareto index.
    scale_pareto : array_like
        Scale parameter of the pareto distribution.
    scale_exponential : array_like
        Scale parameter of the exponential distribution.
    k : int
        The order of the statistical moment.

    Returns
    -------
    ndarray
        The central moment.
    """
    raw_moment_1 = raw_moment(index_pareto, scale_pareto, scale_exponential, 1)
    sum_raw_moments = 0
    for i in range(k + 1):
        raw_moment_i = raw_moment(index_pareto, scale_pareto, scale_exponential, i)
        sum_raw_moments += sp.binom(k, i) * raw_moment_i * (-raw_moment_1) ** (k - i)

    return sum_raw_moments


def sf(x, index_pareto, scale_pareto, scale_exponential):
    """
    Calculate the survival function for a tapered Pareto distribution.

    Parameters
    ----------
    x : array_like
        Quantiles.
    index_pareto : array_like
        Pareto index.
    scale_pareto : array_like
        Scale parameter of the pareto distribution.
    scale_exponential : array_like
        Scale parameter of the exponential distribution.

    Returns
    -------
    ndarray
        Probability of exceedance
    """
    # two concurrent survival processes ... both have to be survived
    sf_pareto = (x / scale_pareto) ** (-index_pareto)
    sf_exponential = np.exp(-(x - scale_pareto) / scale_exponential)

    # really, both
    sf = sf_pareto * sf_exponential

    return np.where(x > scale_pareto, sf, 1)


def inverse_sf(p, index_pareto, scale_pareto, scale_exponential):
    """
    Calculate the inverse survival function of the tapered Pareto distribution distribution.

    Parameters
    ----------
    p : array_like
        Probability of exceedance.
    index_pareto : array_like
        Pareto index.
    scale_pareto : array_like
        Scale parameter of the pareto distribution.
    scale_exponential : float
        Scale parameter of the exponential distribution.

    Returns
    -------
    ndarray
        Quantiles corresponding to the probability of exceedance.

    Notes
    -----
    The inverse survival function was obtained with the use of Wolfram Alpha [1]_

    References
    ----------
    .. [1] Wolfram Alpha (2023): https://www.wolframalpha.com/input?i=solve+p%3D%28%28t%2Fm%29**%28index_pareto%29%29*exp%28%28t-m%29%2Fc%29+for+m, last accessed 2023-07-14.
    """
    alpha = (scale_pareto / scale_exponential) / index_pareto
    realization = (
        index_pareto
        * scale_exponential
        * sp.lambertw(p ** (-1 / index_pareto) * alpha * np.exp(alpha))
    )

    return np.real(realization)


def pdf(x, index_pareto, scale_pareto, scale_exponential):
    """
    Calculate the probability density function for a tapered Pareto distribution.

    Parameters
    ----------
    x : array_like
        Quantiles.
    index_pareto : array_like
        Pareto index.
    scale_pareto : array_like
        Scale parameter of the pareto distribution.
    scale_exponential : array_like
        Scale parameter of the exponential distribution.

    Returns
    -------
    ndarray
        Probability density.

    Notes
    -----
    The probability density function is calculated as the negative derivative of the survival function.
    Given that:
    .. math:: S_{tp}(m) = S_p(m) * S_e(m)
    we get:
    .. math:: f_{tp}(m) = - \frac{dS_{tp}(m)}{dm}
    .. math:: f_{tp}(m) = - \frac{d(S_p(m) * S_e(m))}\frac{dm}
    .. math:: f_{tp}(m) = f_p(m) * S_e(m) + S_p(m) * f_e(m)
    .. math:: f_{tp}(m) = (\frac{f_p(m)}{S_p(m)} + \frac{f_e(m)}{S_e(m)}) * S_{tp}(m)
    .. math:: f_{tp}(m) = (frac{index\_pareto}{m} + \frac{1}{scale\_exponential} * S_{tp}(m)

    """

    pdf = ((index_pareto / x) + (1 / scale_exponential)) * sf(
        x, index_pareto, scale_pareto, scale_exponential
    )

    return np.where(x > scale_pareto, pdf, 0)
