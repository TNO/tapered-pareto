import numpy as np


def moment_magnitude(seismic_moment):
    """
    Calculate the moment magnitude of an earthquake from its moment according to Hanks and Kanamori (1979).

    Parameters
    ----------
    moment : float
        The moment of the earthquake.

    Returns
    -------
    float

    References
    ----------
    .. [1] Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. Journal of Geophysical Research: Solid Earth, 84(B5), 2348-2350.
    """
    # moment = 10**(1.5 * moment_magnitude + 9.05)

    return (np.log10(seismic_moment) - 9.05) / 1.5


def seismic_moment(moment_magnitude):
    """
    Calculate the moment of an earthquake from its magnitude according to Hanks and Kanamori (1979).

    Parameters
    ----------
    moment_magnitude : float
        The moment magnitude of the earthquake.

    Returns
    -------
    float

    References
    ----------
    .. [1] Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. Journal of Geophysical Research: Solid Earth, 84(B5), 2348-2350.
    """

    return 10 ** (1.5 * moment_magnitude + 9.05)
