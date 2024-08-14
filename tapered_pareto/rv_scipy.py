import scipy.stats as st
import scipy.special as sp
import numpy as np

# from mpmath import mp


class tapered_pareto_gen(st.rv_continuous):
    "exponentially tapered pareto distribution"

    def _sf(self, x, index_pareto, scale_pareto, scale_exponential):
        pareto_sf = st.pareto.sf(x, index_pareto, loc=0, scale=scale_pareto)
        expon_sf = st.expon.sf(x, loc=scale_pareto, scale=scale_exponential)
        return pareto_sf * expon_sf

    def _pdf(self, x, index_pareto, scale_pareto, scale_exponential):
        sf = self._sf(x, index_pareto, scale_pareto, scale_exponential)
        factor = (index_pareto / x) + (1 / scale_exponential)
        return factor * sf

    def _cdf(self, x, index_pareto, scale_pareto, scale_exponential):
        cdf_pareto = st.pareto.cdf(x, index_pareto, loc=0, scale=scale_pareto)
        cdf_expon = st.expon.cdf(x, loc=scale_pareto, scale=scale_exponential)
        cdf = cdf_pareto + cdf_expon - cdf_pareto * cdf_expon
        return cdf

    def _isf(self, q, index_pareto, scale_pareto, scale_exponential):
        alpha = (scale_pareto / scale_exponential) / index_pareto
        p = q
        x = (
            index_pareto
            * scale_exponential
            * sp.lambertw(p ** (-1 / index_pareto) * alpha * np.exp(alpha))
        )
        return np.real(x)

    def _ppf(self, q, index_pareto, scale_pareto, scale_exponential):
        alpha = (scale_pareto / scale_exponential) / index_pareto
        p = 1 - q
        x = (
            index_pareto
            * scale_exponential
            * sp.lambertw(p ** (-1 / index_pareto) * alpha * np.exp(alpha))
        )
        return np.real(x)

    def _munp(self, n, index_pareto, scale_pareto, scale_exponential):
        # from Kagan and Schoenberg (2001)
        if n == 0:
            expectation = 1
        else:
            r = scale_pareto / scale_exponential
            gamma = sp.gamma(n - index_pareto)
            gammaincc = gamma * sp.gammaincc(n - index_pareto, r)
            # unfortunately the scipy implementation does not allow (n - index_pareto)
            # equal or smaller than 0; the mpmath library can be used to compute the
            # incomplete gamma function for negative input, but does not allow for
            # vectorized application
            # inc_upper_gamma = mp.gammainc(n - index_pareto, alpha)
            expectation = scale_exponential**n * (
                r**n + n * (r**index_pareto) * np.exp(r) * gammaincc
            )
            return expectation

    def _rvs(
        self,
        index_pareto,
        scale_pareto,
        scale_exponential,
        size=None,
        random_state=None,
    ):
        rs = random_state
        pareto_rvs = st.pareto.rvs(
            index_pareto, loc=0, scale=scale_pareto, size=size, random_state=rs
        )
        expon_rvs = st.expon.rvs(
            loc=scale_pareto, scale=scale_exponential, size=size, random_state=rs
        )

        return np.minimum(pareto_rvs, expon_rvs)

    def _get_support(self, index_pareto, scale_pareto, scale_exponential):
        return st.pareto.support(index_pareto, loc=0, scale=scale_pareto)

    def _argcheck(self, index_pareto, scale_pareto, scale_exponential):
        check_index_pareto = index_pareto > 0
        check_scale_pareto = scale_pareto > 0
        check_scale_exponential = scale_exponential > 0
        return check_index_pareto & check_scale_pareto & check_scale_exponential


tapered_pareto = tapered_pareto_gen(name="tapered_pareto")
