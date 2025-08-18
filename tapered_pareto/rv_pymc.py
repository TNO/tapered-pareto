import numpy as np

import pytensor.tensor as pt
from pytensor.tensor.random.op import RandomVariable
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none
from typing import Tuple

from .rv_scipy import tapered_pareto

# following instructions from
# https://www.pymc.io/projects/docs/en/stable/contributing/implementing_distribution.html


class TaperedParetoRV(RandomVariable):
    name: str = "tapered_pareto"

    signature: str = "(),(),()->()"
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("TaperedPareto", "\\operatorname{TaperedPareto}")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        index_pareto: np.ndarray,
        scale_pareto: np.ndarray,
        scale_exponential: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        return tapered_pareto.rvs(
            index_pareto, scale_pareto, scale_exponential, random_state=rng, size=size
        )


# Create the actual `RandomVariable` `Op`...
tapered_pareto_rv = TaperedParetoRV()


class TaperedPareto(PositiveContinuous):
    rv_op = tapered_pareto_rv

    # dist() is responsible for returning an instance of the rv_op.
    # We pass the standard parametrizations to super().dist
    @classmethod
    def dist(cls, index_pareto, scale_pareto, scale_exponential, **kwargs):
        index_pareto = pt.as_tensor_variable(index_pareto)
        scale_pareto = pt.as_tensor_variable(scale_pareto)
        scale_exponential = pt.as_tensor_variable(scale_exponential)

        # The first value-only argument should be a list of the parameters that
        # the rv_op needs in order to be instantiated
        return super().dist([index_pareto, scale_pareto, scale_exponential], **kwargs)

    # moment returns a symbolic expression for the stable moment from which to start sampling
    # the variable, given the implicit `rv`, `size` and `param1` ... `paramN`.
    # This is typically a "representative" point such as the the mean or mode.
    # here, we use the first order moment from Kagan and Schoenberg (2001)
    def moment(rv, size, index_pareto, scale_pareto, scale_exponential):
        n = 1  # order of the moment
        r = scale_pareto / scale_exponential
        gamma = pt.math.gamma(n - index_pareto)
        gammaincc = gamma * pt.math.gammaincc(n - index_pareto, r)
        moment = scale_exponential**n * (
            r**n + n * (r**index_pareto) * np.exp(r) * gammaincc
        )

        if not rv_size_is_none(size):
            moment = pt.full(size, moment)
        return moment

    # Logp returns a symbolic expression for the elementwise log-pdf or log-pmf evaluation
    # of the variable given the `value` of the variable and the parameters `param1` ... `paramN`.
    def logp(value, index_pareto, scale_pareto, scale_exponential):
        sf_pareto = (value / scale_pareto) ** (-index_pareto)
        sf_exponential = pt.exp(-(value - scale_pareto) / scale_exponential)
        sf = sf_pareto * sf_exponential
        pdf = ((index_pareto / value) + (1 / scale_exponential)) * sf
        logp_expression = pt.log(pdf)

        # A switch is often used to enforce the distribution support domain
        bounded_logp_expression = pt.switch(
            value - scale_pareto >= 0,
            logp_expression,
            -np.inf,
        )

        # We use `check_parameters` for parameter validation. After the default expression,
        # multiple comma-separated symbolic conditions can be added.
        # Whenever a bound is invalidated, the returned expression raises an error
        # with the message defined in the optional `msg` keyword argument.
        return check_parameters(
            bounded_logp_expression,
            index_pareto > 0,
            scale_pareto > 0,
            scale_exponential > 0,
            msg="index_pareto > 0, scale_pareto > 0, scale_exponential > 0",
        )
