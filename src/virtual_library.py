from typing import Protocol
from jaxtyping import Float, Array
import jax.numpy as jnp

################################################################################
# Jax implementations of test functions from Virtual Library
# https://www.sfu.ca/~ssurjano/optimization.html
# All functions assume inputs are normalized to the unit hypercube [0, 1]^d
# All functions have been shifted so that the global minimum is (approx) 0.0
#################################################################################


class TestFunction(Protocol):
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]: ...


###################################################################################
# region Many Local Minima


class Ackley(TestFunction):
    """
    The Ackley function is characterized by a nearly flat outer region, and a large hole at the centre.
    See https://www.sfu.ca/~ssurjano/ackley.html for original implementation and more details.
    """

    def __init__(self, a: int = 20, b: float = 0.2, c: float = 2 * jnp.pi):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-32.768, 32.768]
        x = 65.536 * x - 32.768

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/ackleyr.html
        mean1 = jnp.mean(x**2, axis=-1)
        mean2 = jnp.mean(jnp.cos(self.c * x), axis=-1)

        term1 = -self.a * jnp.exp(-self.b * jnp.sqrt(mean1))
        term2 = -jnp.exp(mean2)

        y = term1 + term2 + self.a + jnp.exp(1)
        return y


class Bukin6(TestFunction):
    """
    The sixth Bukin function has many local minima, all of which lie in a ridge.
    See https://www.sfu.ca/~ssurjano/bukin6.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Bukin6 is only defined for 2D inputs"

        # denormalize to x1 in [-15, -5], x2 in [-3, 3]
        x1 = 10 * x[..., 0] - 15
        x2 = 6 * x[..., 1] - 3

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/bukin6r.html
        term1 = 100 * jnp.sqrt(jnp.abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * jnp.abs(x1 + 10)
        y = term1 + term2
        return y


class CrossInTray(TestFunction):
    """
    The Cross-in-Tray function has multiple global minima, in the characteristic "cross" pattern.
    See https://www.sfu.ca/~ssurjano/crossit.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "CrossInTray is only defined for 2D inputs"

        # denormalize to [-10, 10]
        x1 = 20 * x[..., 0] - 10
        x2 = 20 * x[..., 1] - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/crossitr.html
        fact1 = jnp.sin(x1) * jnp.sin(x2)
        fact2 = jnp.exp(jnp.abs(100 - jnp.sqrt(x1**2 + x2**2) / jnp.pi))
        y = -0.0001 * (jnp.abs(fact1 * fact2) + 1) ** 0.1

        # shift so that global minimum is 0.0
        y = y + 2.06261
        return y


class DropWave(TestFunction):
    """
    The Drop-Wave function is multimodal and highly complex.
    See https://www.sfu.ca/~ssurjano/drop.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "DropWave is only defined for 2D inputs"

        # denormalize to [-5.12, 5.12]
        x1 = 10.24 * x[..., 0] - 5.12
        x2 = 10.24 * x[..., 1] - 5.12

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/dropr.html
        frac1 = 1 + jnp.cos(12 * jnp.sqrt(x1**2 + x2**2))
        frac2 = 0.5 * (x1**2 + x2**2) + 2
        y = -frac1 / frac2

        # shift so that global minimum is 0.0
        y = y + 1.0
        return y


class EggHolder(TestFunction):
    """
    The Eggholder function is a difficult function to optimize, because of the large number of local minima.
    See https://www.sfu.ca/~ssurjano/egg.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "EggHolder is only defined for 2D inputs"

        # denormalize to [-512, 512]
        x1 = 1024 * x[..., 0] - 512
        x2 = 1024 * x[..., 1] - 512

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/eggr.html
        term1 = -(x2 + 47) * jnp.sin(jnp.sqrt(jnp.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * jnp.sin(jnp.sqrt(jnp.abs(x1 - (x2 + 47))))
        y = term1 + term2

        # shift so that global minimum is 0.0
        y = y + 959.6407
        return y


class GramacyLee(TestFunction):
    """
    The Gramacy-Lee function is a 1D function with many local minima.
    See https://www.sfu.ca/~ssurjano/grlee12.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 1"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 1, "GramacyLee is only defined for 1D inputs"

        # denormalize to [0.5, 2.5]
        x = 2 * x[..., 0] + 0.5

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/grlee12r.html
        term1 = jnp.sin(10 * jnp.pi * x) / (2 * x)
        term2 = (x - 1) ** 4
        y = term1 + term2

        # shift so that global minimum is 0.0
        y = y + 0.869011135
        return y


class Griewank(TestFunction):
    """
    The Griewank function has many widespread local minima, which are regularly distributed.
    See https://www.sfu.ca/~ssurjano/griewank.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-600, 600]
        x = 1200 * x - 600

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/griewankr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        sum_term = jnp.sum(x**2 / 4000, axis=-1)
        prod_term = jnp.prod(jnp.cos(x / jnp.sqrt(ii)), axis=-1)
        y = sum_term - prod_term + 1
        return y


class HolderTable(TestFunction):
    """
    The Holder Table function has many local minima, with four global minima.
    See https://www.sfu.ca/~ssurjano/holder.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "HolderTable is only defined for 2D inputs"

        # denormalize to [-10, 10]
        x1 = 20 * x[..., 0] - 10
        x2 = 20 * x[..., 1] - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/holderr.html
        fact1 = jnp.sin(x1) * jnp.cos(x2)
        fact2 = jnp.exp(jnp.abs(1 - jnp.sqrt(x1**2 + x2**2) / jnp.pi))
        y = -jnp.abs(fact1 * fact2)

        # shift so that global minimum is 0.0
        y = y + 19.2085
        return y


class Langermann(TestFunction):
    """
    The Langermann function is multimodal, with many unevenly distributed local minima.
    See https://www.sfu.ca/~ssurjano/langer.html for original implementation and more details.
    """

    A: Float[Array, "d m"] = jnp.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]).T
    c: Float[Array, "m"] = jnp.array([1, 2, 5, 2, 3])

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [0, 10]
        x = 10 * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/langerr.html
        inner = jnp.sum((x[..., None] - self.A) ** 2, axis=-2)
        term1 = jnp.exp(-inner / jnp.pi)
        term2 = jnp.cos(jnp.pi * inner)
        y = jnp.sum(self.c * term1 * term2, axis=-1)

        # shift so that global minimum is 0.0
        y = y + 4.1558
        return y


class Levy(TestFunction):
    """
    The Levy function has many local minima, with a regular distribution.
    See https://www.sfu.ca/~ssurjano/levy.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-10, 10]
        x = 20 * x - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/levyr.html
        w = 1 + (x - 1) / 4
        w1, wi, wd = w[..., 0], w[..., 1:-1], w[..., -1]
        term1 = jnp.sin(jnp.pi * w1) ** 2
        term2 = (wi - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * wi + 1.0) ** 2)
        term3 = (wd - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * wd) ** 2)
        y = term1 + term2.sum(axis=-1) + term3
        return y


class Levy13(TestFunction):
    """
    The Levy N.13 function is a 2D function with many local minima.
    See https://www.sfu.ca/~ssurjano/levy13.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Levy13 is only defined for 2D inputs"

        # denormalize to [-10, 10]
        x1 = 20 * x[..., 0] - 10
        x2 = 20 * x[..., 1] - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/levy13r.html
        term1 = jnp.sin(3 * jnp.pi * x1) ** 2
        term2 = (x1 - 1) ** 2 * (1 + jnp.sin(3 * jnp.pi * x2) ** 2)
        term3 = (x2 - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * x2) ** 2)
        y = term1 + term2 + term3
        return y


class Rastrigin(TestFunction):
    """
    The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima are regularly distributed.
    See https://www.sfu.ca/~ssurjano/rastr.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-5.12, 5.12]
        x = 10.24 * x - 5.12

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rastrr.html
        terms = x**2 - 10 * jnp.cos(2 * jnp.pi * x)
        y = jnp.sum(terms, axis=-1)

        # shift so that global minimum is 0.0
        y = y + 10 * x.shape[-1]
        return y


class Schaffer2(TestFunction):
    """
    The Schaffer N.2 function is a 2D function with many local minima.
    See https://www.sfu.ca/~ssurjano/schaffer2.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Schaffer2 is only defined for 2D inputs"

        # denormalize to [-100, 100]
        x1 = 200 * x[..., 0] - 100
        x2 = 200 * x[..., 1] - 100

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/schaffer2r.html
        fact1 = jnp.sin(x1**2 - x2**2) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        y = 0.5 + fact1 / fact2
        return y


class Schaffer4(TestFunction):
    """
    The Schaffer N.4 function is a 2D function with many local minima.
    See https://www.sfu.ca/~ssurjano/schaffer4.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Schaffer4 is only defined for 2D inputs"

        # denormalize to [-100, 100]
        x1 = 200 * x[..., 0] - 100
        x2 = 200 * x[..., 1] - 100

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/schaffer4r.html
        fact1 = jnp.cos(jnp.sin(jnp.abs(x1**2 - x2**2))) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        y = 0.5 + fact1 / fact2

        # shift so that global minimum is 0.0
        y = y - 0.29257873
        return y


class Schwefel(TestFunction):
    """
    The Schwefel function is complex, with many local minima.
    See https://www.sfu.ca/~ssurjano/schwef.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-500, 500]
        x = 1000 * x - 500

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/schwefr.html
        terms = x * jnp.sin(jnp.sqrt(jnp.abs(x)))
        y = -jnp.sum(terms, axis=-1)

        # shift so that global minimum is 0.0
        return y + 418.9829 * x.shape[-1]


class Shubert(TestFunction):
    """
    The Shubert function has several local minima and many global minima.
    See https://www.sfu.ca/~ssurjano/shubert.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Shubert is only defined for 2D inputs"

        # denormalize to [-5.12, 5.12]
        x1 = 10.24 * x[..., 0] - 5.12
        x2 = 10.24 * x[..., 1] - 5.12

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/shubertr.html
        ii = jnp.arange(1, 6)
        term1 = ii * jnp.cos((ii + 1) * x1[..., None] + ii)
        term2 = ii * jnp.cos((ii + 1) * x2[..., None] + ii)
        y = jnp.sum(term1, axis=-1) * jnp.sum(term2, axis=-1)

        # shift so that global minimum is 0.0
        y = y + 186.7309
        return y


# endregion
#################################################################################

#################################################################################
# region Bowl-Shaped


class Bohachevsky1(TestFunction):
    """
    The Bohachevsky functions all have the same similar bowl shape.
    See https://www.sfu.ca/~ssurjano/boha.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Bohachevsky1 is only defined for 2D inputs"

        # denormalize to [-100, 100]
        x1 = 200 * x[..., 0] - 100
        x2 = 200 * x[..., 1] - 100

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boha1r.html
        term1 = x1**2
        term2 = 2 * x2**2
        term3 = -0.3 * jnp.cos(3 * jnp.pi * x1)
        term4 = -0.4 * jnp.cos(4 * jnp.pi * x2)
        y = term1 + term2 + term3 + term4

        # shift so that global minimum is 0.0
        y = y + 0.7
        return y


class Bohachevsky2(TestFunction):
    """
    The Bohachevsky functions all have the same similar bowl shape.
    See https://www.sfu.ca/~ssurjano/boha.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Bohachevsky2 is only defined for 2D inputs"

        # denormalize to [-100, 100]
        x1 = 200 * x[..., 0] - 100
        x2 = 200 * x[..., 1] - 100

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boha2r.html
        term1 = x1**2
        term2 = 2 * x2**2
        term3 = -0.3 * jnp.cos(3 * jnp.pi * x1) * jnp.cos(4 * jnp.pi * x2)
        y = term1 + term2 + term3

        # shift so that global minimum is 0.0
        y = y + 0.3
        return y


class Bohachevsky3(TestFunction):
    """
    The Bohachevsky functions all have the same similar bowl shape.
    See https://www.sfu.ca/~ssurjano/boha.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Bohachevsky3 is only defined for 2D inputs"

        # denormalize to [-100, 100]
        x1 = 200 * x[..., 0] - 100
        x2 = 200 * x[..., 1] - 100

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boha3r.html
        term1 = x1**2
        term2 = 2 * x2**2
        term3 = -0.3 * jnp.cos(3 * jnp.pi * x1 + 4 * jnp.pi * x2)
        y = term1 + term2 + term3

        # shift so that global minimum is 0.0
        y = y + 0.3
        return y


class Perm0(TestFunction):
    """
    The Perm 0 function has a single global minimum.
    See https://www.sfu.ca/~ssurjano/perm0db.html for original implementation and more details.
    """

    def __init__(self, beta: float = 10.0):
        self.beta = beta

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-d, d]
        d = x.shape[-1]
        x = 2 * d * x - d

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/perm0dbr.html
        ii = jnp.arange(1, d + 1)
        xi = x[..., None, :] ** ii[..., None]  # array with x_j**i, shaped (..., i, j)
        inner = (ii + self.beta) * (xi - 1 / ii)
        outer = jnp.sum(jnp.sum(inner, axis=-1) ** 2, axis=-1)
        y = outer
        return y


class RotatedHyperEllipsoid(TestFunction):
    """
    The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal.
    It is an extension of the Axis Parallel Hyper-Ellipsoid function, also referred to as the Sum Squares function.
    See https://www.sfu.ca/~ssurjano/rothyp.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-65.536, 65.536]
        x = 131.072 * x - 65.536

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rothypr.html
        inner = jnp.cumsum(x**2, axis=-1)
        y = jnp.sum(inner, axis=-1)
        return y


class Sphere(TestFunction):
    """
    The Sphere function has d local minima except for the global one. It is continuous, convex and unimodal.
    See https://www.sfu.ca/~ssurjano/spheref.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-5.12, 5.12]
        x = 10.24 * x - 5.12

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/spherefr.html
        y = jnp.sum(x**2, axis=-1)
        return y


class SumPowers(TestFunction):
    """
    The Sum of Different Powers function is unimodal.
    See https://www.sfu.ca/~ssurjano/sumpow.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-1, 1]
        x = 2 * x - 1

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/sumpowr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        y = jnp.sum(jnp.abs(x) ** (ii + 1), axis=-1)
        return y


class SumSquares(TestFunction):
    """
    The Sum Squares function, also referred to as the Axis Parallel Hyper-Ellipsoid function, has no local minimum except the global one.
    It is continuous, convex and unimodal.
    See https://www.sfu.ca/~ssurjano/sumsqu.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-5.12, 5.12]
        x = 10.24 * x - 5.12

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/sumsqur.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        y = jnp.sum(ii * x**2, axis=-1)
        return y


class Trid(TestFunction):
    """
    The Trid function has no local minimum except the global one.
    See https://www.sfu.ca/~ssurjano/trid.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-d**2, d**2]
        d = x.shape[-1]
        x = 2 * d**2 * x - d**2

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/tridr.html
        sum1 = jnp.sum((x - 1) ** 2, axis=-1)
        sum2 = jnp.sum(x[..., 1:] * x[..., :-1], axis=-1)
        y = sum1 - sum2

        # shift so that global minimum is 0.0
        y = y + d * (d + 4) * (d - 1) / 6
        return y


# endregion
################################################################################

#################################################################################
# region Plate-Shaped


class Booth(TestFunction):
    """
    The Booth function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/booth.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Booth is only defined for 2D inputs"

        # denormalize to [-10, 10]
        x1 = 20 * x[..., 0] - 10
        x2 = 20 * x[..., 1] - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boothr.html
        term1 = (x1 + 2 * x2 - 7) ** 2
        term2 = (2 * x1 + x2 - 5) ** 2
        y = term1 + term2
        return y


class Matyas(TestFunction):
    """
    The Matyas function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/matya.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Matyas is only defined for 2D inputs"

        # denormalize to [-10, 10]
        x1 = 20 * x[..., 0] - 10
        x2 = 20 * x[..., 1] - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/matyar.html
        term1 = 0.26 * (x1**2 + x2**2)
        term2 = -0.48 * x1 * x2
        y = term1 + term2
        return y


class McCormick(TestFunction):
    """
    The McCormick function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/mccorm.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "McCormick is only defined for 2D inputs"

        # denormalize to x1 in [-1.5, 4], x2 in [-3, 4]
        x1 = 5 * x[..., 0] - 1.5
        x2 = 7 * x[..., 1] - 3

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/mccormr.html
        term1 = jnp.sin(x1 + x2)
        term2 = (x1 - x2) ** 2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        y = term1 + term2 + term3 + term4 + 1

        # shift so that global minimum is 0.0
        y = y + 1.9133
        return y


class PowerSum(TestFunction):
    """
    The Power Sum function. The recommended value of the b-vector, for d = 4, is: b = (8, 18, 44, 114).
    See https://www.sfu.ca/~ssurjano/powersum.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [0, d]
        d = x.shape[-1]
        x = d * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/powersumr.html
        # this b ensures x* = (1, 2, ..., d) and f(x*) = 0.0
        ii = jnp.arange(1, d + 1)
        b = jnp.sum(ii[:, None] ** ii, axis=-2)
        inner = jnp.sum(x[..., None] ** ii, axis=-2)
        y = jnp.sum((inner - b) ** 2, axis=-1)
        return y


class Zakharov(TestFunction):
    """
    The Zakharov function has no local minima except the global one.
    See https://www.sfu.ca/~ssurjano/zakharov.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-5, 10]
        x = 15 * x - 5

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/zakharovr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        sum1 = jnp.sum(x**2, axis=-1)
        sum2 = jnp.sum(0.5 * ii * x, axis=-1)
        y = sum1 + sum2**2 + sum2**4
        return y


# endregion
################################################################################

#################################################################################
# region Valley-Shaped


class Camel3(TestFunction):
    """
    The Three-Hump Camel function has three local minima.
    See https://www.sfu.ca/~ssurjano/camel3.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Camel3 is only defined for 2D inputs"

        # denormalize to [-5, 5]
        x1 = 10 * x[..., 0] - 5
        x2 = 10 * x[..., 1] - 5

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/camel3r.html
        term1 = 2 * x1**2
        term2 = -1.05 * x1**4
        term3 = x1**6 / 6
        term4 = x1 * x2
        term5 = x2**2
        y = term1 + term2 + term3 + term4 + term5
        return y


class Camel6(TestFunction):
    """
    The Six-Hump Camel function has six local minima, two of which are global.
    See https://www.sfu.ca/~ssurjano/camel6.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Camel6 is only defined for 2D inputs"

        # denormalize to x1 in [-3, 3], x2 in [-2, 2]
        x1 = 6 * x[..., 0] - 3
        x2 = 4 * x[..., 1] - 2

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/camel6r.html
        term1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        y = term1 + term2 + term3

        # shift so that global minimum is 0.0
        y = y + 1.0316
        return y


class DixonPrice(TestFunction):
    """
    The Dixon-Price function is continuous and unimodal.
    See https://www.sfu.ca/~ssurjano/dixonpr.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-10, 10]
        x = 20 * x - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/dixonprr.html
        ii = jnp.arange(2, x.shape[-1] + 1)
        term1 = (x[..., 0] - 1) ** 2
        term2 = ii * (2 * x[..., 1:] ** 2 - x[..., :-1]) ** 2
        y = term1 + jnp.sum(term2, axis=-1)
        return y


class Rosenbrock(TestFunction):
    """
    The Rosenbrock function, also referred to as the Valley or Banana function, has a narrow, curved valley containing the global minimum.
    See https://www.sfu.ca/~ssurjano/rosen.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-5, 10]
        x = 15 * x - 5

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rosenr.html
        term1 = 100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2
        term2 = (x[..., :-1] - 1) ** 2
        y = jnp.sum(term1 + term2, axis=-1)
        return y


# endregion
################################################################################

#################################################################################
# region Steep Ridges/Drops


class DeJong5(TestFunction):
    """
    The fifth function of De Jong is multimodal, with very sharp drops on a mainly flat surface.
    See https://www.sfu.ca/~ssurjano/dejong5.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "DeJong5 is only defined for 2D inputs"

        # denormalize to [-65.536, 65.536]
        x1 = 131.072 * x[..., 0] - 65.536
        x2 = 131.072 * x[..., 1] - 65.536

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/dejong5r.html
        A = jnp.array([[-32, -16, 0, 16, 32]] * 5)
        ii = jnp.arange(1, 25 + 1)
        term1 = (x1[..., None] - A.flatten()) ** 6
        term2 = (x2[..., None] - A.T.flatten()) ** 6
        inner = jnp.sum(1 / (ii + term1 + term2), axis=-1)
        y = 1 / (0.002 + inner)

        # shift so that global minimum is 0.0
        y = y - 0.998
        return y


class Easom(TestFunction):
    """
    The Easom function has several local minima. It is unimodal, and the global minimum has a small area relative to the search space.
    See https://www.sfu.ca/~ssurjano/easom.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Easom is only defined for 2D inputs"

        # denormalize to [-100, 100]
        x1 = 200 * x[..., 0] - 100
        x2 = 200 * x[..., 1] - 100

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/easomr.html
        term1 = -jnp.cos(x1) * jnp.cos(x2)
        term2 = jnp.exp(-((x1 - jnp.pi) ** 2) - (x2 - jnp.pi) ** 2)
        y = term1 * term2

        # shift so that global minimum is 0.0
        y = y + 1.0
        return y


class Michalewicz(TestFunction):
    """
    The Michalewicz function has d! local minima, and it is multimodal.
    The parameter m defines the steepness of they valleys and ridges, a larger m leads to a more difficult search.
    See https://www.sfu.ca/~ssurjano/michal.html for original implementation and more details.
    """

    def __init__(self, m: float = 10.0):
        self.m = m

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [0, pi]
        x = jnp.pi * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/michalr.html
        d = x.shape[-1]
        ii = jnp.arange(1, d + 1)
        inner = jnp.sin(x) * jnp.sin(ii / jnp.pi * x**2) ** (2 * self.m)
        y = -jnp.sum(inner, axis=-1)

        # shift so that global minimum is 0.0
        # fit a power law on numerical approximations of the global minimum
        # d=1 -> f*=-0.8013, d=5 -> f*=-4.687658, d=10 -> f*=-9.66015
        y = y + 0.8013 * d ** jnp.log10(9.66015 / 0.8013)
        return y


# endregion
################################################################################

#################################################################################
# region Other Functions


class Beale(TestFunction):
    """
    The Beale function is multimodal, with sharp peaks at the corners of the input domain.
    See https://www.sfu.ca/~ssurjano/beale.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Beale is only defined for 2D inputs"

        # denormalize to [-4.5, 4.5]
        x1 = 9 * x[..., 0] - 4.5
        x2 = 9 * x[..., 1] - 4.5

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/bealer.html
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2**2) ** 2
        term3 = (2.625 - x1 + x1 * x2**3) ** 2
        y = term1 + term2 + term3
        return y


class Branin(TestFunction):
    """
    The Branin, or Branin-Hoo, function has three global minima.
    See https://www.sfu.ca/~ssurjano/branin.html for original implementation and more details.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 5.1 / (2.0 * jnp.pi) ** 2,
        c: float = 5.0 / jnp.pi,
        r: float = 6.0,
        s: float = 10.0,
        t: float = 1.0 / (8.0 * jnp.pi),
    ):
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "Branin is only defined for 2D inputs"

        # denormalize to x1 in [-5, 10], x2 in [0, 15]
        x1 = 15 * x[..., 0] - 5
        x2 = 15 * x[..., 1]

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/braninr.html
        term1 = self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2
        term2 = self.s * (1 - self.t) * jnp.cos(x1)
        y = term1 + term2 + self.s

        # shift so that global minimum is 0.0
        y = y - 0.397887
        return y


class Colville(TestFunction):
    """
    The Colville function is a 4D function with several local minima.
    See https://www.sfu.ca/~ssurjano/colville.html for original implementation
    """

    def __call__(self, x: Float[Array, "... 4"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 4, "Colville function is only defined for 4D inputs."

        # denormalize to [-10, 10]
        x1 = 20 * x[..., 0] - 10
        x2 = 20 * x[..., 1] - 10
        x3 = 20 * x[..., 2] - 10
        x4 = 20 * x[..., 3] - 10

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/colviller.html
        term1 = 100 * (x1**2 - x2) ** 2
        term2 = (x1 - 1) ** 2
        term3 = (x3 - 1) ** 2
        term4 = 90 * (x3**2 - x4) ** 2
        term5 = 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        term6 = 19.8 * (x2 - 1) * (x4 - 1)
        y = term1 + term2 + term3 + term4 + term5 + term6
        return y


class Forrester(TestFunction):
    """
    This function is a simple one-dimensional test function.
    It is multimodal, with one global minimum, one local minimum and a zero-gradient inflection point.
    See https://www.sfu.ca/~ssurjano/forretal08.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 1"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 1, "Forrester is only defined for 1D inputs"

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/forretal08r.html
        fact1 = (6 * x - 2) ** 2
        fact2 = jnp.sin(12 * x - 4)
        y = fact1 * fact2

        # shift so that global minimum is 0.0
        y = y + 6.02075
        return y


class GoldsteinPrice(TestFunction):
    """The Goldstein-Price function has several local minima.
    See https://www.sfu.ca/~ssurjano/goldpr.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 2, "GoldsteinPrice is only defined for 2D inputs"

        # denormalize to [-2, 2]
        x1 = 4 * x[..., 0] - 2
        x2 = 4 * x[..., 1] - 2

        # port of matlab implementation from https://www.sfu.ca/~ssurjano/Code/goldprr.html
        fact1a = (x1 + x2 + 1) ** 2
        fact1b = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        fact1 = 1 + fact1a * fact1b
        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        fact2 = 30 + fact2a * fact2b
        y = fact1 * fact2

        # shift so that global minimum is 0.0
        y = y - 3.0
        return y


class Hartmann3(TestFunction):
    """
    The 3-dimensional Hartmann function has 4 local minima.
    See https://www.sfu.ca/~ssurjano/hart3.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 3, "Hartmann3 is only defined for 3D inputs"

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/hart3r.html
        alpha: Float[Array, "4"] = jnp.array([1.0, 1.2, 3.0, 3.2])
        A: Float[Array, "3 4"] = jnp.array(
            [
                [3.0, 0.1, 3.0, 0.1],
                [10.0, 10.0, 10.0, 10.0],
                [30.0, 35.0, 30.0, 35.0],
            ]
        )
        P: Float[Array, "3 4"] = jnp.array(
            [
                [0.3689, 0.4699, 0.1091, 0.0381],
                [0.1170, 0.4387, 0.8732, 0.5743],
                [0.2673, 0.7470, 0.5547, 0.8828],
            ]
        )
        inner = jnp.sum(A * (x[..., None] - P) ** 2, axis=-2)
        y = -jnp.sum(alpha * jnp.exp(-inner), axis=-1)

        # shift so that global minimum is 0.0
        y = y + 3.86278
        return y


class Hartmann4(TestFunction):
    """
    The 4-dimensional Hartmann function is multimodal. It is given here in the form of Picheny et al. (2012) having a mean of zero and a variance of one.
    See https://www.sfu.ca/~ssurjano/hart4.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 4"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 4, "Hartmann4 is only defined for 4D inputs"

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/hart4r.html
        alpha: Float[Array, "4"] = jnp.array([1.0, 1.2, 3.0, 3.2])
        A: Float[Array, "4 4"] = jnp.array(
            [
                [10.0, 3.0, 17.0, 3.5],
                [0.05, 10.0, 17.0, 0.1],
                [3.0, 3.5, 1.7, 10.0],
                [17.0, 8.0, 0.05, 10.0],
            ]
        )
        P: Float[Array, "4 4"] = jnp.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124],
                [0.2329, 0.4135, 0.8307, 0.3736],
                [0.2348, 0.1451, 0.3522, 0.2883],
                [0.4047, 0.8828, 0.8732, 0.5743],
            ]
        )
        inner = jnp.sum(A * (x[..., None, :] - P) ** 2, axis=-1)
        outer = jnp.sum(alpha * jnp.exp(-inner), axis=-1)
        y = (1.1 - outer) / 0.839

        # shift so that global minimum is 0.0
        y = y + 3.1345
        return y


class Hartmann6(TestFunction):
    """
    The 6-dimensional Hartmann function is multimodal. It is given here in the form of Picheny et al. (2012) having a mean of zero and a variance of one.
    See https://www.sfu.ca/~ssurjano/hart6.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 6"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 6, "Hartmann6 is only defined for 6D inputs"

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/hart6r.html
        alpha: Float[Array, "4"] = jnp.array([1.0, 1.2, 3.0, 3.2])
        A: Float[Array, "6 4"] = jnp.array(
            [
                [10.0, 0.05, 3.0, 17.0],
                [3.0, 10.0, 3.5, 8.0],
                [17.0, 17.0, 1.7, 0.05],
                [3.5, 0.1, 10.0, 10.0],
                [1.7, 8.0, 17.0, 0.1],
                [8.0, 14.0, 8.0, 14.0],
            ]
        )
        P: Float[Array, "6 4"] = jnp.array(
            [
                [0.1312, 0.2329, 0.2348, 0.4047],
                [0.1696, 0.4135, 0.1451, 0.8828],
                [0.5569, 0.8307, 0.3522, 0.8732],
                [0.0124, 0.3736, 0.2883, 0.5743],
                [0.8283, 0.1004, 0.3047, 0.1091],
                [0.5886, 0.9991, 0.0665, 0.0381],
            ]
        )
        inner = jnp.sum(A * (x[..., None] - P) ** 2, axis=-2)
        outer = jnp.sum(alpha * jnp.exp(-inner), axis=-1)
        y = -(2.58 + outer) / 1.94

        # shift so that global minimum is 0.0
        y = y + 3.0621033
        return y


class Perm(TestFunction):
    """
    The Perm function is multimodal.
    See https://www.sfu.ca/~ssurjano/permdb.html for original implementation and more details.
    """

    def __init__(self, beta: float = 10.0):
        self.beta = beta

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-d, d]
        d = x.shape[-1]
        x = 2 * d * x - d

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/perm0dbr.html
        ii = jnp.arange(1, d + 1)
        xi = (x[..., None, :] / ii) ** ii[..., None]
        inner = (ii ** ii[..., None] + self.beta) * (xi - 1)
        outer = jnp.sum(jnp.sum(inner, axis=-1) ** 2, axis=-1)
        y = outer
        return y


class Powell(TestFunction):
    """
    The Powell function is multimodal.
    See https://www.sfu.ca/~ssurjano/powell.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... 4d"]) -> Float[Array, "..."]:
        assert x.shape[-1] % 4 == 0, "Powell function only defined for dimensions 4n."

        # denormalize to [-4, 5]
        x = 9 * x - 4

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/powellr.html
        xx1 = x[..., 0::4]
        xx2 = x[..., 1::4]
        xx3 = x[..., 2::4]
        xx4 = x[..., 3::4]
        term1 = (xx1 + 10 * xx2) ** 2
        term2 = 5 * (xx3 - xx4) ** 2
        term3 = (xx2 - 2 * xx3) ** 4
        term4 = 10 * (xx1 - xx4) ** 4
        y = jnp.sum(term1 + term2 + term3 + term4, axis=-1)
        return y


class Shekel(TestFunction):
    """
    The Shekel function has m local minima.
    See https://www.sfu.ca/~ssurjano/shekel.html for original implementation and more details.
    """

    def __init__(self, m: int = 10):
        assert 1 <= m <= 10, "Shekel function is only defined for m in [1, 10]."

        self.b: Float[Array, "m"] = 0.1 * jnp.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])[:m]
        self.C: Float[Array, "4 m"] = jnp.array(
            [
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            ]
        )[:, :m]
        self.ymin = [
            -10.0000,
            -10.0277,
            -10.0433,
            -10.1043,
            -10.1532,
            -10.1704,
            -10.4029,
            -10.4226,
            -10.4832,
            -10.5364,
        ][m - 1]

    def __call__(self, x: Float[Array, "... 4"]) -> Float[Array, "..."]:
        assert x.shape[-1] == 4, "Shekel function is only defined for 4D inputs."

        # denormalize to [0, 10]
        x = 10 * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/shekelr.html
        inner = jnp.sum((x[..., None] - self.C) ** 2, axis=-2)
        y = jnp.sum(1 / (inner + self.b))

        # shift so that global minimum is 0.0
        y = y - self.ymin
        return y


class StyblinskiTang(TestFunction):
    """
    The The Styblinski-Tang function is multimodal.
    See https://www.sfu.ca/~ssurjano/stybtang.html for original implementation and more details.
    """

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # denormalize to [-5, 5]
        x = 10 * x - 5

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/stybtangr.html
        y = 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x, axis=-1)

        # shift so that global minimum is 0.0
        y = y + 39.16599 * x.shape[-1]
        return y


# endregion
#################################################################################
