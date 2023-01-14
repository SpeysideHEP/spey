import numpy as np
import copy


class Data:
    """A very simple observed container to collect all the data
    needed to fully define a specific statistical model"""

    def __init__(
        self,
        observed,
        backgrounds,
        covariance,
        third_moment=None,
        nsignal=None,
        name="model",
        deltas_rel=0.2,
        lumi=None,
    ):
        """
        :param observed: number of observed events per dataset
        :param backgrounds: expected bg per dataset
        :param covariance: uncertainty in background, as a covariance matrix
        :param nsignal: number of signal events in each dataset
        :param name: give the model a name, just for convenience
        :param deltas_rel: the assumed relative error on the signal hypotheses.
                           The default is 20%.
        :param lumi: luminosity of dataset in 1/fb, or None
        """
        self.observed = self.convert(observed)  # Make sure observed number of events are integers
        ## self.observed = np.around(self.convert(observed)) #Make sure observed number of events are integers
        self.backgrounds = self.convert(backgrounds)
        self.n = len(self.observed)
        self.covariance = self._convertCov(covariance)
        self.nsignal = self.convert(nsignal)
        self.lumi = lumi
        if self.nsignal is None:
            if len(self.backgrounds) == 1:
                # doesnt matter, does it?
                self.nsignal = np.array([1.0])
            self.signal_rel = self.convert(1.0)
        elif self.nsignal.sum():
            self.signal_rel = self.nsignal / self.nsignal.sum()
        else:
            self.signal_rel = np.array([0.0] * len(self.nsignal))

        self.third_moment = self.convert(third_moment)
        if (
            type(self.third_moment) != type(None)
            and np.sum([abs(x) for x in self.third_moment]) < 1e-10
        ):
            self.third_moment = None
        self.name = name
        self.deltas_rel = deltas_rel
        self._computeABC()

    def get_expected(self):
        return Data(
            self.backgrounds,
            self.backgrounds,
            self.covariance,
            self.third_moment,
            self.nsignal,
            f"{self.name}_exp",
            self.deltas_rel,
        )

    def totalCovariance(self, nsig):
        """get the total covariance matrix, taking into account
        also signal uncertainty for the signal hypothesis <nsig>.
        If nsig is None, the predefined signal hypothesis is taken.
        """
        if self.isLinear():
            cov_tot = self.V + self.var_s(nsig)
        else:
            cov_tot = self.covariance + self.var_s(nsig)
        return cov_tot

    def zeroSignal(self):
        """
        Is the total number of signal events zero?
        """
        if self.nsignal is None:
            return True
        return len(self.nsignal[self.nsignal > 0.0]) == 0

    def var_s(self, nsig=None):
        """
        The signal variances. Convenience function.

        :param nsig: If None, it will use the model expected number of signal events,
                    otherwise will return the variances for the input value using the relative
                    signal uncertainty defined for the model.

        """

        if nsig is None:
            nsig = self.nsignal
        else:
            nsig = self.convert(nsig)
        return np.diag((nsig * self.deltas_rel) ** 2)

    def isScalar(self, obj):
        """
        Determine if obj is a scalar (float or int)
        """

        if isinstance(obj, np.ndarray):
            ## need to treat separately since casting array([0.]) to float works
            return False
        try:
            _ = float(obj)
            return True
        except (ValueError, TypeError):
            pass
        return False

    def convert(self, obj):
        """
        Convert object to numpy arrays.
        If object is a float or int, it is converted to a one element
        array.
        """

        if type(obj) == type(None):
            return obj
        if self.isScalar(obj):
            return np.array([obj])
        return np.array(obj)

    def __str__(self):
        return self.name + " (%d dims)" % self.n

    def _convertCov(self, obj):

        if self.isScalar(obj):
            return np.array([[obj]])
        if isinstance(obj[0], list):
            return np.array(obj)
        if isinstance(obj[0], float):
            ## if the matrix is flattened, unflatten it.
            return np.array([obj[self.n * i : self.n * (i + 1)] for i in range(self.n)])

        return obj

    def _computeABC(self):
        """
        Compute the terms A, B, C, rho, V. Corresponds with
        Eqs. 1.27-1.30 in arXiv:1809.05548
        """
        self.V = self.covariance
        if self.third_moment is None:
            self.A = None
            self.B = None
            self.C = None
            return

        covD = self.diagCov()
        C = []
        for m2, m3 in zip(covD, self.third_moment):
            if m3 == 0.0:
                m3 = 1e-30
            k = -np.sign(m3) * np.sqrt(2.0 * m2)
            dm = np.sqrt(8.0 * m2**3 / m3**2 - 1.0)
            C.append(k * np.cos(4.0 * np.pi / 3.0 + np.arctan(dm) / 3.0))

        self.C = np.array(C)  ## C, as define in Eq. 1.27 (?) in the second paper
        self.B = np.sqrt(covD - 2 * self.C**2)  ## B, as defined in Eq. 1.28(?)
        self.A = self.backgrounds - self.C  ## A, Eq. 1.30(?)
        self.rho = np.array([[0.0] * self.n] * self.n)  ## Eq. 1.29 (?)
        for x in range(self.n):
            for y in range(x, self.n):
                bxby = self.B[x] * self.B[y]
                cxcy = self.C[x] * self.C[y]
                e = (4.0 * cxcy) ** (-1) * (
                    np.sqrt(bxby**2 + 8 * cxcy * self.covariance[x][y]) - bxby
                )
                self.rho[x][y] = e
                self.rho[y][x] = e

        self.sandwich()
        # self.V = sandwich ( self.B, self.rho )

    def sandwich(self):
        """
        Sandwich product
        """

        ret = np.array([[0.0] * len(self.B)] * len(self.B))
        for x in range(len(self.B)):
            for y in range(x, len(self.B)):
                T = self.B[x] * self.B[y] * self.rho[x][y]
                ret[x][y] = T
                ret[y][x] = T
        self.V = ret

    def isLinear(self):
        """
        Statistical model is linear, i.e. no quadratic term in poissonians
        """

        return type(self.C) == type(None)

    def diagCov(self):
        """
        Diagonal elements of covariance matrix. Convenience function.
        """

        return np.diag(self.covariance)

    def correlations(self):
        """
        Correlation matrix, computed from covariance matrix.
        Convenience function.
        """

        if hasattr(self, "corr"):
            return self.corr

        self.corr = copy.deepcopy(self.covariance)
        for x in range(self.n):
            self.corr[x][x] = 1.0
            for y in range(x + 1, self.n):
                rho = self.corr[x][y] / np.sqrt(self.covariance[x][x] * self.covariance[y][y])
                self.corr[x][y] = rho
                self.corr[y][x] = rho
        return self.corr

    def rel_signals(self, mu):
        """
        Returns the number of expected relative signal events, for all datasets,
        given total signal strength mu. For mu=1, the sum of the numbers = 1.

        :param mu: Total number of signal events summed over all datasets.
        """

        return mu * self.signal_rel

    def nsignals(self, mu):
        """
        Returns the number of expected signal events, for all datasets,
        given total signal strength mu.

        :param mu: Total number of signal events summed over all datasets.
        """

        return mu * self.nsignal
