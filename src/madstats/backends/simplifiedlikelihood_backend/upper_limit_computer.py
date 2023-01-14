from scipy import optimize
from numpy import array
import numpy as np
import copy

from typing import Text, Optional, Union, Tuple

from smodels.tools.statistics import CLsfromNLL, determineBrentBracket
from smodels.experiment.exceptions import SModelSExperimentError as SModelSError

from .data import Data
from .likelihood_computer import LikelihoodComputer


class UpperLimitComputer:
    debug_mode = False

    def __init__(self, ntoys: float = 30000, cl: float = 0.95):

        """
        :param ntoys: number of toys when marginalizing
        :param cl: desired quantile for limits
        """
        self.toys = ntoys
        self.cl = cl

    def getUpperLimitOnSigmaTimesEff(
        self, model, marginalize=False, toys=None, expected=False, trylasttime=False
    ):
        """upper limit on the fiducial cross section sigma times efficiency,
            summed over all signal regions, i.e. sum_i xsec^prod_i eff_i
            obtained from the defined Data (using the signal prediction
            for each signal regio/dataset), by using
            the q_mu test statistic from the CCGV paper (arXiv:1007.1727).

        :params marginalize: if true, marginalize nuisances, else profile them
        :params toys: specify number of toys. Use default is none
        :params expected: if false, compute observed,
                          true: compute a priori expected, "posteriori":
                          compute a posteriori expected
        :params trylasttime: if True, then dont try extra
        :returns: upper limit on fiducial cross section
        """
        ul = self.getUpperLimitOnMu(
            model, marginalize=marginalize, toys=toys, expected=expected, trylasttime=trylasttime
        )

        if ul == None:
            return ul
        if model.lumi is None:
            raise Exception(
                f"asked for upper limit on fiducial xsec, but no lumi given with the data"
            )
            return ul
        xsec = sum(model.nsignal) / model.lumi
        return ul * xsec

    def getCLsRootFunc(
        self,
        data: Data,
        marginalize: Optional[bool] = False,
        toys: Optional[float] = None,
        expected: Optional[Union[bool, Text]] = False,
        trylasttime: Optional[bool] = False,
    ) -> Tuple:
        """
        Obtain the function "CLs-alpha[0.05]" whose root defines the upper limit,
        plus mu_hat and sigma_mu
        :param data: statistical model
        :param marginalize: if true, marginalize nuisances, else profile them
        :param toys: specify number of toys. Use default is none
        :param expected: if false, compute observed,
                          true: compute a priori expected, "posteriori":
                          compute a posteriori expected
        :param trylasttime: if True, then dont try extra
        :return: mu_hat, sigma_mu, CLs-alpha
        """

        if data.zeroSignal():
            """only zeroes in efficiencies? cannot give a limit!"""
            return None, None, None
        if toys is None:
            toys = self.toys

        model = copy.deepcopy(data)
        if expected:
            # Set expected model where nobs are equal to the nb
            theta_hat_ = np.zeros(data.observed.shape)
            if expected == "posteriori":
                tempc = LikelihoodComputer(data, toys)
                theta_hat_, _ = tempc.findThetaHat(0)
            model.observed = model.backgrounds + theta_hat_

        computer = LikelihoodComputer(model, toys)
        mu_hat = computer.findMuHat(allowNegativeSignals=False, extended_output=False)

        # Compute nuisance parameters that maximizes the likelihood without signal
        theta_hat0, _ = computer.findThetaHat(0.0)
        sigma_mu0 = computer.getSigmaMu(mu_hat, theta_hat0)

        nll0 = computer.likelihood(mu_hat, marginalize=marginalize, nll=True)
        if np.isinf(nll0) and not marginalize and not trylasttime:
            print(
                "nll is infinite in profiling! we switch to marginalization, but only for this one!"
            )
            marginalize = True
            # TODO convert rel_signals to signals
            nll0 = computer.likelihood(mu=mu_hat, marginalize=True, nll=True)
            if np.isinf(nll0):
                print("marginalization didnt help either. switch back.")
                marginalize = False
            else:
                print("marginalization worked.")

        aModel = copy.deepcopy(model)
        aModel.observed = array([x + y for x, y in zip(model.backgrounds, theta_hat0)])
        aModel.name = aModel.name + "A"
        compA = LikelihoodComputer(aModel, toys)
        mu_hatA = compA.findMuHat()
        nll0A = compA.likelihood(mu=mu_hatA, marginalize=marginalize, nll=True)

        def clsRoot(mu: float, return_type: Text = "CLs-alpha") -> float:
            """
            Calculate the root
            :param mu: float POI
            :param return_type: (Text) can be "CLs-alpha", "1-CLs", "CLs"
                        CLs-alpha: returns CLs - 0.05
                        1-CLs: returns 1-CLs value
                        CLs: returns CLs value
            """
            nll = computer.likelihood(mu, marginalize=marginalize, nll=True)
            nllA = compA.likelihood(mu, marginalize=marginalize, nll=True)
            return CLsfromNLL(nllA, nll0A, nll, nll0, return_type=return_type)

        return mu_hat, sigma_mu0, clsRoot

    def getUpperLimitOnMu(
        self, model, marginalize=False, toys=None, expected=False, trylasttime=False
    ):
        """upper limit on the signal strength multiplier mu
            obtained from the defined Data (using the signal prediction

            for each signal regio/dataset), by using
            the q_mu test statistic from the CCGV paper (arXiv:1007.1727).

        :params marginalize: if true, marginalize nuisances, else profile them
        :params toys: specify number of toys. Use default is none
        :params expected: if false, compute observed,
                          true: compute a priori expected, "posteriori":
                          compute a posteriori expected
        :params trylasttime: if True, then dont try extra
        :returns: upper limit on the signal strength multiplier mu
        """
        mu_hat, sigma_mu, clsRoot = self.getCLsRootFunc(
            model, marginalize, toys, expected, trylasttime
        )
        if mu_hat == None:
            return None
        try:
            a, b = determineBrentBracket(mu_hat, sigma_mu, clsRoot, allowNegative=False)
        except SModelSError as e:
            return None
        mu_lim = optimize.brentq(clsRoot, a, b, rtol=1e-03, xtol=1e-06)
        return mu_lim

    def computeCLs(
        self,
        model: Data,
        marginalize: bool = False,
        toys: float = None,
        expected: Union[bool, Text] = False,
        trylasttime: bool = False,
        return_type: Text = "1-CLs",
    ) -> float:
        """
        Compute the exclusion confidence level of the model (1-CLs)
        :param model: statistical model
        :param marginalize: if true, marginalize nuisances, else profile them
        :param toys: specify number of toys. Use default is none
        :param expected: if false, compute observed,
                          true: compute a priori expected, "posteriori":
                          compute a posteriori expected
        :param trylasttime: if True, then dont try extra
        :param return_type: (Text) can be "CLs-alpha", "1-CLs", "CLs"
                        CLs-alpha: returns CLs - 0.05 (alpha)
                        1-CLs: returns 1-CLs value
                        CLs: returns CLs value
        """
        _, _, clsRoot = self.getCLsRootFunc(model, marginalize, toys, expected, trylasttime)
        ret = clsRoot(1.0, return_type=return_type)
        # its not an uppser limit on mu, its on nsig
        return ret
