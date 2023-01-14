import numpy as np
from functools import reduce
import scipy
from scipy import stats, optimize, integrate, special, linalg
from numpy import sqrt, exp, log, sign, array, ndarray


class LikelihoodComputer:

    debug_mode = False

    def __init__(self, data, toys=30000):
        """
        :param data: a Data object.
        :param toys: number of toys when marginalizing
        """

        self.model = data
        self.toys = toys

    def dNLLdMu(self, mu, theta_hat=None):
        """
        d (- ln L)/d mu, if L is the likelihood. The function
        whose root gives us muhat, i.e. the mu that maximizes
        the likelihood.

        :param mu: total number of signal events
        :param theta_hat: array with nuisance parameters, if None then
                          compute them

        """
        if isinstance(mu, (list, np.ndarray)) and len(mu) == 1:
            mu = float(mu[0])
        if theta_hat is None:
            theta_hat, _ = self.findThetaHat(mu)
        nsig = self.model.nsignal
        if not self.model.isLinear():
            raise ValueError("implemented only for linear model")
        # n_pred^i := mu s_i + b_i + theta_i
        # NLL = sum_i [ - n_obs^i * ln ( n_pred^i ) + n_pred^i ]
        # d NLL / d mu = sum_i [ - ( n_obs^i * s_ i ) / n_pred_i + s_i ]

        # Define relative signal strengths:
        n_pred = mu * nsig + self.model.backgrounds + theta_hat

        for ctr, d in enumerate(n_pred):
            if d == 0.0:
                if (self.model.observed[ctr] * nsig[ctr]) == 0.0:
                    #    logger.debug("zero denominator, but numerator also zero, so we set denom to 1.")
                    n_pred[ctr] = 1e-5
                else:
                    # n_pred[ctr]=1e-5
                    raise Exception(
                        "we have a zero value in the denominator at pos %d, with a non-zero numerator. dont know how to handle."
                        % ctr
                    )
        ret = -self.model.observed * nsig / n_pred + nsig

        if isinstance(ret, (np.ndarray, list)):
            ret = sum(ret)
        return ret

    def extendedOutput(self, extended_output, default=None):
        if extended_output:
            ret = {"muhat": default, "sigma_mu": default, "lmax": default}
            return ret
        return default

    def findAvgr(self, theta_hat):
        """from the difference observed - background, find got inital
        values for lower and upper"""
        mu_c = self.model.observed - self.model.backgrounds - theta_hat
        mu_r, wmu_r = [], []
        hessian = self.d2NLLdMu2(1.0, theta_hat)
        wtot = 0.0
        for s in zip(mu_c, self.model.nsignal, hessian):
            if s[1] > 1e-16:
                w = 1.0  # 1e-5
                if s[2] > 0.0:
                    w = s[2]
                wtot += w
                r = s[0] / s[1]
                mu_r.append(r)
                wmu_r.append(w * r)
        if len(mu_r) == 0:
            return None, None, None
        ret = min(mu_r), sum(wmu_r) / wtot, max(mu_r)
        return ret

    def d2NLLdMu2(self, mu, theta_hat, allowZeroHessian=True):
        """the hessian of the likelihood of mu, at mu,
        which is the Fisher information
        which is approximately the inverse of the covariance
        :param allowZeroHessian: if false and sum(observed)==0, then replace
                                 observed with expected
        """
        # nll=-nobs*ln(mu*s + b + theta) + ( mu*s + b + theta)
        # d nll / d mu = - nobs * s / ( mu*s + b + theta) + s
        # d2nll / dmu2 = nobs * s**2 / ( mu*s + b + theta )**2
        n_pred = mu * self.model.nsignal + self.model.backgrounds + theta_hat
        for i, s in enumerate(n_pred):
            if s == 0.0:  # the denominator in the hessian is 0?
                if (self.model.observed[i] * self.model.nsignal[i]) == 0.0:
                    #    logger.debug("zero denominator, but numerator also zero, so we set denom to 1.")
                    n_pred[i] = 1.0
                else:
                    raise Exception(
                        f"we have a zero value in the denominator at pos {i}, with a non-zero numerator. dont know how to handle."
                    )
        obs = self.model.observed
        if sum(obs) == 0 and not allowZeroHessian:
            obs = self.model.backgrounds
        hessian = obs * self.model.nsignal**2 / n_pred**2
        if sum(hessian) == 0.0 and not allowZeroHessian:
            # if all observations are zero, we replace them by the expectations
            if sum(self.model.observed) == 0:
                hessian = self.model.nsignal**2 / n_pred
        return hessian

    # def findMuHat(
    def findMuHatViaBracketing(
        self, allowNegativeSignals=False, extended_output=False, nll=False, marginalize=False
    ):
        """
        Find the most likely signal strength mu via a brent bracketing technique
        given the relative signal strengths in each dataset (signal region).

        :param allowNegativeSignals: if true, then also allow for negative values
        :param extended_output: if true, return also sigma_mu, the estimate of the error of mu_hat,
         and lmax, the likelihood at mu_hat
        :param nll: if true, return nll instead of lmax in the extended output

        :returns: mu_hat, i.e. the maximum likelihood estimate of mu, if extended output is
        requested, it returns a dictionary with mu_hat, sigma_mu -- the standard deviation around mu_hat, and lmax, i.e. the likelihood at mu_hat
        """
        if (self.model.backgrounds == self.model.observed).all():
            return self.extendedOutput(extended_output, 0.0)
        nsig = self.model.nsignal

        if isinstance(nsig, list, np.ndarray):
            nsig = np.array(nsig)

        nsig[nsig == 0.0] = 1e-20
        if sum(nsig < 0.0):
            raise Exception("Negative relative signal strengths!")

        ## we need a very rough initial guess for mu(hat), to come
        ## up with a first theta
        # self.nsig = array([0.]*len(self.model.observed))
        self.mu = 1.0
        ## we start with theta_hat being all zeroes
        # theta_hat = array([0.]*len(self.model.observed))
        mu_hat_old, mu_hat = 0.0, 1.0
        ctr = 0
        widener = 3.0
        while (
            abs(mu_hat - mu_hat_old) > 1e-10
            and abs(mu_hat - mu_hat_old) / (mu_hat + mu_hat_old) > 0.5e-2
            and ctr < 20
        ):
            theta_hat, _ = self.findThetaHat(mu_hat)
            ctr += 1
            mu_hat_old = mu_hat
            minr, avgr, maxr = self.findAvgr(theta_hat)
            # for i,s in enumerate ( signal_rel ):
            #    if abs(s) < 1e-19:
            #        mu_c[i]=0.
            ## find mu_hat by finding the root of 1/L dL/dmu. We know
            ## that the zero has to be between min(mu_c) and max(mu_c).
            lstarters = [
                avgr - 0.2 * abs(avgr),
                minr,
                0.0,
                -1.0,
                1.0,
                10.0,
                -0.1,
                0.1,
                -100.0,
                100.0,
                -1000.0,
            ]
            closestl, closestr = None, float("inf")
            for lower in lstarters:
                lower_v = self.dNLLdMu(lower, theta_hat)
                if lower_v < 0.0:
                    break
                if lower_v < closestr:
                    closestl, closestr = lower, lower_v
            if lower_v > 0.0:
                return self.extendedOutput(extended_output, 0.0)
            ustarters = [
                avgr + 0.2 * abs(avgr),
                maxr,
                0.0,
                1.0,
                10.0,
                -1.0 - 0.1,
                0.1,
                100.0,
                -100.0,
                1000.0,
                -1000.0,
                0.01,
                -0.01,
            ]
            closestl, closestr = None, float("inf")
            for upper in ustarters:
                upper_v = self.dNLLdMu(upper, theta_hat)
                if upper_v > 0.0:
                    break
                if upper_v < closestr:
                    closestl, closestr = upper, upper_v
            if upper_v < 0.0:
                return self.extendedOutput(extended_output, 0.0)
            mu_hat = scipy.optimize.brentq(self.dNLLdMu, lower, upper, args=(theta_hat,), rtol=1e-9)
            if not allowNegativeSignals and mu_hat < 0.0:
                mu_hat = 0.0
                theta_hat, _ = self.findThetaHat(mu_hat)
            self.theta_hat = theta_hat

        if extended_output:
            sigma_mu = self.getSigmaMu(mu_hat, theta_hat)
            llhd = self.likelihood(mu_hat, marginalize=marginalize, nll=nll)
            # print ( f"returning {allowNegativeSignals}: mu_hat {mu_hat}+-{sigma_mu} llhd {llhd}" )
            ret = {"muhat": mu_hat, "sigma_mu": sigma_mu, "lmax": llhd}
            return ret
        return mu_hat

    def getSigmaMu(self, mu, theta_hat):
        """
        Get an estimate for the standard deviation of mu at <mu>, from
        the inverse hessian
        """
        if not self.model.isLinear():
            print("implemented only for linear model")
        # d^2 mu NLL / d mu^2 = sum_i [ n_obs^i * s_i**2 / n_pred^i**2 ]
        hessian = self.d2NLLdMu2(mu, theta_hat, allowZeroHessian=False)
        hessian = sum(hessian)
        if hessian == 0.0:
            hessian = 1e-10
        """
            # if all observations are zero, we replace them by the expectations
            if sum(self.model.observed) == 0:
                hessian = sum(nsig**2 / n_pred)
        """
        stderr = float(np.sqrt(1.0 / hessian))
        return stderr

    # Define integrand (gaussian_(bg+signal)*poisson(nobs)):
    # def prob(x0, x1 )
    def llhdOfTheta(self, theta, nll=True):
        """likelihood for nuicance parameters theta, given signal strength
            self.mu. notice, by default it returns nll
        :param theta: nuisance parameters
        :params nll: if True, compute negative log likelihood
        """
        # theta = array ( thetaA )
        # ntot = self.model.backgrounds + self.nsig
        nsig = self.mu * self.model.nsignal
        if self.model.isLinear():
            lmbda = self.model.backgrounds + nsig + theta
        else:
            lmbda = nsig + self.model.A + theta + self.model.C * theta**2 / self.model.B**2
        lmbda[lmbda <= 0.0] = 1e-30  ## turn zeroes to small values
        obs = self.model.observed

        def is_integer(x):
            if isinstance(x, (int, np.int64, np.int32)):
                return True
            elif isinstance(x, float):
                return x.is_integer()
            return False

        ## not needed for now
        allintegers = np.all([is_integer(i) for i in obs])
        if nll:
            if allintegers:
                poisson = scipy.stats.poisson.logpmf(obs, lmbda)
            else:
                poisson = -lmbda + obs * np.log(lmbda) - scipy.special.loggamma(obs + 1)
        else:
            if allintegers:
                poisson = scipy.stats.poisson.pmf(obs, lmbda)
            else:
                # poisson = np.exp(-lmbda)*np.power(lmbda,obs)/special.gamma(obs+1)
                logpoiss = -lmbda + obs * np.log(lmbda) - scipy.special.loggamma(obs + 1)
                poisson = np.exp(logpoiss)
        try:
            M = [0.0] * len(theta)
            C = self.model.V
            # if self.model.n == 1: I think not a good idea
            #    C = self.model.totalCovariance(self.nsig)
            dTheta = theta - M
            expon = -0.5 * np.dot(np.dot(dTheta, self.weight), dTheta) + self.logcoeff
            # print ( "expon", expon, "coeff", self.coeff )
            if nll:
                gaussian = expon  #  np.log ( self.coeff )
                # gaussian2 = stats.multivariate_normal.logpdf(theta,mean=M,cov=C)
                ret = -gaussian - sum(poisson)
            else:
                gaussian = np.exp(expon)
                # gaussian = self.coeff * np.exp ( expon )
                # gaussian2 = stats.multivariate_normal.pdf(theta,mean=M,cov=C)
                ret = gaussian * (reduce(lambda x, y: x * y, poisson))
            return ret
        except ValueError as e:
            raise Exception("ValueError %s, %s" % (e, self.model.V))
            # raise Exception("ValueError %s, %s" % ( e, self.model.totalCovariance(self.nsig) ))
            # raise Exception("ValueError %s, %s" % ( e, self.model.V ))

    def dNLLdTheta(self, theta):
        """the derivative of nll as a function of the thetas.
        Makes it easier to find the maximum likelihood."""
        # print ( f"nsig {self.nsig} {self.model.nsignal}" )
        nsig = self.mu * self.model.nsignal
        if self.model.isLinear():
            xtot = theta + self.model.backgrounds + nsig
            xtot[xtot <= 0.0] = 1e-30  ## turn zeroes to small values
            nllp_ = self.ones - self.model.observed / xtot + np.dot(theta, self.weight)
            return nllp_
        lmbda = nsig + self.model.A + theta + self.model.C * theta**2 / self.model.B**2
        lmbda[lmbda <= 0.0] = 1e-30  ## turn zeroes to small values
        # nllp_ = ( self.ones - self.model.observed / lmbda + np.dot( theta , self.weight ) ) * ( self.ones + 2*self.model.C * theta / self.model.B**2 )
        T = self.ones + 2 * self.model.C / self.model.B**2 * theta
        nllp_ = T - self.model.observed / lmbda * (T) + np.dot(theta, self.weight)
        return nllp_

    def d2NLLdTheta2(self, theta):
        """the Hessian of nll as a function of the thetas.
        Makes it easier to find the maximum likelihood."""
        # xtot = theta + self.ntot
        nsig = self.mu * self.model.nsignal
        if self.model.isLinear():
            xtot = theta + self.model.backgrounds + nsig
            xtot[xtot <= 0.0] = 1e-30  ## turn zeroes to small values
            nllh_ = self.weight + np.diag(self.model.observed / (xtot**2))
            return nllh_
        lmbda = nsig + self.model.A + theta + self.model.C * theta**2 / self.model.B**2
        lmbda[lmbda <= 0.0] = 1e-30  ## turn zeroes to small values
        T = self.ones + 2 * self.model.C / self.model.B**2 * theta
        # T_i = 1_i + 2*c_i/B_i**2 * theta_i
        nllh_ = (
            self.weight
            + np.diag(self.model.observed * T**2 / (lmbda**2))
            - np.diag(self.model.observed / lmbda * 2 * self.model.C / self.model.B**2)
            + np.diag(2 * self.model.C / self.model.B**2)
        )
        return nllh_

    def getThetaHat(
        self, nobs: np.ndarray, nb: np.ndarray, mu: float, covb: np.ndarray, max_iterations: int
    ) -> np.ndarray:
        """
        Compute nuisance parameter theta that maximizes the likelihood (poisson * gauss)
        by setting dNLL / dTheta = 0

        :param nobs: number of observed events (unnecessary input)
        :param nb: number of backgrounds (unnecessary input)
        :param mu: signal strength
        :param covb: covatiance matrix (given from experiment) (unnecessary input)
        :param max_iterations: max number of iterations (input never been used)
        :return:
        """

        nsig = mu * self.model.nsignal
        self.mu = mu
        # \sigma^2 = \Sigma + diag( (N_{sig} * \Delta_{sys})^2 )
        sigma2 = covb + np.diag((nsig * self.model.deltas_rel) ** 2)
        ## for now deal with variances only
        ntot = nb + nsig
        cov = np.array(sigma2)
        # weight = cov**(-1) ## weight matrix
        weight = linalg.inv(cov)
        diag_cov = np.diag(cov)
        # first: no covariances:
        q = diag_cov * (ntot - nobs)
        p = ntot + diag_cov
        thetamaxes = []
        # thetamax = -p / 2.0 * (1 - sign(p) * sqrt(1.0 - 4 * q / p**2))
        thetamax = -p / 2.0 + sign(p) * sqrt(p**2 / 4 - q)
        thetamaxes.append(thetamax)
        ndims = len(p)

        def distance(theta1, theta2):
            for ctr, i in enumerate(theta1):
                if i == 0.0:
                    theta1[ctr] = 1e-20
            for ctr, i in enumerate(theta2):
                if i == 0.0:
                    theta2[ctr] = 1e-20
            return sum(np.abs(theta1 - theta2) / np.abs(theta1 + theta2))

        ictr = 0
        while ictr < max_iterations:
            ictr += 1
            q = diag_cov * (ntot - nobs)
            p = ntot + diag_cov
            for i in range(ndims):
                # q[i] = diag_cov[i] * ( ntot[i] - nobs[i] )
                # p[i] = ntot[i] + diag_cov[i]
                for j in range(ndims):
                    if i == j:
                        continue
                    dq = thetamax[j] * ntot[i] * diag_cov[i] * weight[i, j]
                    dp = thetamax[j] * weight[i, j] * diag_cov[i]
                    if abs(dq / q[i]) > 0.3:
                        # logger.warning ( "too big a change in iteration." )
                        dq = np.abs(0.3 * q[i]) * np.sign(dq)
                    if abs(dp / p[i]) > 0.3:
                        # logger.warning ( "too big a change in iteration." )
                        dp = np.abs(0.3 * p[i]) * np.sign(dp)
                    q[i] += dq
                    p[i] += dp
                # thetamax = -p / 2.0 * (1 - sign(p) * sqrt(1.0 - 4 * q / p**2))
                thetamax = -p / 2.0 + sign(p) * sqrt(p**2 / 4 - q)
            thetamaxes.append(thetamax)
            if len(thetamaxes) > 2:
                d1 = distance(thetamaxes[-2], thetamax)
                d2 = distance(thetamaxes[-3], thetamaxes[-2])
                if d1 > d2:
                    raise Exception("diverging when computing thetamax: %f > %f" % (d1, d2))
                if d1 < 1e-5:
                    return thetamax
        return thetamax

    def findThetaHat(self, mu: float):
        """Compute nuisance parameters theta that maximize our likelihood
        (poisson*gauss).
        """
        mu = float(mu)
        if np.isinf(mu):
            return None
        nsig = mu * self.model.nsignal

        ## first step is to disregard the covariances and solve the
        ## quadratic equations
        ini = self.getThetaHat(
            self.model.observed, self.model.backgrounds, mu, self.model.covariance, 0
        )
        self.cov_tot = self.model.V
        # if self.model.n == 1:
        #    self.cov_tot = self.model.totalCovariance ( nsig )
        # if not self.model.isLinear():
        # self.cov_tot = self.model.V + self.model.var_s(nsig)
        # self.cov_tot = self.model.totalCovariance (nsig)
        self.weight = np.linalg.inv(self.cov_tot)
        # self.coeff = 1.
        logdet = np.linalg.slogdet(self.cov_tot)
        self.logcoeff = -self.model.n / 2 * np.log(2 * np.pi) - 0.5 * logdet[1]
        # self.coeff = (2*np.pi)**(-self.model.n/2) * np.exp(-.5* logdet[1] )
        # print ( "coeff", self.coeff, "n", self.model.n, "det", np.linalg.slogdet ( self.cov_tot ) )
        # print ( "cov_tot", self.cov_tot[:10] )
        self.ones = 1.0
        if type(self.model.observed) in [list, ndarray]:
            self.ones = np.ones(len(self.model.observed))
        self.gammaln = special.gammaln(self.model.observed + 1)
        try:
            ret_c = optimize.fmin_ncg(
                self.llhdOfTheta,
                ini,
                fprime=self.dNLLdTheta,
                fhess=self.d2NLLdTheta2,
                full_output=True,
                disp=0,
            )
            # then always continue with TNC
            if type(self.model.observed) in [int, float]:
                bounds = [(-10 * self.model.observed, 10 * self.model.observed)]
            else:
                bounds = [(-10 * x, 10 * x) for x in self.model.observed]
            ini = ret_c
            ret_c = optimize.fmin_tnc(
                self.llhdOfTheta, ret_c[0], fprime=self.dNLLdTheta, disp=0, bounds=bounds
            )
            # print ( "[findThetaHat] mu=%s bg=%s observed=%s V=%s, nsig=%s theta=%s, nll=%s" % ( self.nsig[0]/self.model.efficiencies[0], self.model.backgrounds, self.model.observed,self.model.covariance, self.nsig, ret_c[0], self.nllOfNuisances(ret_c[0]) ) )
            if ret_c[-1] not in [0, 1, 2]:
                return ret_c[0], ret_c[-1]
            else:
                return ret_c[0], 0
                logger.debug("tnc worked.")

            ret = ret_c[0]
            return ret, -2
        except (IndexError, ValueError) as e:
            logger.error("exception: %s. ini[-3:]=%s" % (e, ini[-3:]))
            raise Exception("cov-1=%s" % (self.model.covariance + self.model.var_s(nsig)) ** (-1))
        return ini, -1

    def marginalizedLLHD1D(self, mu, nll):
        """
        Return the likelihood (of 1 signal region) to observe nobs events given the
        predicted background nb, error on this background (deltab),
        signal strength of mu and the relative error on the signal (deltas_rel).

        :param mu: predicted signal strength (float)
        :param nobs: number of observed events (float)
        :param nb: predicted background (float)
        :param deltab: uncertainty on background (float)

        :return: likelihood to observe nobs events (float)

        """
        nsig = self.model.nsignal * mu
        self.sigma2 = self.model.covariance + self.model.var_s(nsig)  ## (self.model.deltas)**2
        self.sigma_tot = sqrt(self.sigma2)
        self.lngamma = math.lgamma(self.model.observed[0] + 1)
        #     Why not a simple gamma function for the factorial:
        #     -----------------------------------------------------
        #     The scipy.stats.poisson.pmf probability mass function
        #     for the Poisson distribution only works for discrete
        #     numbers. The gamma distribution is used to create a
        #     continuous Poisson distribution.
        #
        #     Why not a simple gamma function for the factorial:
        #     -----------------------------------------------------
        #     The gamma function does not yield results for integers
        #     larger than 170. Since the expression for the Poisson
        #     probability mass function as a whole should not be huge,
        #     the exponent of the log of this expression is calculated
        #     instead to avoid using large numbers.

        # Define integrand (gaussian_(bg+signal)*poisson(nobs)):
        def prob(x, nsig):
            poisson = exp(self.model.observed * log(x) - x - self.lngamma)
            gaussian = stats.norm.pdf(x, loc=self.model.backgrounds + nsig, scale=self.sigma_tot)

            return poisson * gaussian

        # Compute maximum value for the integrand:
        xm = self.model.backgrounds + nsig - self.sigma2
        # If nb + nsig = sigma2, shift the values slightly:
        if xm == 0.0:
            xm = 0.001
        xmax = (
            xm
            * (1.0 + sign(xm) * sqrt(1.0 + 4.0 * self.model.observed * self.sigma2 / xm**2))
            / 2.0
        )

        # Define initial integration range:
        nrange = 5.0
        a = max(0.0, xmax - nrange * sqrt(self.sigma2))
        b = xmax + nrange * self.sigma_tot
        like = integrate.quad(prob, a, b, (nsig), epsabs=0.0, epsrel=1e-3)[0]
        if like == 0.0:
            return 0.0

        # Increase integration range until integral converges
        err = 1.0
        ctr = 0
        while err > 0.01:
            ctr += 1
            if ctr > 10.0:
                raise Exception("Could not compute likelihood within required precision")

            like_old = like
            nrange = nrange * 2
            a = max(0.0, (xmax - nrange * self.sigma_tot)[0][0])
            b = (xmax + nrange * self.sigma_tot)[0][0]
            like = integrate.quad(prob, a, b, (nsig), epsabs=0.0, epsrel=1e-3)[0]
            if like == 0.0:
                continue
            err = abs(like_old - like) / like

        # Renormalize the likelihood to account for the cut at x = 0.
        # The integral of the gaussian from 0 to infinity gives:
        # (1/2)*(1 + Erf(mu/sqrt(2*sigma2))), so we need to divide by it
        # (for mu - sigma >> 0, the normalization gives 1.)
        norm = (1.0 / 2.0) * (
            1.0 + special.erf((self.model.backgrounds + nsig) / sqrt(2.0 * self.sigma2))
        )
        like = like / norm

        if nll:
            like = -log(like)

        return like[0][0]

    def marginalizedLikelihood(self, mu, nll):
        """compute the marginalized likelihood of observing nsig signal event"""
        if (
            self.model.isLinear() and self.model.n == 1
        ):  ## 1-dimensional non-skewed llhds we can integrate analytically
            return self.marginalizedLLHD1D(mu, nll)
        nsig = mu * self.model.nsignal

        vals = []
        self.gammaln = special.gammaln(self.model.observed + 1)
        thetas = stats.multivariate_normal.rvs(
            mean=[0.0] * self.model.n,
            # cov=(self.model.totalCovariance(nsig)),
            cov=self.model.V,
            size=self.toys,
        )  ## get ntoys values
        for theta in thetas:
            if self.model.isLinear():
                lmbda = nsig + self.model.backgrounds + theta
            else:
                lmbda = nsig + self.model.A + theta + self.model.C * theta**2 / self.model.B**2
            if self.model.isScalar(lmbda):
                lmbda = array([lmbda])
            for ctr, v in enumerate(lmbda):
                if v <= 0.0:
                    lmbda[ctr] = 1e-30
                # print ( "lmbda=",lmbda )
            poisson = self.model.observed * np.log(lmbda) - lmbda - self.gammaln
            # poisson = np.exp(self.model.observed*np.log(lmbda) - lmbda - self.model.backgrounds - self.gammaln)
            vals.append(np.exp(sum(poisson)))
            # vals.append ( reduce(lambda x, y: x*y, poisson) )
        mean = np.mean(vals)
        if nll:
            if mean == 0.0:
                mean = 1e-100
            mean = -log(mean)
        return mean

    def profileLikelihood(self, mu: float, nll: bool):
        """compute the profiled likelihood for mu.
        Warning: not normalized.
        Returns profile likelihood and error code (0=no error)
        """
        # compute the profiled (not normalized) likelihood of observing
        # nsig signal events
        theta_hat, _ = self.findThetaHat(mu)
        if self.debug_mode:
            self.theta_hat = theta_hat
        ret = self.llhdOfTheta(theta_hat, nll)

        return ret

    def likelihood(self, mu: float, marginalize: bool = False, nll: bool = False) -> float:
        """compute likelihood for mu, profiling or marginalizing the nuisances
        :param mu: float Parameter of interest, signal strength
        :param marginalize: if true, marginalize, if false, profile
        :param nll: return nll instead of likelihood
        """
        if marginalize:
            # p,err = self.profileLikelihood ( nsig, deltas )
            return self.marginalizedLikelihood(mu, nll)
            # print ( "p,l=",p,l,p/l )
        else:
            return self.profileLikelihood(mu, nll)

    def lmax(self, marginalize=False, nll=False, allowNegativeSignals=False):
        """convenience function, computes likelihood for nsig = nobs-nbg,
        :param marginalize: if true, marginalize, if false, profile nuisances.
        :param nll: return nll instead of likelihood
        :param allowNegativeSignals: if False, then negative nsigs are replaced with 0.
        """
        if len(self.model.observed) == 1:
            dn = self.model.observed - self.model.backgrounds
            if not allowNegativeSignals and dn[0] < 0.0:
                dn = [0.0]
            self.muhat = float(dn[0])
            if abs(self.model.nsignal) > 1e-100:
                self.muhat = float(dn[0] / self.model.nsignal[0])
            self.sigma_mu = np.sqrt(self.model.observed[0] + self.model.covariance[0][0])
            return self.likelihood(marginalize=marginalize, nll=nll, mu=self.muhat)
        fmh = self.findMuHat(
            allowNegativeSignals=allowNegativeSignals, extended_output=True, nll=nll
        )
        muhat_, sigma_mu, lmax = fmh["muhat"], fmh["sigma_mu"], fmh["lmax"]
        self.muhat = muhat_
        self.sigma_mu = sigma_mu
        return self.likelihood(marginalize=marginalize, nll=nll, mu=muhat_)

    def findMuHat(
        # def findMuHatViaGradientDescent(
        self,
        allowNegativeSignals=False,
        extended_output=False,
        nll=False,
        marginalize=False,
    ):
        """
        Find the most likely signal strength mu via gradient descent
        given the relative signal strengths in each dataset (signal region).

        :param allowNegativeSignals: if true, then also allow for negative values
        :param extended_output: if true, return also sigma_mu, the estimate of the error of mu_hat,
         and lmax, the likelihood at mu_hat
        :param nll: if true, return nll instead of lmax in the extended output

        :returns: mu_hat, i.e. the maximum likelihood estimate of mu, if extended output is
        requested, it returns mu_hat, sigma_mu -- the standard deviation around mu_hat, and llhd,
        the likelihood at mu_hat
        """
        theta_hat, _ = self.findThetaHat(0.0)
        minr, avgr, maxr = self.findAvgr(theta_hat)
        theta_hat, _ = self.findThetaHat(avgr)
        minr, avgr, maxr = self.findAvgr(theta_hat)

        def myllhd(mu: float):
            theta = self.findThetaHat(mu=float(mu))
            ret = self.likelihood(nll=True, marginalize=marginalize, mu=mu)
            return ret

        import scipy.optimize

        ominr = minr
        if minr > 0.0:
            minr = 0.5 * minr
        if minr < 0.0:
            minr = 2.0 * minr
        if maxr > 0.0:
            maxr = 3.0 * maxr + 1e-5
        if maxr <= 0.0:
            maxr = 0.3 * maxr + 1e-5

        bounds = [(minr, maxr)]
        if not allowNegativeSignals:
            bounds = [(0, max(maxr, 1e-5))]
        assert bounds[0][1] > bounds[0][0], f"bounds are in wrong order: {bounds}"
        o = scipy.optimize.minimize(myllhd, x0=avgr, bounds=bounds, jac=self.dNLLdMu)
        llhd = o.fun
        if not nll:
            llhd = np.exp(-o.fun)
        """
        hess = o.hess_inv
        try:
            hess = hess.todense()
        except Exception as e:
            pass
        """
        mu_hat = float(o.x[0])
        if extended_output:
            sigma_mu = self.getSigmaMu(mu_hat, theta_hat)
            llhd = self.likelihood(mu_hat, marginalize=marginalize, nll=nll)
            # sigma_mu = float(np.sqrt(hess[0][0]))
            ret = {"muhat": mu_hat, "sigma_mu": sigma_mu, "lmax": llhd}
            return ret
        return mu_hat

    def chi2(self, marginalize=False):
        """
        Computes the chi2 for a given number of observed events nobs given
        the predicted background nb, error on this background deltab,
        expected number of signal events nsig and the relative error on
        signal (deltas_rel).
        :param marginalize: if true, marginalize, if false, profile
        :param nsig: number of signal events
        :return: chi2 (float)

        """

        # Compute the likelhood for the null hypothesis (signal hypothesis) H0:
        llhd = self.likelihood(1.0, marginalize=marginalize, nll=True)

        # Compute the maximum likelihood H1, which sits at nsig = nobs - nb
        # (keeping the same % error on signal):
        if len(self.model.observed) == 1:
            # TODO this nsig initiation seems wrong and changing maxllhd to likelihood
            # fails ./testStatistics.py : zero division error in L115
            mu_hat = (self.model.observed - self.model.backgrounds) / self.model.nsignal
            maxllhd = self.likelihood(mu_hat, marginalize=marginalize, nll=True)
        else:
            maxllhd = self.lmax(marginalize=marginalize, nll=True, allowNegativeSignals=False)
        chi2 = 2.0 * (llhd - maxllhd)

        if not np.isfinite(chi2):
            raise Exception("chi2 is not a finite number! %s,%s,%s" % (chi2, llhd, maxllhd))
        # Return the test statistic -2log(H0/H1)
        return chi2
