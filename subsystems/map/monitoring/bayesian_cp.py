import numpy as np


class BayesianChangePointDetector:
    """
    Bayesian Online Change Point Detection
    Gaussian unknown mean (Student-t predictive)
    Stable implementation for geophysical signals
    """

    def __init__(self, hazard=1 / 200):
        self.hazard = hazard

        # prior hyperparameters (very important!)
        self.mu0 = 0
        self.kappa0 = 1
        self.alpha0 = 1
        self.beta0 = 1

    @staticmethod
    def student_t_pdf(x, mu, kappa, alpha, beta):
        kappa = max(kappa, 1e-6)
        alpha = max(alpha, 1e-6)
        beta = max(beta, 1e-6)

        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        scale = max(scale, 1e-6)

        dof = max(2 * alpha, 1e-6)

        return (1 + ((x - mu) / scale) ** 2 / dof) ** (-(dof + 1) / 2)

    def run(self, signal):
        n = len(signal)
        rr = np.zeros((n + 1, n + 1))
        rr[0, 0] = 1

        mu = np.zeros((n + 1, n + 1))
        kappa = np.zeros((n + 1, n + 1))
        alpha = np.zeros((n + 1, n + 1))
        beta = np.zeros((n + 1, n + 1))

        mu[0, 0] = self.mu0
        kappa[0, 0] = self.kappa0
        alpha[0, 0] = self.alpha0
        beta[0, 0] = self.beta0

        cp_prob = np.zeros(n)

        for t in range(1, n):
            x = signal[t]

            pred_prob = np.zeros(t)
            for r in range(t):
                pred_prob[r] = self.student_t_pdf(
                    x, mu[r, t - 1], kappa[r, t - 1], alpha[r, t - 1], beta[r, t - 1]
                )

            growth = rr[t - 1, :t] * pred_prob * (1 - self.hazard)
            cp = np.sum(rr[t - 1, :t] * pred_prob * self.hazard)

            rr[t, 1 : t + 1] = growth
            rr[t, 0] = cp
            rr[t, : t + 1] /= np.sum(rr[t, : t + 1])

            cp_prob[t] = rr[t, 0]

            # update posterior
            for r in range(t):
                kappa_new = kappa[r, t - 1] + 1
                mu_new = (kappa[r, t - 1] * mu[r, t - 1] + x) / kappa_new
                alpha_new = alpha[r, t - 1] + 0.5
                beta_new = (
                    beta[r, t - 1]
                    + 0.5 * kappa[r, t - 1] * (x - mu[r, t - 1]) ** 2 / kappa_new
                )

                mu[r + 1, t] = mu_new
                kappa[r + 1, t] = kappa_new
                alpha[r + 1, t] = alpha_new
                beta[r + 1, t] = beta_new

        return cp_prob
