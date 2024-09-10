# implement the Gaussian Process model
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib.pyplot as plt

class GP:
    def __init__(self, kernel, noise=1e-3):
        self.kernel = kernel
        self.noise = noise
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        K = self.kernel(X, X)
        self.L = la.cholesky(K + self.noise * np.eye(len(X)), lower=True)
        self.alpha = la.cho_solve((self.L, True), y)

    def predict(self, X):
        Ks = self.kernel(self.X, X)
        mu = np.dot(Ks.T, self.alpha)
        v = la.solve(self.L, Ks)
        var = self.kernel(X, X) - np.dot(v.T, v)
        return mu, var

    def sample(self, X, n=1):
        K = self.kernel(X, X)
        L = la.cholesky(K + self.noise * np.eye(len(X)), lower=True)
        f = np.dot(L, np.random.randn(len(X), n))
        return f

    def log_likelihood(self):
        return -0.5 * np.dot(self.y, self.alpha) - np.sum(np.log(np.diag(self.L))) - 0.5 * len(self.y) * np.log(2 * np.pi)

    def optimize(self):
        def f(x):
            self.noise = np.exp(x[0])
            self.fit(self.X, self.y)
            return -self.log_likelihood()

        x0 = np.log([self.noise])
        opt.minimize(f, x0, method='L-BFGS-B')

# test
gp = GP(lambda X, Y: np.exp(-0.5 * np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1))
        , noise=1e-3)
X = np.random.rand(10, 1)
y = np.sin(X * 10) + 0.1 * np.random.randn(10, 1)
gp.fit(X, y)
print(gp.log_likelihood())
gp.optimize()
print(gp.noise)
print(gp.log_likelihood())
Xs = np.linspace(0, 1, 100)[:, None]
mu, var = gp.predict(Xs)
plt.plot(X, y, 'o')
plt.plot(Xs, mu)
plt.fill_between(Xs.ravel(), mu.ravel() - 2 * np.sqrt(np.diag(var)), mu.ravel() + 2 * np.sqrt(np.diag(var)), alpha=0.2)
plt.show()
f = gp.sample(Xs, n=5)
plt.plot(Xs, f)
plt.show()
