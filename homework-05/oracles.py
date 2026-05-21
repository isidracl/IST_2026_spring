import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = len(self.b)
        Ax = self.matvec_Ax(x)
        margins = -self.b * Ax
        return (1.0 / m) * np.sum(np.logaddexp(0, margins)) + (self.regcoef / 2.0) * np.dot(x, x)

    def grad(self, x):
        m = len(self.b)
        Ax = self.matvec_Ax(x)
        sigmas = expit(-self.b * Ax)
        return -(1.0 / m) * self.matvec_ATx(self.b * sigmas) + self.regcoef * x

    def hess(self, x):
        m = len(self.b)
        Ax = self.matvec_Ax(x)
        s = expit(self.b * Ax) * expit(-self.b * Ax)  # shape (m,)
        return (1.0 / m) * self.matmat_ATsA(s) + self.regcoef * np.eye(len(x))


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._cache_x = None
        self._cache_Ax = None
        self._cache_d = None
        self._cache_Ad = None
        self._cache_x_hat = None
        self._cache_Ax_hat = None

    def _get_Ax(self, x):
        if self._cache_x_hat is not None and np.allclose(x, self._cache_x_hat):
            self._cache_x = x.copy()
            self._cache_Ax = self._cache_Ax_hat
            self._cache_x_hat = None
            self._cache_Ax_hat = None
            return self._cache_Ax
        if self._cache_x is None or not np.allclose(x, self._cache_x):
            self._cache_x = x.copy()
            self._cache_Ax = self.matvec_Ax(x)
        return self._cache_Ax

    def _get_Ad(self, d):
        if self._cache_d is None or not np.allclose(d, self._cache_d):
            self._cache_d = d.copy()
            self._cache_Ad = self.matvec_Ax(d)
        return self._cache_Ad

    def func(self, x):
        Ax = self._get_Ax(x)
        m = len(self.b)
        return (1.0 / m) * np.sum(np.logaddexp(0, -self.b * Ax)) + \
               (self.regcoef / 2.0) * np.dot(x, x)

    def grad(self, x):
        Ax = self._get_Ax(x)
        m = len(self.b)
        sigmas = expit(-self.b * Ax)
        return -(1.0 / m) * self.matvec_ATx(self.b * sigmas) + self.regcoef * x

    def hess(self, x):
        Ax = self._get_Ax(x)
        m = len(self.b)
        s = expit(self.b * Ax) * expit(-self.b * Ax)
        return (1.0 / m) * self.matmat_ATsA(s) + self.regcoef * np.eye(len(x))

    def func_directional(self, x, d, alpha):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(d)
        Ax_hat = Ax + alpha * Ad
        x_hat = x + alpha * d
        self._cache_x_hat = x_hat.copy()
        self._cache_Ax_hat = Ax_hat
        m = len(self.b)
        return float((1.0 / m) * np.sum(np.logaddexp(0, -self.b * Ax_hat)) +
                     (self.regcoef / 2.0) * np.dot(x_hat, x_hat))

    def grad_directional(self, x, d, alpha):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(d)
        Ax_hat = Ax + alpha * Ad
        x_hat = x + alpha * d
        self._cache_x_hat = x_hat.copy()
        self._cache_Ax_hat = Ax_hat
        m = len(self.b)
        sigmas = expit(-self.b * Ax_hat)
        return float(-(1.0 / m) * np.dot(self.b * sigmas, Ad) +
                    self.regcoef * np.dot(x_hat, d))


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        # A^T * diag(s) * A
        if scipy.sparse.issparse(A):
            return A.T.dot(A.multiply(s.reshape(-1, 1)))
        else:
            return A.T.dot(A * s.reshape(-1, 1))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    grad = np.zeros(n)
    f0 = func(x)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        grad[i] = (func(x + eps * e_i) - f0) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    H = np.zeros((n, n))
    f0 = func(x)
    f_i = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_i[i] = func(x + eps * e_i)
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1.0
            e_j[j] = 1.0
            f_ij = func(x + eps * e_i + eps * e_j)
            val = (f_ij - f_i[i] - f_i[j] + f0) / (eps ** 2)
            H[i, j] = val
            H[j, i] = val
    return H
