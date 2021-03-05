# https://github.com/smcantab/information_entropy/blob/master/information_entropy/examples/ising/utils.py

from __future__ import division
import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad, nquad
from scipy.special import ellipk, ellipkm1


def _ising_exact_magnetization_2D(T, k=1., J=1.):
    beta = 1. / (k * T)
    K = beta * J
    if T < 2. / np.log(1. + np.sqrt(2.)):
        m = (1. - np.sinh(2 * K) ** (-4)) ** (1. / 8)
    else:
        m = 0.
    return m


ising_exact_magnetization_2D = np.vectorize(_ising_exact_magnetization_2D)


def _ising_exact_energy_2D(T, k=1., J=1.):
    # http://www.lps.ens.fr/~krzakala/ISINGMODEL.pdf
    beta = 1. / (k * T)
    K = beta * J
    m = 2 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    u = - J / np.tanh(2 * K) * (1. + (2 * np.tanh(2 * K) ** 2 - 1.) * (2. / np.pi) * ellipk(m ** 2))
    return u


ising_exact_energy_2D = np.vectorize(_ising_exact_energy_2D)


# def _ising_exact_free_energy_2D_single(T, k=1., J=1.):
#     beta = 1. / (k * T)
#     K = beta * J
#     m = np.sinh(2 * K) ** (-2)
#     ff = lambda t: np.log(np.cosh(2 * K) ** 2 + np.sqrt(1 + m ** 2 - 2 * m * np.cos(2 * t)) / m)
#     f = (-1. / beta) * (np.log(2) / 2. + (1. / (2 * np.pi)) * quad(ff, 0, np.pi)[0])
#     return f

def _ising_exact_free_energy_2D_single(T, k=1., J=1.):
    beta = 1. / (k * T)
    K = 2. / (np.cosh(2*beta*J) / np.tanh(2*beta*J))
    ff = lambda t: np.log(0.5 * (1+np.sqrt(1-K**2 * np.sin(t)**2)))
    f = -np.log(2 * np.cosh(2*beta*J))/beta - 1./(beta * 2 * np.pi) * quad(ff, 0, np.pi)[0]
    return f


def _ising_exact_free_energy_2D(T, k=1., J=1.):
    # from http://www.lps.ens.fr/~krzakala/ISINGMODEL.pdf
    beta = 1. / (k * T)
    K = beta * J
    ff = lambda t1, t2: np.log(np.cosh(2 * K) ** 2 - np.sinh(2 * K) * (np.cos(t1) + np.cos(t2)))
    f = (-1. / beta) * (np.log(2) + nquad(ff, [[0, np.pi], [0, np.pi]])[0] / (2 * np.pi ** 2))
    return f


ising_exact_free_energy_2D = np.vectorize(_ising_exact_free_energy_2D)

ising_exact_free_energy_2D_single = np.vectorize(_ising_exact_free_energy_2D_single)


def ising_exact_entropy_2D(T, k=1, J=1, single=False):
    if single:
        s = (ising_exact_energy_2D(T, k, J) - ising_exact_free_energy_2D_single(T, k, J)) / T
    else:
        s = (ising_exact_energy_2D(T, k, J) - ising_exact_free_energy_2D(T, k, J)) / T
    return s
