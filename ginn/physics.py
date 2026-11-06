import numpy as np
import torch

torch.set_default_dtype(torch.float64)

C_KMS = 299792.458
C_AUYR = 63239.7263
M_SUN_GEOM_AU = 1.47662504 / 149597870.7
DEG = np.pi / 180.0
TWO_PI = 2.0 * np.pi

def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor(x), torch.tensor(w)


GL_PHI_X, GL_PHI_W = gauss_legendre(32)

GL_T_X, GL_T_W = gauss_legendre(48)


import numpy as np
import torch


class Prior:
    lo = -np.inf
    hi = np.inf

    def chi2(self, x):
        raise NotImplementedError

    def curvature(self):
        raise NotImplementedError

    def logpdf_np(self, x):
        raise NotImplementedError

    def inside(self, x):
        x = np.asarray(x)
        return (x >= self.lo) & (x <= self.hi)


    def project_(self, param):
        with torch.no_grad():
            param.clamp_(self.lo, self.hi)


class Uniform(Prior):
    def __init__(self, lo, hi):
        assert hi > lo
        self.lo, self.hi = float(lo), float(hi)

    def chi2(self, x):
        return torch.zeros((), dtype=torch.float64)

    def curvature(self):
        return 0.0

    def logpdf_np(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full_like(x, -np.inf)
        out[self.inside(x)] = -np.log(self.hi - self.lo)
        return out

    def __repr__(self):
        return f"Uniform({self.lo:g}, {self.hi:g})"


from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np
import torch


@dataclass
class HairParam:
    name: str
    latex: str
    prior: Prior
    init: float
    scale: float
    gr_value: float = 0.0


@dataclass
class Metric:
    name: str
    latex: str
    f: Callable
    g: Callable
    h: Callable
    hair: List[HairParam] = field(default_factory=list)
    precession: Callable = None
    label: str = ''

    @property
    def hair_names(self):
        return [hp.name for hp in self.hair]

    def hair_by_name(self, name):
        return next(hp for hp in self.hair if hp.name == name)


def gr_precession(M_AU, a, e):
    return 6.0 * np.pi * M_AU / (a * (1.0 - e ** 2))


def schw_f(r, par, M):
    return 1.0 - 2.0 * M / r

def schw_g(r, par, M):
    return 1.0 / schw_f(r, par, M)

def schw_h(r, par, M):
    return r ** 2

def schw_precession(M, a, e, par):
    return gr_precession(M, a, e)


def horn_f(r, par, M):
    q = par['q']
    return 1.0 - 2.0 * M / r + q * M / r * torch.log(r / (2.0 * M))

def horn_g(r, par, M):
    return 1.0 / horn_f(r, par, M)

def horn_h(r, par, M):
    return r ** 2

def horn_precession(M, a, e, par):
    q = par['q']
    s = torch.sqrt(1.0 - e ** 2) if torch.is_tensor(e) else np.sqrt(1.0 - e ** 2)
    return gr_precession(M, a, e) + np.pi * q * (1.0 - s) / e ** 2


def yuk_Phi(r, par, M):
    d = par['delta']
    lam = 10.0 ** par['log10_lambda']
    return -2.0 * M / ((1.0 + d) * r) * (1.0 + d * torch.exp(-r / lam))

def yuk_f(r, par, M):
    return 1.0 + yuk_Phi(r, par, M)

def yuk_g(r, par, M):
    return 1.0 - yuk_Phi(r, par, M)

def yuk_h(r, par, M):
    return r ** 2

def yuk_precession(M, a, e, par):
    d = par['delta']
    lam = 10.0 ** par['log10_lambda']
    s = torch.sqrt(1.0 - e ** 2) if torch.is_tensor(e) else np.sqrt(1.0 - e ** 2)
    return gr_precession(M, a, e) + np.pi * d / (1 + d) * s * (a / lam) ** 2


AU_M = 1.495978707e11

def _lam_au(par):
    return 10.0 ** par['log10_Lambda'] * AU_M ** 2

def sds_f(r, par, M):
    return 1.0 - 2.0 * M / r - _lam_au(par) * r ** 2 / 3.0

def sds_g(r, par, M):
    return 1.0 / sds_f(r, par, M)

def sds_h(r, par, M):
    return r ** 2

def sds_precession(M, a, e, par):
    s = torch.sqrt(1.0 - e ** 2) if torch.is_tensor(e) else np.sqrt(1.0 - e ** 2)
    return gr_precession(M, a, e) + np.pi * _lam_au(par) * a ** 3 * s / M


def rn_f(r, par, M):
    return 1.0 - 2.0 * M / r + par['Q'] ** 2 * M ** 2 / r ** 2

def rn_g(r, par, M):
    return 1.0 / rn_f(r, par, M)

def rn_h(r, par, M):
    return r ** 2

def rn_precession(M, a, e, par):
    return gr_precession(M, a, e) * (1.0 - par['Q'] ** 2 / 6.0)


def bardeen_f(r, par, M):
    return 1.0 - 2.0 * M * r ** 2 / (r ** 2 + (par['g_b'] * M) ** 2) ** 1.5

def bardeen_g(r, par, M):
    return 1.0 / bardeen_f(r, par, M)

def _jnw_b_gamma(par, M):
    s = torch.sqrt(1.0 + par['q_jnw'] ** 2)
    return 2.0 * M * s, 1.0 / s


def _jnw_r_of_R(R, par, M, n_iter=8):
    b, gam = _jnw_b_gamma(par, M)
    r = R + 0.5 * b - M
    for _ in range(n_iter):
        F = torch.log(r) + 0.5 * (1.0 - gam) * torch.log(1.0 - b / r) - torch.log(R)
        dF = (2.0 * r - b - 2.0 * M) / (2.0 * r * (r - b))
        r = r - F / dF
    return r, b, gam


def jnw_f(R, par, M):
    r, b, gam = _jnw_r_of_R(R, par, M)
    return (1.0 - b / r) ** gam

def jnw_g(R, par, M):
    r, b, gam = _jnw_r_of_R(R, par, M)
    dRdr = R * (2.0 * r - b - 2.0 * M) / (2.0 * r * (r - b))
    return 1.0 / ((1.0 - b / r) ** gam * dRdr ** 2)

def jnw_h(R, par, M):
    return R ** 2

def gr_only_precession(M, a, e, par):
    return gr_precession(M, a, e)


def ghs_f(r, par, M):
    return 1.0 - 2.0 * M / r

def ghs_g(r, par, M):
    return 1.0 / ghs_f(r, par, M)

def ghs_h(r, par, M):
    return r * (r - 2.0 * par['a_dil'] * M)

def ghs_precession(M, a, e, par):
    return gr_precession(M, a, e) * (1.0 - par['a_dil'] / 3.0)


def make_metrics(horndeski_prior=None, yukawa_delta_prior=None, yukawa_lambda_prior=None,
                 sds_prior=None, dilaton_prior=None, rn_prior=None,
                 jnw_prior=None, bardeen_prior=None):

    horndeski_prior = horndeski_prior or Uniform(-0.05, 0.05)
    yukawa_delta_prior = yukawa_delta_prior or Uniform(-0.5, 0.5)
    yukawa_lambda_prior = yukawa_lambda_prior or Uniform(1.0, 6.0)
    sds_prior = sds_prior or Uniform(-70.0, -20.0)
    rn_prior = rn_prior or Uniform(-50.0, 50.0)
    dilaton_prior = dilaton_prior or Uniform(-50.0, 50.0)
    jnw_prior = jnw_prior or Uniform(-600.0, 600.0)
    bardeen_prior = bardeen_prior or Uniform(-200.0, 200.0)
    return {
        'schwarzschild': Metric(
            name='schwarzschild',
            label='Schwarzschild',
            latex=r'$f=1-2M/r$',
            f=schw_f, g=schw_g, h=schw_h, hair=[],
            precession=schw_precession),
        'hairy_horndeski': Metric(
            name='hairy_horndeski',
            label='Horndeski',
            latex=r'$f=1-2M/r+q_H\,(M/r)\ln(r/2M)$',
            f=horn_f, g=horn_g, h=horn_h,
            hair=[HairParam(name='q', latex=r'$q_H$', prior=horndeski_prior,
                            init=1e-3, scale=1e-3, gr_value=0.0)],
            precession=horn_precession),
        'yukawa': Metric(
            name='yukawa',
            label='Yukawa',
            latex=r'$\Phi=-\frac{2M}{(1+\delta)r}(1+\delta e^{-r/\lambda})$',
            f=yuk_f, g=yuk_g, h=yuk_h,
            hair=[HairParam(name='delta', latex=r'$\delta$', prior=yukawa_delta_prior,
                            init=0.01, scale=0.01, gr_value=0.0),
                  HairParam(name='log10_lambda', latex=r'$\log_{10}(\lambda/{\rm AU})$',
                            prior=yukawa_lambda_prior, init=2.5, scale=0.1, gr_value=np.inf)],
            precession=yuk_precession),
        'sds': Metric(
            name='sds',
            label='SdS',
            latex=r'$f=1-2M/r-\Lambda r^2/3$',
            f=sds_f, g=sds_g, h=sds_h,
            hair=[HairParam(name='log10_Lambda', latex=r'$\log_{10}(\Lambda/{\rm m^{-2}})$',
                            prior=sds_prior, init=-60.0, scale=0.5, gr_value=-np.inf)],
            precession=sds_precession),
        'reissner_nordstrom': Metric(
            name='reissner_nordstrom',
            label='RN',
            latex=r'$f=1-2M/r+q_{\rm RN}^2M^2/r^2$',
            f=rn_f, g=rn_g, h=rn_h,
            hair=[HairParam(name='Q', latex=r'$q_{\rm RN}$', prior=rn_prior, init=1.0, scale=1.0,
                            gr_value=0.0)],
            precession=rn_precession),
        'bardeen': Metric(
            name='bardeen',
            label='Bardeen',
            latex=r'$f=1-2Mr^2/(r^2+q_B^2M^2)^{3/2}$',
            f=bardeen_f, g=bardeen_g, h=rn_h,
            hair=[HairParam(name='g_b', latex=r'$q_B$', prior=bardeen_prior, init=10.0, scale=5.0,
                            gr_value=0.0)],
            precession=gr_only_precession),
        'jnw': Metric(
            name='jnw',
            label='JNW',
            latex=r'$f=(1-b/r)^\gamma,\ b=2M\sqrt{1+q_{\rm JNW}^2}$',
            f=jnw_f, g=jnw_g, h=jnw_h,
            hair=[HairParam(name='q_jnw', latex=r'$q_{\rm JNW}$', prior=jnw_prior, init=20.0, scale=5.0,
                            gr_value=0.0)],
            precession=gr_only_precession),
        'ghs_dilaton': Metric(
            name='ghs_dilaton',
            label='GHS dilaton',
            latex=r'$h=r(r-2a_{\rm dil}M)$',
            f=ghs_f, g=ghs_g, h=ghs_h,
            hair=[HairParam(name='a_dil', latex=r'$b/M$', prior=dilaton_prior,
                            init=0.5, scale=1.0, gr_value=0.0)],
            precession=ghs_precession),
    }


METRICS = make_metrics()


import numpy as np
import torch


def kepler_E(M_anom, e, n_iter=12):
    n_orb = torch.floor((M_anom + np.pi) / TWO_PI)
    Mr = M_anom - n_orb * TWO_PI
    E = Mr + 0.85 * e * torch.sign(Mr)
    for _ in range(n_iter):
        E = E - (E - e * torch.sin(E) - Mr) / (1.0 - e * torch.cos(E))
    return E + n_orb * TWO_PI


def chi_from_E(E, e):
    n_orb = torch.floor((E + np.pi) / TWO_PI)
    Er = E - n_orb * TWO_PI
    chi = 2.0 * torch.atan2(torch.sqrt(1.0 + e) * torch.sin(Er / 2.0),
                            torch.sqrt(1.0 - e) * torch.cos(Er / 2.0))
    return chi + n_orb * TWO_PI


def E_from_chi(chi, e):
    n_orb = torch.floor((chi + np.pi) / TWO_PI)
    cr = chi - n_orb * TWO_PI
    E = 2.0 * torch.atan2(torch.sqrt(1.0 - e) * torch.sin(cr / 2.0),
                          torch.sqrt(1.0 + e) * torch.cos(cr / 2.0))
    return E + n_orb * TWO_PI


def kepler_true_anomaly(M_anom, e):
    return chi_from_E(kepler_E(M_anom, e), e)


def dchi_dt_kepler(chi, e, n):
    return n * (1.0 + e * torch.cos(chi)) ** 2 / (1.0 - e ** 2) ** 1.5


def EL_solve(metric, a, e, par, M):
    r_p, r_a = a * (1 - e), a * (1 + e)
    f_a, f_p = metric.f(r_a, par, M), metric.f(r_p, par, M)
    h_a, h_p = metric.h(r_a, par, M), metric.h(r_p, par, M)
    A11, A12 = 1 / f_a, -1 / h_a
    A21, A22 = 1 / f_p, -1 / h_p
    det = A11 * A22 - A12 * A21
    E2 = (A22 - A12) / det
    L2 = (A11 - A21) / det
    return E2, L2


def _safe_chi(chi_w, floor=1e-3):
    d0 = torch.minimum(chi_w, TWO_PI - chi_w)
    chi_w = torch.where((d0 < floor) & (chi_w < np.pi), torch.full_like(chi_w, floor), chi_w)
    chi_w = torch.where((d0 < floor) & (chi_w >= np.pi), torch.full_like(chi_w, TWO_PI - floor), chi_w)
    d_pi = torch.abs(chi_w - np.pi)
    chi_w = torch.where((d_pi < floor) & (chi_w < np.pi), torch.full_like(chi_w, np.pi - floor), chi_w)
    chi_w = torch.where((d_pi < floor) & (chi_w >= np.pi), torch.full_like(chi_w, np.pi + floor), chi_w)
    return chi_w


def geodesic_rates(chi, a, e, metric, E2, L2, par, M):

    p = a * (1 - e ** 2)
    chi_w = _safe_chi(torch.remainder(chi, TWO_PI))
    c, s = torch.cos(chi_w), torch.sin(chi_w)
    r = p / (1 + e * c)
    r_p, r_a = a * (1 - e), a * (1 + e)
    r_ref = torch.where(c >= 0, r_p, r_a)
    f_ref, h_ref = metric.f(r_ref, par, M), metric.h(r_ref, par, M)
    f_r, h_r = metric.f(r, par, M), metric.h(r, par, M)

    bracket = E2 * (1 / f_r - 1 / f_ref) - L2 * (1 / h_r - 1 / h_ref)
    rdot2 = torch.clamp(bracket, min=0.0) / metric.g(r, par, M)
    drdchi = p * e * s / (1 + e * c) ** 2
    dchidtau = torch.sqrt(torch.clamp(rdot2 / drdchi ** 2, min=1e-300))
    dtdtau = torch.sqrt(E2) / f_r
    dphidtau = torch.sqrt(L2) / h_r
    return dchidtau / dtdtau * C_AUYR, dphidtau / dtdtau * C_AUYR, r


def _dphi_dchi(chi, a, e, metric, E2, L2, par, M):
    dchi, dphi, _ = geodesic_rates(chi, a, e, metric, E2, L2, par, M)
    return dphi / dchi


def _dt_dE(E, a, e, metric, E2, L2, par, M):
    chi = chi_from_E(E, e)
    dchi, _, _ = geodesic_rates(chi, a, e, metric, E2, L2, par, M)
    dchi_dE = torch.sqrt(1 - e ** 2) / (1 - e * torch.cos(E))
    return dchi_dE / dchi


def _gl_integral(fun, upper, xg, wg):
    upper = upper.reshape(-1, 1)
    half = 0.5 * upper
    nodes = half + half * xg.view(1, -1)
    vals = fun(nodes.reshape(-1)).reshape(nodes.shape)
    return (half.squeeze(-1)) * (vals * wg.view(1, -1)).sum(dim=1)


def orbit_constants(metric, a, e, par, M):
    E2, L2 = EL_solve(metric, a, e, par, M)
    two_pi = torch.tensor([TWO_PI])
    P_r = _gl_integral(lambda x: _dt_dE(x, a, e, metric, E2, L2, par, M), two_pi, GL_T_X, GL_T_W)[0]
    dphi = _gl_integral(lambda x: _dphi_dchi(x, a, e, metric, E2, L2, par, M), two_pi,
                        GL_PHI_X, GL_PHI_W)[0] - TWO_PI
    return E2, L2, P_r, dphi


def phi_of_chi(chi, a, e, metric, E2, L2, par, M, dphi_orbit):
    chi = chi.reshape(-1)
    n_orb = torch.floor(chi / TWO_PI)
    chi_w = chi - n_orb * TWO_PI
    part = _gl_integral(lambda x: _dphi_dchi(x, a, e, metric, E2, L2, par, M), chi_w,
                        GL_PHI_X, GL_PHI_W)
    return n_orb * (TWO_PI + dphi_orbit) + part


def time_of_E(E, a, e, metric, E2, L2, par, M, P_r):
    E = E.reshape(-1)
    n_orb = torch.floor(E / TWO_PI)
    E_w = E - n_orb * TWO_PI
    part = _gl_integral(lambda x: _dt_dE(x, a, e, metric, E2, L2, par, M), E_w, GL_T_X, GL_T_W)
    return n_orb * P_r + part


def exact_chi_of_t(t, t0, a, e, metric, E2, L2, par, M, P_r, n_newton=6):

    t = t.reshape(-1)
    tau = t - t0
    n_r = TWO_PI / P_r
    with torch.no_grad():
        Ek = kepler_E(n_r.detach() * tau.detach(), e.detach())
        for _ in range(n_newton):
            T = time_of_E(Ek, a.detach(), e.detach(), metric, E2.detach(), L2.detach(),
                          {k: v.detach() if torch.is_tensor(v) else v for k, v in par.items()},
                          M.detach(), P_r.detach())
            dT = _dt_dE(Ek, a.detach(), e.detach(), metric, E2.detach(), L2.detach(),
                        {k: v.detach() if torch.is_tensor(v) else v for k, v in par.items()},
                        M.detach())
            Ek = Ek - (T - tau.detach()) / dT
    T = time_of_E(Ek, a, e, metric, E2, L2, par, M, P_r)
    dT = _dt_dE(Ek, a, e, metric, E2, L2, par, M)
    E = Ek - (T - tau) / dT
    return chi_from_E(E, e)
