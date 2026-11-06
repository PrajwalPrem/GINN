from .physics import C_KMS, C_AUYR, M_SUN_GEOM_AU, DEG, TWO_PI
from . import physics as G

import numpy as np
import torch


class S2Data:
    def __init__(self, t_pos, ra, ra_e, dec, dec_e, t_rv, vlos, vlos_e, label='S2'):
        T = lambda x: torch.as_tensor(np.asarray(x, dtype=float)).reshape(-1)
        self.t_pos, self.ra, self.ra_e, self.dec, self.dec_e = map(T, (t_pos, ra, ra_e, dec, dec_e))
        self.t_rv, self.vlos, self.vlos_e = map(T, (t_rv, vlos, vlos_e))
        self.label = label

    @classmethod
    def from_csv(cls, pos_file, rv_file):
        pos = np.genfromtxt(pos_file, delimiter=',', encoding='utf-8-sig')
        rv = np.genfromtxt(rv_file, delimiter=',', encoding='utf-8-sig')
        return cls(pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3], pos[:, 4],
                   rv[:, 0], rv[:, 1], rv[:, 2], label='Gillessen+2017')

    @property
    def n_points(self):
        return 2 * self.t_pos.numel() + self.t_rv.numel()

    @property
    def t_span(self):
        t = torch.cat([self.t_pos, self.t_rv])
        return float(t.min()), float(t.max())

    @property
    def t_ref(self):
        return float(self.t_pos.mean())

    def replace_values(self, ra=None, dec=None, vlos=None, label=None):
        return S2Data(self.t_pos, self.ra if ra is None else ra, self.ra_e,
                      self.dec if dec is None else dec, self.dec_e,
                      self.t_rv, self.vlos if vlos is None else vlos, self.vlos_e,
                      label=label or self.label)

    def summary(self):
        t0, t1 = self.t_span
        return (f"{self.label}: {self.t_pos.numel()} astrometric epochs, {self.t_rv.numel()} RVs, "
                f"{t0:.2f}-{t1:.2f};  N_data = {self.n_points}")


import numpy as np
from scipy.optimize import least_squares, minimize


def _kepler_E_np(M, e, n_iter=30):
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    E = M + 0.85 * e * np.sign(M)
    for _ in range(n_iter):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def _ti_linfit(t, ra, ra_e, dec, dec_e, P, e, tp):
    E = _kepler_E_np(2 * np.pi * (t - tp) / P, e)
    D = np.stack([np.cos(E) - e, np.sqrt(max(1 - e ** 2, 1e-8)) * np.sin(E)], axis=1)

    def wlsq(y, err):
        coef, *_ = np.linalg.lstsq(D / err[:, None], y / err, rcond=None)
        return coef, np.sum(((D @ coef - y) / err) ** 2)
    (A, F), c1 = wlsq(dec, dec_e)
    (B, G), c2 = wlsq(ra, ra_e)
    return A, B, F, G, c1 + c2


def _ti_angles(i, Om, om):
    i, Om, om = np.deg2rad([i, Om, om])
    A = np.cos(Om) * np.cos(om) - np.sin(Om) * np.sin(om) * np.cos(i)
    B = np.sin(Om) * np.cos(om) + np.cos(Om) * np.sin(om) * np.cos(i)
    F = -np.cos(Om) * np.sin(om) - np.sin(Om) * np.cos(om) * np.cos(i)
    G = -np.sin(Om) * np.sin(om) + np.cos(Om) * np.cos(om) * np.cos(i)
    return A, B, F, G


def keplerian_prefit(data, P_range=(14., 18.), e_range=(0.8, 0.93), D_nominal=8.3,
                     grid=(24, 20, 40), verbose=True):
    t = data.t_pos.numpy(); ra, ra_e = data.ra.numpy(), data.ra_e.numpy()
    dec, dec_e = data.dec.numpy(), data.dec_e.numpy()
    tp_range = (t.min(), t.min() + P_range[1])

    best = None
    for P in np.linspace(*P_range, grid[0]):
        for e in np.linspace(*e_range, grid[1]):
            for tp in np.linspace(*tp_range, grid[2]):
                c = _ti_linfit(t, ra, ra_e, dec, dec_e, P, e, tp)[-1]
                if best is None or c < best[0]:
                    best = (c, P, e, tp)

    def obj(x):
        P, e, tp = x
        if not (0.1 < e < 0.97 and P > 1):
            return 1e12
        return _ti_linfit(t, ra, ra_e, dec, dec_e, P, e, tp)[-1]
    res = minimize(obj, best[1:], method='Nelder-Mead',
                   options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
    P, e, tp = res.x
    A, B, F, G, chi2 = _ti_linfit(t, ra, ra_e, dec, dec_e, P, e, tp)

    def resid(x):
        K, i, Om, om = x
        a_, b_, f_, g_ = _ti_angles(i, Om, om)
        return np.array([K * a_ - A, K * b_ - B, K * f_ - F, K * g_ - G])
    rng = np.random.default_rng(0)
    K0 = np.sqrt(A ** 2 + B ** 2 + F ** 2 + G ** 2) / np.sqrt(2)
    sol_best = None
    for _ in range(60):
        x0 = [K0 * rng.uniform(0.7, 1.3), rng.uniform(0, 180), rng.uniform(0, 360), rng.uniform(0, 360)]
        sol = least_squares(resid, x0, bounds=([0, 0, 0, 0], [10 * K0, 180, 360, 360]))
        if sol_best is None or sol.cost < sol_best.cost:
            sol_best = sol
    a_mas, i_deg, Om, om = sol_best.x

    trv, vr = data.t_rv.numpy(), data.vlos.numpy()
    Erv = _kepler_E_np(2 * np.pi * (trv - tp) / P, e)
    dE = (2 * np.pi / P) / (1 - e * np.cos(Erv))
    vx, vy = -np.sin(Erv) * dE, np.sqrt(1 - e ** 2) * np.cos(Erv) * dE
    ir, omr = np.deg2rad(i_deg), np.deg2rad(om)
    shape = np.sin(ir) * (np.sin(omr) * vx + np.cos(omr) * vy)
    if np.corrcoef(shape, vr - vr.mean())[0, 1] < 0:
        Om, om = np.mod(Om + 180, 360), np.mod(om + 180, 360)

    a_AU = a_mas / 1000.0 * D_nominal * 1000.0
    M = a_AU ** 3 / P ** 2
    init = dict(M=M, a=a_AU, e=e, inc=i_deg, Omega=Om, omega=om, t0=tp, D=D_nominal)
    if verbose:
        print(f"Keplerian pre-fit (astrometry only): chi2 = {chi2:.1f}")
        print("  " + ", ".join(f"{k}={v:.5g}" for k, v in init.items()) + f", P={P:.4f} yr")
    return init


import numpy as np
import torch
import torch.nn as nn


BASE_SPECS = [
    ('logM', 0.02),
    ('loga', 0.01),
    ('e_logit', 0.01),
    ('inc', 0.005),
    ('Omega', 0.005),
    ('omega', 0.005),
    ('t0', 0.01),
    ('logD', 0.01),
    ('vz0', 5.0),
    ('x0', 0.5),
    ('y0', 0.5),
    ('vx0', 0.05),
    ('vy0', 0.05),
]
BASE_NAMES = [n for n, _ in BASE_SPECS]


class CorrectionNet(nn.Module):

    def __init__(self, hidden=(32, 32), out_scale=1e-3):
        super().__init__()
        layers, d = [], 2
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.Tanh()]
            d = hdim
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.out_scale = out_scale
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def weights(self):
        return list(self.net.parameters())

    def value_and_slope(self, chi, params=None):

        p = self.weights() if params is None else params
        h = torch.stack([torch.cos(chi), torch.sin(chi)], dim=-1)
        dh = torch.stack([-torch.sin(chi), torch.cos(chi)], dim=-1)
        for k in range(0, len(p) - 2, 2):
            W, b = p[k], p[k + 1]
            h = torch.tanh(h @ W.T + b)
            dh = (1 - h ** 2) * (dh @ W.T)
        W, b = p[-2], p[-1]
        y = (h @ W.T + b).squeeze(-1)
        dy = (dh @ W.T).squeeze(-1)
        return self.out_scale * y, self.out_scale * dy

    def forward(self, chi, params=None):
        return self.value_and_slope(chi, params)[0]


class OrbitModel(nn.Module):
    def __init__(self, metric, init, t_ref, timing='ginn', hidden=(32, 32),
                 nn_out_scale=1e-3, roemer=True, hair_init=None):
        super().__init__()
        self.metric = metric
        self.t_ref = float(t_ref)
        self.timing = timing
        self.roemer = roemer
        hair_init = hair_init or {}

        centers = [np.log(init['M']), np.log(init['a']), np.log(init['e'] / (1 - init['e'])),
                   init['inc'] * DEG, init['Omega'] * DEG, init['omega'] * DEG, init['t0'],
                   np.log(init['D']), init.get('vz0', 0.0), init.get('x0', 0.0),
                   init.get('y0', 0.0), init.get('vx0', 0.0), init.get('vy0', 0.0)]
        scales = [s for _, s in BASE_SPECS]
        for hp in metric.hair:
            centers.append(float(hair_init.get(hp.name, hp.init)))
            scales.append(hp.scale)
        self.names = BASE_NAMES + metric.hair_names
        self.register_buffer('centers', torch.tensor(centers))
        self.register_buffer('scales', torch.tensor(scales))
        self.register_buffer('free', torch.ones(len(centers)))
        self.register_buffer('W', torch.eye(len(centers)))
        self.u = nn.Parameter(torch.zeros(len(centers)))
        self.net = CorrectionNet(hidden, nn_out_scale)
        n_theta = sum(p.numel() for p in self.net.parameters())
        self.register_buffer('D', torch.zeros(n_theta, len(centers)))
        self.register_buffer('u_ref', torch.zeros(len(centers)))

    def index(self, name):
        return self.names.index(name)

    def internal(self, u=None):
        u = self.u if u is None else u
        return self.centers + self.scales * (self.W @ u)

    def physical(self, u=None):
        x = self.internal(u)
        d = {n: x[i] for i, n in enumerate(self.names)}
        P = dict(M=torch.exp(d['logM']), a=torch.exp(d['loga']), e=torch.sigmoid(d['e_logit']),
                 inc=d['inc'], Omega=d['Omega'], omega=d['omega'], t0=d['t0'],
                 D=torch.exp(d['logD']), vz0=d['vz0'], x0=d['x0'], y0=d['y0'],
                 vx0=d['vx0'], vy0=d['vy0'])
        P['M_AU'] = P['M'] * M_SUN_GEOM_AU
        P['par'] = {n: d[n] for n in self.metric.hair_names}
        return P


    def net_params(self, u=None):

        base = self.net.weights()
        u = self.u if u is None else u
        if not self._coupled:
            return base
        flat = self.D @ (u - self.u_ref)
        out, i = [], 0
        for p in base:
            n = p.numel()
            out.append(p + flat[i:i + n].view_as(p))
            i += n
        return out

    @property
    def _coupled(self):
        return bool(self.D.abs().max() > 0)

    def fold_coupling(self):

        if self._coupled:
            with torch.no_grad():
                for p, q in zip(self.net.weights(), self.net_params()):
                    p.copy_(q)
                self.D.zero_()
                self.u_ref.copy_(self.u)

    def rebase(self, W=None):

        self.fold_coupling()
        with torch.no_grad():
            self.centers.copy_(self.internal())
            self.u.zero_()
            self.u_ref.zero_()
            if W is not None:
                self.W.copy_(W)

    def _set_internal(self, x):
        with torch.no_grad():
            y = ((x - self.centers) / self.scales).unsqueeze(-1)
            self.u.copy_(torch.linalg.solve_triangular(self.W, y, upper=False).squeeze(-1))

    def set_value(self, name, value):
        x = self.internal().detach().clone()
        x[self.index(name)] = value
        self._set_internal(x)

    def fix(self, name, fixed=True):

        k = self.index(name)
        self.rebase()
        with torch.no_grad():
            self.W[k, :] = 0.0
            self.W[:, k] = 0.0
            self.W[k, k] = 1.0
        self.free[k] = 0.0 if fixed else 1.0

    def project_to_prior_(self):

        if not self.metric.hair:
            return
        x = self.internal().detach().clone()
        changed = False
        for hp in self.metric.hair:
            k = self.index(hp.name)
            v = float(x[k])
            if v < hp.prior.lo or v > hp.prior.hi:
                x[k] = min(max(v, hp.prior.lo), hp.prior.hi)
                changed = True
        if changed:
            self._set_internal(x)

    def summary(self):
        P = self.physical()
        s = (f"M = {P['M'].item()/1e6:.4f}e6 Msun | a = {P['a'].item():.2f} AU | e = {P['e'].item():.5f} | "
             f"i = {P['inc'].item()/DEG:.3f} | Om = {P['Omega'].item()/DEG:.3f} | om = {P['omega'].item()/DEG:.3f} | "
             f"t_p = {P['t0'].item():.4f} | D = {P['D'].item():.4f} kpc | vz0 = {P['vz0'].item():.2f} km/s")
        s += (f"\n   frame: x0 = {P['x0'].item():+.3f} mas | y0 = {P['y0'].item():+.3f} mas | "
              f"vx0 = {P['vx0'].item():+.4f} mas/yr | vy0 = {P['vy0'].item():+.4f} mas/yr")
        for n, v in P['par'].items():
            s += f" | {n} = {v.item():.4g}"
        return s

    def orbit_constants(self, P):
        return G.orbit_constants(self.metric, P['a'], P['e'], P['par'], P['M_AU'])

    def nn_delta(self, chi_K, u=None):
        p = self.net_params(u)
        zero = torch.zeros(1, dtype=chi_K.dtype)
        return self.net(chi_K, p) - self.net(zero, p)

    def chi_of_t(self, t, P, C, timing=None, u=None):
        timing = timing or self.timing
        E2, L2, P_r, _ = C
        if timing == 'exact':
            return G.exact_chi_of_t(t, P['t0'], P['a'], P['e'], self.metric, E2, L2,
                                    P['par'], P['M_AU'], P_r)
        n_r = TWO_PI / P_r
        chi_K = G.kepler_true_anomaly(n_r * (t - P['t0']), P['e'])
        if timing == 'kepler':
            return chi_K
        return chi_K + self.nn_delta(chi_K, u)

    def _orbit_state(self, t, P, C, timing, u=None):
        E2, L2, P_r, dphi = C
        a, e, M, par = P['a'], P['e'], P['M_AU'], P['par']
        chi = self.chi_of_t(t, P, C, timing, u)
        phi = G.phi_of_chi(chi, a, e, self.metric, E2, L2, par, M, dphi)
        chi_w = torch.remainder(chi, TWO_PI)
        p = a * (1 - e ** 2)
        r = p / (1 + e * torch.cos(chi_w))
        f, g, h = self.metric.f(r, par, M), self.metric.g(r, par, M), self.metric.h(r, par, M)
        E, L = torch.sqrt(E2), torch.sqrt(L2)
        R = torch.sqrt(h)
        rad = torch.clamp(E2 / f - 1 - L2 / h, min=1e-30)
        sgn = torch.where(torch.sin(chi_w) >= 0, 1.0, -1.0)
        dtdtau = E / f
        v_r = sgn * torch.sqrt(rad / g) / dtdtau
        v_p = R * (L / h) / dtdtau
        x, y = R * torch.cos(phi), R * torch.sin(phi)
        vx = v_r * torch.cos(phi) - v_p * torch.sin(phi)
        vy = v_r * torch.sin(phi) + v_p * torch.cos(phi)

        cO, sO = torch.cos(P['Omega']), torch.sin(P['Omega'])
        co, so = torch.cos(P['omega']), torch.sin(P['omega'])
        ci, si = torch.cos(P['inc']), torch.sin(P['inc'])
        A = cO * co - sO * so * ci
        B = sO * co + cO * so * ci
        F = -cO * so - sO * co * ci
        Gc = -sO * so + cO * co * ci
        Cz, Hz = so * si, co * si
        dec = A * x + F * y
        ra = B * x + Gc * y
        Z = Cz * x + Hz * y
        vz = Cz * vx + Hz * vy
        return ra, dec, Z, vz, dtdtau

    def forward(self, t, timing=None, u=None):
        timing = timing or self.timing
        t = t.reshape(-1)
        P = self.physical(u)
        C = self.orbit_constants(P)
        t_em = t
        if self.roemer:
            _, _, Z, _, _ = self._orbit_state(t, P, C, timing, u)
            t_em = t - Z / C_AUYR
        ra, dec, Z, vz, dtdtau = self._orbit_state(t_em, P, C, timing, u)
        D_pc = 1000.0 * P['D']
        ra_mas = 1000.0 * ra / D_pc + P['x0'] + P['vx0'] * (t - self.t_ref)
        dec_mas = 1000.0 * dec / D_pc + P['y0'] + P['vy0'] * (t - self.t_ref)
        vlos = C_KMS * (dtdtau * (1.0 + vz) - 1.0) + P['vz0']
        return ra_mas, dec_mas, vlos

    def predict_data(self, data, timing=None, u=None):
        n = data.t_pos.numel()
        ra, dec, v = self.forward(torch.cat([data.t_pos, data.t_rv]), timing, u)
        return ra[:n], dec[:n], v[n:]

    def residuals(self, data, timing=None, u=None):
        ra, dec, v = self.predict_data(data, timing, u)
        return torch.cat([(ra - data.ra) / data.ra_e, (dec - data.dec) / data.dec_e,
                          (v - data.vlos) / data.vlos_e])

    def collocation_E(self, n, jitter=None):
        shift = 0.5 if jitter is None else jitter
        return (torch.arange(n, dtype=torch.float64) + shift) / n * TWO_PI


    def physics_residual(self, E_c, u=None):
        P = self.physical(u)
        C = self.orbit_constants(P)
        E2, L2, P_r, _ = C
        n_r = TWO_PI / P_r
        e = P['e']
        chi_K = G.chi_from_E(E_c, e)
        p = self.net_params(u)
        y, dy = self.net.value_and_slope(chi_K, p)
        chi = chi_K + y - self.net(torch.zeros(1, dtype=E_c.dtype), p)
        dchi_dt = (1.0 + dy) * G.dchi_dt_kepler(chi_K, e, n_r)
        target, _, _ = G.geodesic_rates(chi, P['a'], e, self.metric, E2, L2, P['par'], P['M_AU'])
        return (dchi_dt - target) / target.detach()

    def precession(self, u=None):
        P = self.physical(u)
        _, _, _, dphi = self.orbit_constants(P)
        gr = 6 * np.pi * P['M_AU'] / (P['a'] * (1 - P['e'] ** 2))
        return dphi, dphi / gr


from dataclasses import dataclass
import math
import numpy as np
import torch
from tqdm.auto import tqdm


@dataclass
class LossConfig:
    eps_phys: float = 1e-4
    n_colloc: int = 256
    sigma_w: float = 10.0
    external_precession: tuple = None


def nn_weight_chi2(model, sigma_w):
    return sum((p ** 2).sum() for p in model.net.parameters()) / sigma_w ** 2


def objective(model, data, cfg, timing=None, jitter=None, eps=None):

    eps = cfg.eps_phys if eps is None else eps
    timing = timing or model.timing
    res = model.residuals(data, timing)
    terms = {'data': (res ** 2).sum()}
    if timing == 'ginn':
        E_c = model.collocation_E(cfg.n_colloc, jitter)
        R = model.physics_residual(E_c)
        terms['phys'] = ((R / eps) ** 2).sum()
        if cfg.sigma_w:
            terms['nn_prior'] = nn_weight_chi2(model, cfg.sigma_w)
    if model.metric.hair:
        P = model.physical()
        terms['hair_prior'] = sum(hp.prior.chi2(P['par'][hp.name]) for hp in model.metric.hair)
    if cfg.external_precession is not None:
        f_obs, sig = cfg.external_precession
        _, f_sp = model.precession()
        terms['precession'] = ((f_sp - f_obs) / sig) ** 2
    total = sum(terms.values())
    return total, terms


def jacobian_rows(outputs, params):

    rows = []
    for i in range(outputs.numel()):
        g = torch.autograd.grad(outputs[i], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                               for gi, p in zip(g, params)]))
    return torch.stack(rows)


def jacobian_u(model, data, timing='kepler', cfg=None):
    u0 = model.u.detach().clone()

    def res(u):
        r = model.residuals(data, timing, u)
        if cfg is not None and cfg.external_precession is not None:
            f_obs, sig = cfg.external_precession
            r = torch.cat([r, ((model.precession(u)[1] - f_obs) / sig).reshape(1)])
        return r
    return torch.func.jacfwd(res)(u0) * model.free


def safe_inv(H, rel=1e-12):
    H = 0.5 * (H + H.T)
    ev, V = torch.linalg.eigh(H)
    ev = torch.clamp(ev, min=float(ev.max()) * rel + 1e-300)
    Hi = V @ torch.diag(1.0 / ev) @ V.T
    return 0.5 * (Hi + Hi.T)

def compute_preconditioner(model, data, cfg, ridge=1e-2):

    model.fold_coupling()
    J = jacobian_u(model, data, 'kepler', cfg)
    H = J.T @ J
    free = model.free.bool()
    for hp in model.metric.hair:
        k = model.index(hp.name)
        s = (model.scales[k] * model.W[k, k]).item()
        H[k, k] += hp.prior.curvature() * s ** 2
    H = H + ridge * torch.eye(len(H))
    Hf = H[free][:, free]
    Lf = torch.linalg.cholesky(safe_inv(Hf))
    L = torch.eye(len(H))
    idx = torch.where(free)[0]
    L[idx.unsqueeze(1), idx.unsqueeze(0)] = Lf
    model.rebase(model.W @ L)


def compute_coupling(model, cfg, damping=1e-4):

    model.fold_coupling()
    E_c = model.collocation_E(cfg.n_colloc)
    R = model.physics_residual(E_c) / cfg.eps_phys
    theta = model.net.weights()
    J = jacobian_rows(R, [model.u] + theta)
    n_u = model.u.numel()
    B, A = J[:, :n_u] * model.free, J[:, n_u:]
    lam = damping * torch.linalg.matrix_norm(A, ord=2) ** 2
    if cfg.sigma_w:
        lam = lam + 1.0 / cfg.sigma_w ** 2
    K = A @ A.T + lam * torch.eye(A.shape[0])
    D = -A.T @ torch.linalg.solve(K, B)
    with torch.no_grad():
        model.D.copy_(D)
        model.u_ref.copy_(model.u)


def train(model, data, cfg, n_steps=4000, lr_u=0.1, lr_nn=5e-4, lr_floor=0.01,
          precondition=(0.0,), eval_every=25, progress=True,
          desc=None):

    timing = model.timing
    pre_steps = {int(f * n_steps) for f in precondition}
    groups = [{'params': [model.u], 'lr': lr_u}]
    if timing == 'ginn':
        groups.append({'params': list(model.net.parameters()), 'lr': lr_nn})
    opt = torch.optim.Adam(groups)
    cosine = lambda s: lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * min(s, n_steps) / n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine)

    hist = {k: [] for k in ['total', 'data', 'phys', 'phys_rms', 'eps', 'hair']}
    PARAM_KEYS = ['M', 'a', 'e', 'inc', 'Omega', 'omega', 't0', 'D', 'vz0', 'x0', 'y0', 'vx0', 'vy0']
    hist['params'] = {k: [] for k in PARAM_KEYS + model.metric.hair_names}
    hist['precondition_steps'] = sorted(pre_steps)
    best = (np.inf, None)
    gen = torch.Generator().manual_seed(1234)

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def evaluate():
        tot, _ = objective(model, data, cfg)
        return tot.item()

    it = tqdm(range(n_steps), desc=desc or f'{model.metric.name}/{timing}', disable=not progress)
    for step in it:
        if step in pre_steps:
            compute_preconditioner(model, data, cfg)
            for p in [model.u] + list(model.net.parameters()):
                opt.state[p] = {}
            best = (np.inf, None)
        opt.zero_grad()
        jitter = torch.rand(1, generator=gen).item()
        eps = cfg.eps_phys if timing == 'ginn' else None
        total, terms = objective(model, data, cfg, jitter=jitter, eps=eps)
        if not torch.isfinite(total):
            if best[1] is not None:
                model.load_state_dict(best[1])
            for g in opt.param_groups:
                g['lr'] *= 0.3
            continue
        total.backward()
        model.u.grad.mul_(model.free)
        opt.step()
        sched.step()
        model.project_to_prior_()

        hist['total'].append(total.item())
        hist['data'].append(terms['data'].item())
        hist['phys'].append(terms['phys'].item() if 'phys' in terms else np.nan)
        hist['hair'].append([v.item() for v in model.physical()['par'].values()])
        if 'phys' in terms:
            hist['phys_rms'].append(eps * np.sqrt(terms['phys'].item() / cfg.n_colloc))
            hist['eps'].append(eps)
        else:
            hist['phys_rms'].append(np.nan)
            hist['eps'].append(np.nan)
        with torch.no_grad():
            P = model.physical()
            for k in PARAM_KEYS:
                hist['params'][k].append(P[k].item())
            for k, v in P['par'].items():
                hist['params'][k].append(v.item())
        if (step % eval_every == 0 or step == n_steps - 1) :
            val = evaluate()
            if val < best[0]:
                best = (val, snapshot())
            it.set_postfix(chi2_data=f"{terms['data'].item():.2f}", best=f"{best[0]:.2f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    hist['best_objective'] = best[0]
    return hist


import torch


def fit(metric, data, init, timing='ginn', cfg=None, n_steps=4000, seed=0, hair_init=None,
        fixed=(), progress=True, hidden=(32, 32), nn_out_scale=1e-3, roemer=True, **train_kw):

    torch.manual_seed(seed)
    model = OrbitModel(metric, init, data.t_ref, timing=timing, hair_init=hair_init,
                       hidden=hidden, nn_out_scale=nn_out_scale, roemer=roemer)
    for name in fixed:
        model.fix(name)
    hist = train(model, data, cfg, n_steps=n_steps, progress=progress,
                 desc=f'{metric.name}/{timing}', **train_kw)
    return model, hist
