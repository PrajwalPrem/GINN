import numpy as np
import torch
from scipy.optimize import minimize
from .core import LossConfig
from . import analysis as an


class Expansion:
    def __init__(self, model, data, post, cfg=None):
        self.cfg = cfg or LossConfig()
        self.model, self.data, self.post = model, data, post
        model.fold_coupling()
        w = model.net.weights()
        self.shapes, self.sizes = [p.shape for p in w], [p.numel() for p in w]
        self.th0 = torch.cat([p.detach().reshape(-1) for p in w])
        self.u0 = model.u.detach().clone()
        self.E_c = model.collocation_E(self.cfg.n_colloc)
        self.n_data = model.residuals(data, 'ginn').numel()
        self.fr = torch.where(post['free'])[0]
        C = post['cov'][self.fr][:, self.fr]
        self.L = torch.linalg.cholesky(0.5 * (C + C.T))
        self.n = len(self.fr)
        self.S = model.scales[:, None] * model.W
        Ju = torch.func.jacfwd(lambda v: self._net_part(v, self.th0))(self.u0)[:, self.fr] @ self.L
        Jt = torch.func.jacrev(lambda t: self._net_part(self.u0, t), chunk_size=256)(self.th0)
        Ht = Jt.T @ Jt + torch.eye(len(self.th0)) / self.cfg.sigma_w ** 2
        self.D = -torch.linalg.solve(Ht, Jt.T @ Ju).detach()
        self.Jt_full = torch.cat([Jt, torch.eye(len(self.th0)) / self.cfg.sigma_w]).detach()
        self.Ht = Ht.detach()
        self.E = None

    def _unflat(self, th):
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(th[i:i + n].view(s))
            i += n
        return out

    def _net_part(self, u, th):
        m = self.model
        m.net_params = lambda u_=None: self._unflat(th)
        try:
            rd = m.residuals(self.data, 'ginn', u)
            rp = m.physics_residual(self.E_c, u) / self.cfg.eps_phys
        finally:
            del m.net_params
        return torch.cat([rd, rp])

    def u_of(self, z):
        return self.u0.clone().index_add(0, self.fr, self.L @ z)

    def R(self, z):
        th = self.th0 + self.D @ z
        if self.E is not None:
            th = th + 0.5 * torch.einsum('tij,i,j->t', self.E, z, z)
        return torch.cat([self._net_part(self.u_of(z), th), th / self.cfg.sigma_w])

    def expand(self):
        z0 = torch.zeros(self.n, dtype=self.u0.dtype)
        self.E = None
        K = torch.func.jacfwd(torch.func.jacfwd(self.R))(z0).detach()
        K = 0.5 * (K + K.transpose(1, 2))
        N, n = K.shape[0], self.n
        self.E = -torch.linalg.solve(self.Ht, self.Jt_full.T @ K.reshape(N, n * n)).reshape(-1, n, n)
        K = K + torch.einsum('nt,tij->nij', self.Jt_full, self.E)
        self.R0 = self.R(z0).detach()
        self.J = torch.func.jacfwd(self.R)(z0).detach()
        nd = self.n_data
        r0, J, K = self.R0[:nd], self.J[:nd], K[:nd]
        Q, _ = torch.linalg.qr(torch.cat([r0[:, None], J, K.reshape(nd, -1)], 1))
        self.r0, self.j = Q.T @ r0, Q.T @ J
        self.k = torch.einsum('na,nij->aij', Q, K)
        self.L0 = float(self.r0 @ self.r0)
        return self

    def chi2_exact(self, z):
        return (self.model.residuals(self.data, 'exact', self.u_of(z)) ** 2).sum()

    def loss(self, Z, kind='dali'):
        Z = torch.as_tensor(np.atleast_2d(Z), dtype=self.u0.dtype)
        if kind == 'laplace':
            return (Z ** 2).sum(1).numpy()
        if kind == 'exact':
            f = torch.func.vmap(self.chi2_exact)
            with torch.no_grad():
                return torch.cat([f(Z[i:i + 1000]) for i in range(0, len(Z), 1000)]).numpy()
        lin = self.r0 + Z @ self.j.T
        if kind == 'newton':
            H = self.j.T @ self.j + torch.einsum('a,aij->ij', self.r0, self.k)
            return (self.L0 + 2 * Z @ (self.j.T @ self.r0) + ((Z @ H) * Z).sum(1)).numpy()
        quad = 0.5 * torch.einsum('aij,ni,nj->na', self.k, Z, Z)
        return ((lin + quad) ** 2).sum(1).numpy()

    def loss_at_centre(self, kind):
        return float(self.loss(np.zeros((1, self.n)), kind)[0])

    def internal_row(self, name):
        k = self.model.index(name)
        return (self.S[k, self.fr] @ self.L).detach(), k

    def physical(self, Z):
        U = self.u0.repeat(len(Z), 1)
        U[:, self.fr] += torch.as_tensor(Z, dtype=U.dtype) @ self.L.T
        return an.physical_values(self.model, U)


def _in_prior(P, Z):
    vals = P.physical(Z)
    ok = np.ones(len(Z), bool)
    for hp in P.model.metric.hair:
        ok &= hp.prior.inside(vals[hp.name])
    return ok


def sample(P, kind='dali', n=600000, seed=1):
    rng = np.random.default_rng(seed)
    if kind == 'laplace':
        Z = rng.standard_normal((min(n, 200000), P.n))
        Z = Z[_in_prior(P, Z)]
        return dict(vals=P.physical(Z), w=np.full(len(Z), 1.0 / len(Z)), Z=Z, n_eff=float(len(Z)))
    import emcee
    L0 = P.loss_at_centre('dali')

    def logp(Z):
        return np.where(_in_prior(P, Z), -0.5 * (P.loss(Z, 'dali') - L0), -np.inf)
    nw = 4 * P.n
    n_steps = max(3000, n // nw)
    s = emcee.EnsembleSampler(nw, P.n, logp, vectorize=True)
    s.run_mcmc(0.3 * rng.standard_normal((nw, P.n)), n_steps, progress=False)
    thin = max(1, (n_steps - 1000) * nw // 200000)
    Z = s.get_chain(discard=1000, thin=thin, flat=True)
    tau = float(np.max(s.get_autocorr_time(discard=1000, quiet=True)))
    return dict(vals=P.physical(Z), w=np.full(len(Z), 1.0 / len(Z)), Z=Z,
                n_eff=(n_steps - 1000) * nw / tau, tau=tau)


def reweight_exact(P, chain, n=6000, seed=2):
    rng = np.random.default_rng(seed)
    Z = chain['Z'][rng.choice(len(chain['Z']), min(n, len(chain['Z'])), replace=False)]
    dL = (P.loss(Z, 'exact') - P.loss_at_centre('exact')) - (P.loss(Z, 'dali') - P.loss_at_centre('dali'))
    w = np.exp(-0.5 * (dL - dL.min()))
    w /= w.sum()
    return dict(vals=P.physical(Z), w=w, Z=Z, n_eff=float(1.0 / np.sum(w ** 2)), n=len(Z))


def pointwise(P, n=20, scale=1.5, seed=0):
    Z = scale * np.random.default_rng(seed).standard_normal((n, P.n))
    ex = P.loss(Z, 'exact') - P.loss_at_centre('exact')
    return {k: np.abs(P.loss(Z, k) - P.loss_at_centre(k) - ex) for k in ('laplace', 'newton', 'dali')}


def wstats(x, w):
    m = np.sum(w * x)
    s = np.sqrt(np.sum(w * (x - m) ** 2))
    o = np.argsort(x)
    c = np.cumsum(w[o])
    q = lambda p: float(np.interp(p, c, x[o]))
    return dict(mean=m, sd=s, skew=np.sum(w * (x - m) ** 3) / s ** 3, lo95=q(0.025), hi95=q(0.975))


def stats(res, keys):
    return {k: wstats(np.asarray(res['vals'][k]), res['w']) for k in keys}


def print_compare(st, keys, names=('dali',), hair=()):
    print(f"{'':8s} {'Gauss-Newton + Laplace':>26s} | " +
          " | ".join(f"{n + ': shift, sd ratio, skew':>34s}" for n in names))
    for k in keys:
        L = st['laplace'][k]
        row = f"{k:8s} {L['mean']:14.6g} +- {L['sd']:<9.3g} |"
        for n in names:
            s = st[n][k]
            row += f" {(s['mean'] - L['mean']) / L['sd']:+10.2f}  {s['sd'] / L['sd']:10.3f}  {s['skew']:+10.2f} |"
        print(row)
    for k in hair:
        print(f"\n{k}: 95% interval  Gauss-Newton + Laplace [{st['laplace'][k]['lo95']:+.4g}, {st['laplace'][k]['hi95']:+.4g}]" +
              "".join(f"   {n} [{st[n][k]['lo95']:+.4g}, {st[n][k]['hi95']:+.4g}]" for n in names))


def profile(P, name, n_sigma=np.linspace(-3, 3, 9), exact=False):
    a, k = P.internal_row(name)
    sig = float(a.norm())
    nvec = a / a.norm()
    Qn, _ = torch.linalg.qr(torch.cat([nvec[:, None], torch.eye(P.n, dtype=a.dtype)], 1))
    N = Qn[:, 1:P.n]
    kinds = ('dali', 'exact') if exact else ('dali',)
    out = dict(name=name, k=k, x0=float(P.model.internal().detach()[k]), dx=np.asarray(n_sigma) * sig,
               sigma=sig, laplace=np.asarray(n_sigma, float) ** 2, **{kd: [] for kd in kinds})
    ys = {kd: np.zeros(P.n - 1) for kd in kinds}
    for t in n_sigma:
        z_a = nvec * float(t)
        f_d = lambda y: float(P.loss((z_a + N @ torch.as_tensor(y)).numpy()[None], 'dali')[0])

        def f_e(y):
            y = torch.as_tensor(y, dtype=a.dtype).requires_grad_(True)
            L = P.chi2_exact(z_a + N @ y)
            g, = torch.autograd.grad(L, y)
            return float(L), g.numpy()
        for kd in kinds:
            f, jac = (f_d, None) if kd == 'dali' else (f_e, True)
            res = minimize(f, ys[kd], jac=jac, method='BFGS', options=dict(gtol=1e-5))
            ys[kd] = res.x
            out[kd].append(float(res.fun))
    for kd in kinds:
        v = np.array(out[kd])
        out[kd] = v - v.min()
    return out


def plot_profile(ax, pr, label=None, transform=None):
    x = pr['x0'] + pr['dx']
    if transform is not None:
        x = transform(x)
    ax.plot(x, pr['laplace'], 'k--', lw=1, label='Gauss-Newton + Laplace')
    ax.plot(x, pr['dali'], '-', color='C0', lw=1.5, label='higher order')
    if 'exact' in pr:
        ax.plot(x, pr['exact'], 'o', color='C3', ms=3.5, label='exact solver')
    ax.set_xlabel(label or pr['name'])
    ax.set_ylabel(r'$\Delta\chi^2$ (profile)')
