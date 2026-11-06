from .physics import DEG, TWO_PI
from .core import (jacobian_rows, jacobian_u, compute_coupling, objective, train, LossConfig,
                   safe_inv)


import copy
import numpy as np
import torch
from tqdm.auto import tqdm


def _residual_blocks(model, data, cfg, timing):
    blocks = {'data': model.residuals(data, timing)}
    if timing == 'ginn':
        E_c = model.collocation_E(cfg.n_colloc)
        blocks['phys'] = model.physics_residual(E_c) / cfg.eps_phys
    if cfg.external_precession is not None:
        f_obs, sig = cfg.external_precession
        blocks['precession'] = ((model.precession()[1] - f_obs) / sig).reshape(1)
    return blocks


def _hair_prior_hessian(model):
    n = model.u.numel()
    Hp = torch.zeros(n, n)
    for hp in model.metric.hair:
        k = model.index(hp.name)
        g = model.scales[k] * model.W[k, :]
        Hp += hp.prior.curvature() * torch.outer(g, g)
    return Hp


SQUARED_HAIR = ('Q', 'q_jnw', 'g_b')
Q_EVAL = {'Q': 2.0, 'q_jnw': 10.0, 'g_b': 10.0}


def _exact_hair_curvature(model, data, cfg, timing, J_u):

    n_u = model.u.numel()
    out = torch.zeros(n_u, n_u)
    for hp in model.metric.hair:
        k = model.index(hp.name)
        if model.free[k] == 0 or hp.name not in SQUARED_HAIR:
            continue
        a = (model.scales[k] * model.W[k, :]).detach()
        e = torch.zeros(n_u); e[k] = 1.0
        v = torch.linalg.solve_triangular(model.W, e.unsqueeze(1), upper=False).squeeze(1) / model.scales[k]
        blocks = _residual_blocks(model, data, cfg, timing)
        r = torch.cat(list(blocks.values()))
        g = torch.autograd.grad(0.5 * (r ** 2).sum(), model.u, create_graph=True)[0]
        hv = torch.autograd.grad((g * v.detach()).sum(), model.u)[0]
        exact = float(v @ hv)
        gn = float((J_u @ v).pow(2).sum())
        if exact > gn:
            out += (exact - gn) * torch.outer(a, a)
    return out


def laplace(model, data, cfg, marginalize_nn=True, timing=None, verbose=True):

    timing = timing or model.timing
    model.fold_coupling()
    blocks = _residual_blocks(model, data, cfg, timing)
    r = torch.cat(list(blocks.values()))
    use_nn = timing == 'ginn' and marginalize_nn
    params = [model.u] + (model.net.weights() if use_nn else [])
    J = jacobian_rows(r, params)
    n_u = model.u.numel()
    free = model.free.bool()
    H = J.T @ J
    H[:n_u, :n_u] += _hair_prior_hessian(model)
    H[:n_u, :n_u] += _exact_hair_curvature(model, data, cfg, timing, J[:, :n_u])
    g = J.T @ r
    if use_nn:
        Huu, Hut, Htt = H[:n_u, :n_u], H[:n_u, n_u:], H[n_u:, n_u:]
        Htt = Htt + torch.eye(Htt.shape[0]) / cfg.sigma_w ** 2
        th = torch.cat([p.detach().reshape(-1) for p in model.net.weights()])
        g_t = g[n_u:] + th / cfg.sigma_w ** 2
        Ht_inv_Htu = torch.linalg.solve(Htt, Hut.T)
        Hm = Huu - Hut @ Ht_inv_Htu
        g_u = g[:n_u] - Ht_inv_Htu.T @ g_t
    else:
        Hm = H[:n_u, :n_u]
        g_u = g[:n_u]
    Hf = Hm[free][:, free]
    Hf = 0.5 * (Hf + Hf.T)
    evals = torch.linalg.eigvalsh(Hf)
    if verbose and evals.min() <= 1e-10 * evals.max():
        print(f"[laplace] near-singular direction(s): lambda_min/lambda_max = "
              f"{(evals.min()/evals.max()).item():.1e}")
    cov = torch.zeros(n_u, n_u)
    idx = torch.where(free)[0]
    Hf_inv = safe_inv(Hf)
    cov[idx.unsqueeze(1), idx.unsqueeze(0)] = Hf_inv
    step_u = torch.zeros(n_u)
    step_u[idx] = -Hf_inv @ g_u[free]
    step_t = -torch.linalg.solve(Htt, g_t + Hut.T @ step_u) if use_nn else None
    return dict(mean=model.u.detach().clone(), cov=0.5 * (cov + cov.T), free=free,
                chi2=float((r ** 2).sum()), marginalized=use_nn, timing=timing,
                step_u=step_u, step_theta=step_t, g_u=g_u.detach())


def _capped(step, max_step=5.0):

    a = float(step.abs().max())
    return step if a <= max_step else step * (max_step / a)


DISPLAY = [
    ('M', r'$M\ [10^6 M_\odot]$', lambda P: P['M'] / 1e6),
    ('D', r'$R_0\ [{\rm kpc}]$', lambda P: P['D']),
    ('a', r'$a\ [{\rm mas}]$', lambda P: P['a'] / P['D']),   # AU / kpc = mas
    ('e', r'$e$', lambda P: P['e']),
    ('inc', r'$i\ [^\circ]$', lambda P: P['inc'] / DEG),
    ('Omega', r'$\Omega\ [^\circ]$', lambda P: P['Omega'] / DEG),
    ('omega', r'$\omega\ [^\circ]$', lambda P: P['omega'] / DEG),
    ('t0', r'$t_p\ [{\rm yr}]$', lambda P: P['t0']),
    ('vz0', r'$v_{z,0}\ [{\rm km/s}]$', lambda P: P['vz0']),
    ('x0', r'$x_0\ [{\rm mas}]$', lambda P: P['x0']),
    ('y0', r'$y_0\ [{\rm mas}]$', lambda P: P['y0']),
    ('vx0', r'$v_{x,0}\ [{\rm mas/yr}]$', lambda P: P['vx0']),
    ('vy0', r'$v_{y,0}\ [{\rm mas/yr}]$', lambda P: P['vy0']),
]


def physical_values(model, U):

    U = torch.as_tensor(U).reshape(-1, model.u.numel())
    X = model.centers + model.scales * (U @ model.W.T)
    X = X.detach()
    d = {n: X[:, i] for i, n in enumerate(model.names)}
    P = dict(M=torch.exp(d['logM']), a=torch.exp(d['loga']), e=torch.sigmoid(d['e_logit']),
             inc=d['inc'], Omega=d['Omega'], omega=d['omega'], t0=d['t0'], D=torch.exp(d['logD']),
             vz0=d['vz0'], x0=d['x0'], y0=d['y0'], vx0=d['vx0'], vy0=d['vy0'])
    out = {k: f(P).numpy() for k, _, f in DISPLAY}
    for hp in model.metric.hair:
        out[hp.name] = d[hp.name].numpy()
    return out


def labels_for(model):
    lab = {k: l for k, l, _ in DISPLAY}
    for hp in model.metric.hair:
        lab[hp.name] = hp.latex
    return lab


def sample_laplace(model, lap, n=20000, seed=0, verbose=True):

    g = torch.Generator().manual_seed(seed)
    cov = lap['cov'] + 1e-300 * torch.eye(len(lap['cov']))
    evals, evecs = torch.linalg.eigh(cov)
    L = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=0.0)))
    Z = torch.randn(n, len(cov), generator=g)
    U = lap['mean'] + Z @ L.T
    vals = physical_values(model, U)
    S = model.scales[:, None] * model.W
    cov_x = (S @ lap['cov'] @ S.T).detach()
    rng = np.random.default_rng(seed)
    for hp in model.metric.hair:
        k = model.index(hp.name)
        if bool(lap['free'][k]) and float(torch.sqrt(cov_x[k, k])) > hp.prior.hi - hp.prior.lo:
            vals[hp.name] = rng.uniform(hp.prior.lo, hp.prior.hi, size=n)
            if verbose:
                print(f"[sample_laplace] {hp.name}: Laplace width exceeds the prior range -> "
                      f"unconstrained, drawn from its prior {hp.prior}")
    keep = np.ones(n, dtype=bool)
    for hp in model.metric.hair:
        keep &= hp.prior.inside(vals[hp.name])
    if verbose and model.metric.hair:
        print(f"[sample_laplace] {100 * keep.mean():.1f}% of samples inside the hair prior "
              f"(the rest are rejected = Laplace x prior).")
    return {k: v[keep] for k, v in vals.items()}


def summarize(samples, keys=None):
    keys = keys or list(samples)
    out = {}
    for k in keys:
        lo, med, hi = np.percentile(samples[k], [15.865, 50, 84.135])
        out[k] = (med, med - lo, hi - med)
    return out


def finalize_profile(model, name, rows):
    v = np.array([r['value'] for r in rows])
    c = np.array([r['chi2'] for r in rows])
    k = np.argsort(c)[:max(5, len(v) // 2)]
    A = np.polyfit(v[k], c[k], 2)
    if A[0] > 0:
        best = -A[1] / (2 * A[0])
        best = float(np.clip(best, v.min(), v.max()))
        sigma_par = float(1.0 / np.sqrt(A[0]))
    else:
        best = float(v[np.argmin(c)])
        sigma_par = np.inf
    fine = np.linspace(v.min(), v.max(), 2001)
    cf = np.interp(fine, v, c)
    dchi = cf - cf.min()
    prior = model.metric.hair_by_name(name).prior
    logp = -0.5 * dchi + prior.logpdf_np(fine)
    post = np.exp(logp - np.nanmax(logp))
    post /= np.trapezoid(post, fine)
    cdf = np.cumsum(post) * (fine[1] - fine[0])
    q = lambda p: float(np.interp(p, cdf, fine))

    def interval(level):
        inside = fine[dchi <= level]
        return (float(inside.min()), float(inside.max())) if inside.size else (np.nan, np.nan)
    above = fine[(fine > fine[np.argmin(cf)]) & (dchi >= 2.71)]
    below = fine[(fine < fine[np.argmin(cf)]) & (dchi >= 2.71)]
    return dict(name=name, values=v, chi2=c, dchi2=c - c.min(), rows=rows, best=best,
                sigma_parabola=sigma_par,
                upper_1sided95=float(above.min()) if above.size else np.nan,
                lower_1sided95=float(below.max()) if below.size else np.nan,
                ci68=interval(1.0), ci95=interval(4.0), fine=fine, posterior=post,
                cred68=(q(0.15865), q(0.84135)), cred95=(q(0.02275), q(0.97725)),
                upper95=q(0.95), lower05=q(0.05), median=q(0.5),
                phys_max=max(r['phys'] for r in rows),
                best_model=min(rows, key=lambda r: r['chi2'])['model'])


def gn_profile(model_start, data, cfg, name, values, n_iter=6, n_retrain=500, tol=0.02,
               progress=True, **train_kw):

    rows = []
    bar = tqdm(list(values), desc=f'GN profile {name}', disable=not progress)
    for v in bar:
        m = copy.deepcopy(model_start)
        m.fix(name)
        m.set_value(name, float(v))
        decs = []
        for it in range(n_iter):
            lap = laplace(m, data, cfg, verbose=False)
            dec = float(-(lap['g_u'] @ lap['step_u']))
            decs.append(dec)
            if dec < tol or it == n_iter - 1:
                break
            with torch.no_grad():
                m.u += _capped(lap['step_u'])
            m.project_to_prior_()
            train(m, data, cfg, n_steps=n_retrain, progress=False, **train_kw)
        with torch.enable_grad():
            _, T = objective(m, data, cfg)
        T = {k: x.item() for k, x in T.items()}
        like = T['data'] + T.get('precession', 0.0) - max(decs[-1], 0.0)
        rows.append(dict(value=float(v), chi2=like, data=T['data'], phys=T.get('phys', 0.0),
                         nn_prior=T.get('nn_prior', 0.0), model=m, iterations=len(decs),
                         converged=decs[-1] < tol, last_decrease=decs[-1], lap=lap))
        bar.set_postfix({name: f'{float(v):.3g}', 'chi2': f'{like:.3f}', 'it': len(decs)})
    return finalize_profile(model_start, name, rows)


from math import erf, sqrt as _sqrt, log as _log


def _phi(z):
    return 0.5 * (1.0 + erf(z / _sqrt(2.0)))


def _internal_gaussian(model, lap):
    S = model.scales[:, None] * model.W
    step = lap.get('step_u', torch.zeros_like(lap['mean']))
    mean_x = model.centers + S @ (lap['mean'] + step)
    cov_x = S @ lap['cov'] @ S.T
    return mean_x.detach(), cov_x.detach()


def mixture_posterior(entries, n=40000, seed=0, verbose=True):

    g = torch.Generator().manual_seed(seed)
    logw, info = [], []
    for e in entries:
        m, lap = e['model'], e['lap']
        free = lap['free']
        mean_x, cov_x = _internal_gaussian(m, lap)
        C = cov_x[free][:, free]
        ld = float(torch.linalg.slogdet(C + 1e-300 * torch.eye(len(C)))[1])
        name, lo, hi = e['cell']
        lw = -0.5 * e['chi2'] + 0.5 * ld + _log(hi - lo)
        lw += float(m.metric.hair_by_name(name).prior.logpdf_np(np.array([0.5 * (lo + hi)]))[0])
        for hp in m.metric.hair:
            k = m.index(hp.name)
            if not bool(free[k]):
                continue
            mu, sd = float(mean_x[k]), float(torch.sqrt(cov_x[k, k]))
            mass = _phi((hp.prior.hi - mu) / sd) - _phi((hp.prior.lo - mu) / sd)
            lw += _log(max(mass, 1e-300)) + float(hp.prior.logpdf_np(np.array([np.clip(mu, hp.prior.lo, hp.prior.hi)]))[0])
        logw.append(lw)
        info.append(dict(cell=e['cell'], chi2=e['chi2'], logdet=ld))
    logw = np.array(logw)
    w = np.exp(logw - logw.max()); w /= w.sum()
    counts = np.random.default_rng(seed).multinomial(n, w)
    out = {}
    for e, c in zip(entries, counts):
        if c == 0:
            continue
        m, lap = e['model'], e['lap']
        step = lap.get('step_u', torch.zeros_like(lap['mean']))
        cov = lap['cov'] + 1e-300 * torch.eye(len(lap['cov']))
        ev, evec = torch.linalg.eigh(0.5 * (cov + cov.T))
        L = evec @ torch.diag(torch.sqrt(torch.clamp(ev, min=0.0)))
        U = (lap['mean'] + step) + torch.randn(3 * c, len(cov), generator=g) @ L.T
        vals = physical_values(m, U)
        name, lo, hi = e['cell']
        vals[name] = np.random.default_rng(seed + c).uniform(lo, hi, size=3 * c)
        keep = np.ones(3 * c, dtype=bool)
        for hp in m.metric.hair:
            keep &= hp.prior.inside(vals[hp.name])
        idx = np.where(keep)[0][:c]
        for k, v in vals.items():
            out.setdefault(k, []).append(v[idx])
    samples = {k: np.concatenate(v) for k, v in out.items()}
    for i, wi in zip(info, w):
        i['weight'] = float(wi)
    if verbose:
        print(f"{'cell':>28s} {'chi2':>9s} {'logdet':>9s} {'weight':>8s}")
        for i in info:
            nm, lo, hi = i['cell']
            print(f"{nm:>14s} [{lo:7.3g},{hi:7.3g}] {i['chi2']:9.3f} {i['logdet']:9.2f} {i['weight']:8.3f}")
    return samples, info


def gn_polish(model, data, cfg, n_iter=8, n_retrain=400, tol=0.02, verbose=True, **train_kw):

    hair = [hp.name for hp in model.metric.hair if model.free[model.index(hp.name)] > 0]
    hist = []

    def at_bound_pushing_out(lap):

        S = model.scales[:, None] * model.W
        dx = S @ lap['step_u']
        x = model.internal().detach()
        act = []
        for h in hair:
            hp = model.metric.hair_by_name(h); k = model.index(h)
            tol = 1e-9 * (hp.prior.hi - hp.prior.lo)
            if (x[k] <= hp.prior.lo + tol and dx[k] < 0) or (x[k] >= hp.prior.hi - tol and dx[k] > 0):
                act.append(h)
        return act

    alpha, prev_dec = 1.0, None
    for it in range(n_iter):
        lap = laplace(model, data, cfg, verbose=False)
        active = at_bound_pushing_out(lap)
        for h in active:
            model.fix(h)
        if active:
            lap = laplace(model, data, cfg, verbose=False)
        dec = float(-(lap['g_u'] @ lap['step_u']))
        P = model.physical()
        hist.append(dict(it=it, chi2_gain=dec, **{h: float(P['par'][h]) for h in hair}))
        if verbose:
            print(f"GN {it}: predicted chi2 gain {dec:8.3f}   " +
                  "  ".join(f"{h} = {float(P['par'][h]):.5g}" for h in hair))
        if dec < tol:
            for h in active:
                model.fix(h, fixed=False)
            break
        before = {h: float(model.physical()['par'][h]) for h in hair}
        with torch.no_grad():
            model.u += alpha * _capped(lap['step_u'])
        model.project_to_prior_()
        after = {h: float(model.physical()['par'][h]) for h in hair}
        if any(before[h] * after[h] < 0 for h in hair) or (prev_dec is not None and dec > prev_dec):
            alpha *= 0.5
        prev_dec = dec
        if alpha < 1.0 / 32:
            break
        for h in hair:
            model.fix(h)
        train(model, data, cfg, n_steps=n_retrain, progress=False, **train_kw)
        for h in hair:
            model.fix(h, fixed=False)
    lap = laplace(model, data, cfg, verbose=False)
    return lap, hist


def hair_estimates(model, lap):

    S = model.scales[:, None] * model.W
    x = model.internal().detach()
    C = S @ lap['cov'] @ S.T
    out = {}
    for hp in model.metric.hair:
        k = model.index(hp.name)
        out[hp.name] = (float(x[k]), float(torch.sqrt(C[k, k])))
    return out


import copy
import numpy as np
import torch


def chi2_parts(model, data, timing=None):
    with torch.no_grad():
        ra, dec, v = model.predict_data(data, timing)
        out = dict(ra=(((ra - data.ra) / data.ra_e) ** 2).sum().item(),
                   dec=(((dec - data.dec) / data.dec_e) ** 2).sum().item(),
                   rv=(((v - data.vlos) / data.vlos_e) ** 2).sum().item())
    out['total'] = out['ra'] + out['dec'] + out['rv']
    return out


def n_free(model, timing=None):
    return int(model.free.sum().item())


def physics_residual_stats(model, n=4000):
    with torch.no_grad():
        R = model.physics_residual(model.collocation_E(n))
    return dict(rms=R.pow(2).mean().sqrt().item(), max=R.abs().max().item())

def timing_comparison(model, t=None, n=3000):
    if t is None:
        P = model.physical()
        t = torch.linspace(float(P['t0']) - 10.5, float(P['t0']) + 16.5, n)
    P = model.physical()
    C = model.orbit_constants(P)
    with torch.no_grad():
        chiK = model.chi_of_t(t, P, C, 'kepler')
        chiG = model.chi_of_t(t, P, C, 'ginn')
        chiE = model.chi_of_t(t, P, C, 'exact')
        obs = {tm: model.forward(t, tm) for tm in ('ginn', 'exact', 'kepler')}
    d = lambda a, b, i: (obs[a][i] - obs[b][i]).numpy()
    pos = lambda a, b: np.hypot(d(a, b, 0), d(a, b, 1))
    return dict(t=t.numpy(), dchi_nn=(chiG - chiK).numpy(), dchi_exact=(chiE - chiK).numpy(),
                pos_nn_minus_exact=pos('ginn', 'exact'), pos_kepler_minus_exact=pos('kepler', 'exact'),
                rv_nn_minus_exact=d('ginn', 'exact', 2), rv_kepler_minus_exact=d('kepler', 'exact', 2))


def nn_information_share(model, data, cfg):
    m = copy.deepcopy(model)
    m.fold_coupling()
    J_f = jacobian_u(m, data, 'ginn', cfg)
    compute_coupling(m, cfg, damping=1e-8)
    J_c = jacobian_u(m, data, 'ginn', cfg)
    Hp = _hair_prior_hessian(m)
    free = m.free.bool()
    S = (m.scales[:, None] * m.W)[free][:, free]

    def sig(J):
        F = (J.T @ J + Hp)[free][:, free] + 1e-10 * torch.eye(int(free.sum()))
        cov_x = S @ torch.linalg.inv(F) @ S.T
        return torch.sqrt(torch.diagonal(cov_x))
    s_f, s_c = sig(J_f), sig(J_c)
    names = [n for n, f in zip(m.names, free) if f]
    out = {}
    for i, n in enumerate(names):
        out[n] = dict(sigma_frozen=s_f[i].item(), sigma_with_nn=s_c[i].item(),
                      info_share=1.0 - (s_c[i] / s_f[i]).item() ** 2)
    return out


def nn_role_report(ginn, data, cfg, kepler=None, exact=None, lap=None, show=True):
    rep = {}
    rep['chi2_ginn'] = chi2_parts(ginn, data, 'ginn')
    rep['chi2_ablation'] = chi2_parts(ginn, data, 'kepler')
    rep['chi2_exact_same_params'] = chi2_parts(ginn, data, 'exact')
    if kepler is not None:
        rep['chi2_kepler_refit'] = chi2_parts(kepler, data, 'kepler')
    if exact is not None:
        rep['chi2_exact_refit'] = chi2_parts(exact, data, 'exact')
    rep['physics'] = physics_residual_stats(ginn)
    tc = timing_comparison(ginn)
    rep['timing'] = dict(rms_dchi_nn=np.sqrt(np.mean(tc['dchi_nn'] ** 2)),
                         rms_dchi_exact=np.sqrt(np.mean(tc['dchi_exact'] ** 2)),
                         rms_mismatch=np.sqrt(np.mean((tc['dchi_nn'] - tc['dchi_exact']) ** 2)),
                         max_pos_err_mas=np.max(tc['pos_nn_minus_exact']),
                         max_rv_err_kms=np.max(np.abs(tc['rv_nn_minus_exact'])),
                         max_pos_effect_mas=np.max(tc['pos_kepler_minus_exact']),
                         max_rv_effect_kms=np.max(np.abs(tc['rv_kepler_minus_exact'])))
    rep['timing_curves'] = tc
    rep['info_share'] = nn_information_share(ginn, data, cfg)
    if exact is not None and lap is not None:
        x_g = ginn.internal().detach()
        x_e = exact.internal().detach()
        S = ginn.scales[:, None] * ginn.W
        sd = torch.sqrt(torch.diagonal(S @ lap['cov'] @ S.T))
        rep['param_shift_sigma'] = {n: ((x_g[i] - x_e[i]) / sd[i]).item()
                                    for i, n in enumerate(ginn.names) if sd[i] > 0}
    if show:
        print_report(rep, data)
    return rep


def print_report(rep, data):
    N = data.n_points
    print("chi2 (N_data = %d)                     RA      Dec      RV     total" % N)
    rows = [('GINN', 'chi2_ginn'),
            ('without network', 'chi2_ablation'),
            ('geodesic timing numerical', 'chi2_exact_same_params'),
            ('Kepler fit', 'chi2_kepler_refit'),
            ('Numerical fit', 'chi2_exact_refit')]

    for lab, k in rows:
        if k in rep:
            c = rep[k]
            print(f"  {lab:43s} {c['ra']:7.2f} {c['dec']:8.2f} {c['rv']:7.2f} {c['total']:9.3f}")
    print("we compare chi-squared improvement due to GINN and exact numerical correction - should be more or less equal")
    if 'chi2_kepler_refit' in rep and 'chi2_exact_refit' in rep:
        print(f"Kepler - exact numerical: chi2(Kepler) - chi2(exact) = "
              f"{rep['chi2_kepler_refit']['total'] - rep['chi2_exact_refit']['total']:.3f}")
        print(f"Kepler - GINN:                     chi2(Kepler) - chi2(GINN)  = "
              f"{rep['chi2_kepler_refit']['total'] - rep['chi2_ginn']['total']:.3f}")
    t = rep['timing']


import numpy as np
import torch


def squared_hair_width(model, data, cfg, name, q_eval=10.0, n_retrain=600, lr_nn=2e-3):
    m = copy.deepcopy(model)
    m.set_value(name, float(q_eval))
    m.fix(name)
    train(m, data, cfg, n_steps=n_retrain, progress=False, lr_nn=lr_nn)
    m.fix(name, fixed=False)
    nu, th, k = m.u.numel(), m.net.weights(), m.index(name)
    blocks = _residual_blocks(m, data, cfg, 'ginn')
    r = torch.cat([v for key, v in blocks.items() if key != 'phys'])
    Jd = jacobian_rows(r, [m.u] + th)
    Jp = jacobian_rows(blocks['phys'], [m.u] + th)
    A, B = Jp[:, nu:], Jp[:, :nu]
    D = -A.T @ torch.linalg.solve(A @ A.T + torch.eye(A.shape[0]) / cfg.sigma_w ** 2, B)
    Jc = (Jd[:, :nu] + Jd[:, nu:] @ D).detach()
    fr = torch.where(m.free.bool())[0]
    Hinv = torch.linalg.inv((Jc.T @ Jc)[fr][:, fr])
    a = (m.scales[k] * m.W[k, :])[fr]
    q0 = float(m.physical()['par'][name])
    q_star = q0 + float(a @ (-Hinv @ (Jc.T @ r.detach())[fr]))
    s = (q0 - q_star) / float(a @ Hinv @ a) / q0
    return dict(s=s, sigma=float(1.0 / np.sqrt(s)) if s > 0 else float('nan'), q_eval=q0)


def _cov_x(model, lap):
    S = model.scales[:, None] * model.W
    return (S @ lap['cov'] @ S.T).detach()


def _conditional_laplace(model, data, cfg, names):
    m = copy.deepcopy(model)
    for n in names:
        m.fix(n)
    lap = laplace(m, data, cfg, verbose=False)
    return _cov_x(m, lap), lap['free']


def free_hair(model, lap=None):
    free = model.free.bool() if lap is None else lap['free']
    return [hp for hp in model.metric.hair if bool(free[model.index(hp.name)])]


def posterior(model, data, cfg, q_eval=None, verbose=True):
    lap = laplace(model, data, cfg, verbose=False)
    sq = [hp for hp in free_hair(model, lap) if hp.name in SQUARED_HAIR]
    post = dict(lap, squared={})
    if not sq:
        return post
    cov_x, _ = _conditional_laplace(model, data, cfg, [hp.name for hp in sq])
    cov_x = cov_x.clone()
    for hp in sq:
        k = model.index(hp.name)
        w = squared_hair_width(model, data, cfg, hp.name, q_eval=(q_eval or Q_EVAL)[hp.name])
        sig = w['sigma'] if np.isfinite(w['sigma']) else hp.prior.hi - hp.prior.lo
        if verbose:
            print(f"{hp.name}: d chi2 / d({hp.name}^2) = {w['s']:.3e}  ->  sigma = {sig:.3g}"
                  f"   (slope measured at {hp.name} = {w['q_eval']:.3g})")
        cov_x[k, :] = 0.0
        cov_x[:, k] = 0.0
        cov_x[k, k] = sig ** 2
        post['squared'][hp.name] = dict(w, sigma=sig)
    S_inv = torch.linalg.inv(model.scales[:, None] * model.W)
    cov_u = S_inv @ cov_x @ S_inv.T
    post['cov'] = 0.5 * (cov_u + cov_u.T)
    return post


def print_estimates(model, post):
    est = hair_estimates(model, post)
    for hp in model.metric.hair:
        k = model.index(hp.name)
        v, s = est[hp.name]
        if not bool(post['free'][k]):
            print(f"{hp.name:>14s} = {v:.5g}   (held fixed)")
            continue
        width = hp.prior.hi - hp.prior.lo
        if s > 0.3 * width:
            flag = '   <- not constrained by the data (error ~ prior range)'
        elif np.isfinite(hp.gr_value):
            flag = f'   ({abs(v - hp.gr_value) / s:.1f} sigma from GR)'
        else:
            flag = ''
        if hp.name in SQUARED_HAIR:
            flag += '   (+ and - give the same metric)'
        print(f"{hp.name:>14s} = {v:.5g} +- {s:.3g}{flag}")
    return est


def _log_mass(mu, sig, lo, hi):
    return _log(max(_phi((hi - mu) / sig) - _phi((lo - mu) / sig), 1e-300))


def evidence(model, data, cfg, post):
    hair = free_hair(model, post)
    cov_x, free = _conditional_laplace(model, data, cfg, [hp.name for hp in hair])
    logdet = float(torch.linalg.slogdet(cov_x[free][:, free])[1])
    est = hair_estimates(model, post)
    occam = 0.0
    for hp in hair:
        v, s = est[hp.name]
        occam += _log(_sqrt(2.0 * np.pi) * s / (hp.prior.hi - hp.prior.lo)) + _log_mass(v, s, hp.prior.lo, hp.prior.hi)
    return dict(chi2=chi2_parts(model, data)['total'], logdet_rest=logdet, ln_occam=occam,
                hair={hp.name: est[hp.name] for hp in hair})


def jeffreys(lnB):
    a = abs(lnB)
    s = 'inconclusive' if a < 1 else 'weak' if a < 2.5 else 'moderate' if a < 5 else 'strong'
    return s if a < 1 else f"{s} for {'the model' if lnB > 0 else 'Schwarzschild'}"


def bayes_factor(ev, ev_ref):
    dchi2 = ev['chi2'] - ev_ref['chi2']
    dvol = 0.5 * (ev['logdet_rest'] - ev_ref['logdet_rest'])
    lnB = -0.5 * dchi2 + ev['ln_occam'] + dvol
    return dict(dchi2=dchi2, ln_occam=ev['ln_occam'], dln_vol=dvol, lnB=lnB, verdict=jeffreys(lnB))


def bayes_factor_profile(prof, prior, ev_ref, lower_edge=True):
    v, c = np.asarray(prof['values']), np.asarray(prof['chi2'])
    mids = 0.5 * (v[1:] + v[:-1])
    lo = prior.lo if lower_edge else v[0] - (mids[0] - v[0])
    edges = np.concatenate([[lo], mids, [min(prior.hi, v[-1] + (v[-1] - mids[-1]))]])
    w = np.diff(edges) / (prior.hi - prior.lo) * np.exp(-0.5 * (c - ev_ref['chi2']))
    lnB = float(np.log(w.sum()))
    dchi2 = float(c.min() - ev_ref['chi2'])
    return dict(dchi2=dchi2, ln_occam=lnB + 0.5 * dchi2, dln_vol=0.0, lnB=lnB, verdict=jeffreys(lnB))


def print_bayes(name, bf):
    print(f"{'model':16s} {'dchi2':>8s} {'ln Occam':>9s} {'dln vol':>8s} {'ln B':>7s}   evidence")
    print(f"{name:16s} {bf['dchi2']:8.2f} {bf['ln_occam']:9.2f} {bf['dln_vol']:8.2f} {bf['lnB']:7.2f}   {bf['verdict']}")
