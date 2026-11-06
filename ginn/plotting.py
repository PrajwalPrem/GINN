import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .physics import DEG


SIGMA_LEVELS_2D = (1 - np.exp(-0.5), 1 - np.exp(-2.0))


COLORS = ['#1f4e9c', '#d1495b', '#2a9d8f', '#e9a03b']


def corner_plot(sample_sets, labels_map, keys, names=None, truths=None, truth_label=None,
                colors=None, smooth=1.0, bins=40, title=None):

    import corner
    colors = colors or COLORS
    names = names or [f'set {i}' for i in range(len(sample_sets))]
    fig = None
    for s, (smp, c) in enumerate(zip(sample_sets, colors)):
        X = np.column_stack([smp[k] for k in keys])
        w = np.ones(len(X)) / len(X)
        from matplotlib.colors import to_rgba
        fig = corner.corner(
            X, fig=fig, weights=w, labels=[labels_map[k] for k in keys], color=c, bins=bins,
            smooth=smooth, smooth1d=smooth, levels=SIGMA_LEVELS_2D, plot_datapoints=False,
            plot_density=False, fill_contours=True, no_fill_contours=True,
            contourf_kwargs=dict(colors=[(1, 1, 1, 0), to_rgba(c, 0.18), to_rgba(c, 0.45)]),
            contour_kwargs=dict(colors=[c], linewidths=1.2),
            quantiles=[0.15865, 0.5, 0.84135] if s == 0 else None,
            show_titles=False,
            label_kwargs=dict(fontsize=12), hist_kwargs=dict(color=c, lw=1.4),
            max_n_ticks=3, labelpad=0.08)
    K = len(keys)
    axes = np.array(fig.axes).reshape(K, K)
    for i, k in enumerate(keys):
        lo, med, hi = np.percentile(sample_sets[0][k], [15.865, 50, 84.135])
        err = min(med - lo, hi - med)
        dec = int(max(1, 1 - np.floor(np.log10(err)))) if err > 0 else 2
        axes[i, i].set_title(f"{labels_map[k]} = ${med:.{dec}f}^{{+{hi - med:.{dec}f}}}_{{-{med - lo:.{dec}f}}}$",
                             fontsize=10)
    for ax_ in fig.axes:
        for axis in (ax_.xaxis, ax_.yaxis):
            fmt = axis.get_major_formatter()
            if hasattr(fmt, 'set_useOffset'):
                fmt.set_useOffset(False)
    if truths is not None:
        import corner
        corner.overplot_lines(fig, [truths.get(k, None) for k in keys], color='0.35',
                              ls='--', lw=1.0)
    handles = [plt.Line2D([], [], color=c, lw=3) for c in colors[:len(sample_sets)]]
    labs = list(names)
    if truths is not None:
        handles.append(plt.Line2D([], [], color='0.35', ls='--'))
        labs.append(truth_label or 'reference')
    fig.legend(handles, labs, loc='upper right', fontsize=12, frameon=False,
               bbox_to_anchor=(0.98, 0.98))
    if title:
        fig.suptitle(title, fontsize=13, x=0.55)
    return fig


def plot_fit(model, data, timing=None, others=None):
    others = others or {}
    t0 = min(data.t_span[0], float(model.physical()['t0'])) - 1
    td = torch.linspace(t0, data.t_span[1] + 1, 3000)
    with torch.no_grad():
        ra_d, dec_d, v_d = model.forward(td, timing)
        ra, dec, v = model.predict_data(data, timing)
    fig = plt.figure(figsize=(15, 8.5))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[3, 1, 3, 1], hspace=0.08, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    ax.errorbar(data.ra, data.dec, xerr=data.ra_e, yerr=data.dec_e, fmt='.', color='k',
                ms=3, alpha=0.6, lw=0.7)
    ax.plot(ra_d, dec_d, color=COLORS[0], lw=1.3, label=f'{(model.metric.label or model.metric.name)} ({timing or model.timing})')
    for (lab, m), c in zip(others.items(), COLORS[1:]):
        with torch.no_grad():
            r2, d2, _ = m.forward(td)
        ax.plot(r2, d2, color=c, lw=1.0, ls='--', label=lab)
    ax.plot(0, 0, 'k+', ms=12)
    ax.invert_xaxis(); ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta\alpha$ [mas]'); ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.legend(fontsize=9, loc='lower left')

    def panel(r, c, tt, y, ye, yd, ym, ylab):
        a = fig.add_subplot(gs[r, c])
        a.errorbar(tt, y, ye, fmt='.', color='k', ms=3, alpha=0.6, lw=0.7)
        a.plot(td, yd, color=COLORS[0], lw=1.2)
        a.set_ylabel(ylab); a.tick_params(labelbottom=False)
        b = fig.add_subplot(gs[r + 1, c], sharex=a)
        b.errorbar(tt, (y - ym) / ye, 1.0, fmt='.', color='k', ms=3, alpha=0.6, lw=0.7)
        b.axhline(0, color=COLORS[0], lw=1)
        b.set_ylim(-4.5, 4.5); b.set_ylabel(r'$\Delta/\sigma$')
        return a, b
    panel(0, 1, data.t_pos, data.ra, data.ra_e, ra_d, ra, r'$\Delta\alpha$ [mas]')
    panel(0, 2, data.t_pos, data.dec, data.dec_e, dec_d, dec, r'$\Delta\delta$ [mas]')
    a, b = panel(2, 1, data.t_rv, data.vlos, data.vlos_e, v_d, v, r'$v_{\rm LOS}$ [km/s]')
    b.set_xlabel('epoch [yr]')
    a2 = fig.add_subplot(gs[2:, 2])
    a2.errorbar(data.t_rv, data.vlos, data.vlos_e, fmt='.', color='k', ms=4, lw=0.8)
    a2.plot(td, v_d, color=COLORS[0])
    tp = float(model.physical()['t0'])
    a2.set_xlim(tp - 0.6, tp + 0.6)
    a2.text(0.03, 0.95, 'pericentre zoom', transform=a2.transAxes, va='top', fontsize=10)
    a2.set_xlabel('epoch [yr]')
    return fig


def plot_timing(tc, data=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1, 1.4], hspace=0.08))
    ax[0].plot(tc['t'], tc['dchi_exact'], color='k', lw=2.4, alpha=0.35, label='exact geodesic')
    ax[0].plot(tc['t'], tc['dchi_nn'], color=COLORS[0], lw=1.2, label='network')
    ax[0].set_ylabel(r'$\chi-\chi_{\rm Kepler}$ [rad]'); ax[0].legend()
    ax[1].plot(tc['t'], tc['dchi_nn'] - tc['dchi_exact'], color=COLORS[1], lw=1)
    ax[1].set_ylabel('network - exact\n[rad]')
    ax[2].semilogy(tc['t'], tc['pos_kepler_minus_exact'] + 1e-12, color='0.4',
                   label='timing effect: |Kepler - exact|')
    ax[2].semilogy(tc['t'], tc['pos_nn_minus_exact'] + 1e-12, color=COLORS[1],
                   label='network error: |GINN - exact|')
    if data is not None:
        ax[2].axhline(float(data.ra_e.median()), color='k', ls=':', label='median astrometric error')
    ax[2].set_ylabel('sky position\ndifference [mas]'); ax[2].set_xlabel('epoch [yr]')
    ax[2].legend(fontsize=8, loc='lower right')
    return fig


def plot_profile(prof, prior=None, reference=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(prof['values'], prof['dchi2'], 'o', color=COLORS[0])
    ax[0].plot(prof['fine'], np.interp(prof['fine'], prof['values'], prof['dchi2']),
               color=COLORS[0], lw=1)
    for lev, ls in [(1, '--'), (4, ':')]:
        ax[0].axhline(lev, color='k', ls=ls, lw=0.8)
    ax[0].axvline(0, color='0.5', lw=0.8)
    ax[0].set_xlabel(prof['name']); ax[0].set_ylabel(r'$\Delta\chi^2_{\rm profile}$')
    ax[0].set_ylim(-0.2, max(6, np.nanmax(prof['dchi2']) * 0.3))
    ax[1].plot(prof['fine'], prof['posterior'], color=COLORS[0], label='profile x prior')
    ax[1].axvspan(*prof['cred68'], color=COLORS[0], alpha=0.2, label='68% credible')
    ax[1].axvline(prof['upper95'], color=COLORS[1], ls='--', label='95% upper')
    if reference is not None:
        ax[1].axvline(reference, color='k', ls=':', label='reference')
    ax[1].set_xlabel(prof['name']); ax[1].set_ylabel('posterior density')
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    return fig


def _mark(ax, hist):
    for s in hist.get('precondition_steps', [])[1:]:
        ax.axvline(s, color='0.6', ls=':', lw=1)


def plot_loss_history(hist, title=''):

    data = np.array(hist['data']); phys = np.array(hist['phys'], dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].plot(data); ax[0].set_yscale('log')
    ax[0].set_title(r'data loss $\chi^2_{\rm data}$' + f'   (final {data[-1]:.3f})')
    if np.isfinite(phys).any():
        ax[1].plot(phys); ax[1].set_yscale('log')
        ax[1].set_title(r'physics loss $\sum_c (R_c/\epsilon)^2$' + f'   (final {phys[-1]:.3f})')
        ax[2].plot(hist['phys_rms'], label=r'rms $R$ (relative geodesic residual)')
        ax[2].plot(hist['eps'], 'k--', lw=1, label=r'tolerance $\epsilon$')
        ax[2].set_yscale('log'); ax[2].legend(); ax[2].set_title('physics residual vs tolerance')
    else:
        for a in ax[1:]:
            a.text(0.5, 0.5, 'no physics term\n(timing = kepler / exact)', ha='center',
                   va='center', transform=a.transAxes)
    for a in ax:
        a.set_xlabel('step'); a.grid(True, alpha=0.3); _mark(a, hist)
    fig.suptitle(title); plt.tight_layout()
    return fig


BASE_HIST = ('M', 'a', 'e', 'inc', 'Omega', 'omega', 't0', 'D', 'vz0', 'x0', 'y0', 'vx0', 'vy0')


def plot_param_history(hist, lit=None, title=''):
    P = {k: np.array(v) for k, v in hist['params'].items()}
    P['p'] = P['a'] * (1 - P['e'] ** 2)
    for k in ('inc', 'Omega', 'omega'):
        P[k] = P[k] / DEG
    P['M'] = P['M'] / 1e6
    rows = [[('e', 'e'), ('M', r'M [$10^6 M_\odot$]'), ('a', 'a [AU]'), ('p', 'p [AU]'), ('D', r'$R_0$ [kpc]')],
            [('inc', 'i [deg]'), ('Omega', r'$\Omega$ [deg]'), ('omega', r'$\omega$ [deg]'),
             ('t0', r'$t_p$ [yr]'), ('vz0', r'$v_{z,0}$ [km/s]')],
            [('x0', r'$x_0$ [mas]'), ('y0', r'$y_0$ [mas]'), ('vx0', r'$v_{x,0}$ [mas/yr]'),
             ('vy0', r'$v_{y,0}$ [mas/yr]')]]
    hair = [(h, h) for h in hist['params'] if h not in BASE_HIST]
    rows[2] += hair[:1]
    if len(hair) > 1:
        rows.append(hair[1:])
    lit = lit or {}
    figs = []
    for row in rows:
        fig, axes = plt.subplots(ncols=5, figsize=(18, 3.4))
        for ax, (k, lab) in zip(axes, row):
            ax.plot(P[k]); ax.set_ylabel(lab); ax.set_xlabel('step')
            ax.grid(True, alpha=0.3); _mark(ax, hist)
            ax.set_title(f'final {P[k][-1]:.6g}', fontsize=9)
            if k in lit:
                ax.axhline(lit[k], color='r', ls='--', alpha=0.6, label=f'lit: {lit[k]}')
                ax.legend(fontsize=8)
        for ax in axes[len(row):]:
            ax.axis('off')
        if title and not figs:
            fig.suptitle(title)
        plt.tight_layout(); figs.append(fig)
    return figs


def plot_convergence(hist, samples, gn_hist=None, title=''):
    """All parameters in one plot: distance to the final value in units of the final 1-sigma error."""
    keys = [k for k in hist['params'] if k in samples]
    scale = {'inc': 1 / DEG, 'Omega': 1 / DEG, 'omega': 1 / DEG, 'M': 1e-6}
    lab = {'M': 'M', 'a': 'a', 'e': 'e', 'inc': 'i', 'Omega': r'$\Omega$', 'omega': r'$\omega$', 't0': r'$t_p$',
           'D': r'$R_0$', 'vz0': r'$v_{z,0}$', 'x0': r'$x_0$', 'y0': r'$y_0$', 'vx0': r'$v_{x,0}$', 'vy0': r'$v_{y,0}$'}
    ncol = 2 if gn_hist else 1
    fig, ax = plt.subplots(1, ncol, figsize=(15 if gn_hist else 10, 4.5),
                           gridspec_kw=dict(width_ratios=[3, 1]) if gn_hist else None)
    ax = np.atleast_1d(ax)
    cmap = plt.get_cmap('tab20')
    styles = ['-', '--', '-.', ':']
    for i, k in enumerate(keys):
        med = np.median(samples[k])
        sig = 0.5 * (np.percentile(samples[k], 84.135) - np.percentile(samples[k], 15.865))
        x = np.array(hist['params'][k]) * scale.get(k, 1.0)
        if k == 'a' and 'D' in hist['params']:
            x = np.array(hist['params']['a']) / np.array(hist['params']['D'])   # mas, as in the samples
        ax[0].semilogy(np.abs(x - med) / sig + 1e-3, color=cmap(i % 20), ls=styles[i % 4], lw=1.2,
                       label=lab.get(k, k))
        if gn_hist and k in gn_hist[0]:
            g = np.array([h[k] for h in gn_hist])
            ax[1].semilogy(np.abs(g - med) / sig + 1e-3, 'o-', color=cmap(i % 20), label=lab.get(k, k))
    ax[0].axhline(1, color='k', lw=0.8)
    ax[0].axhline(0.1, color='k', lw=0.8, ls=':')
    ax[0].set_xlabel('Adam step'); ax[0].set_ylabel(r'$|x - x_{\rm final}|\,/\,\sigma$')
    ax[0].set_ylim(1e-3, None)
    ax[0].legend(ncol=5, fontsize=8, loc='upper right')
    if gn_hist:
        ax[1].axhline(1, color='k', lw=0.8); ax[1].axhline(0.1, color='k', lw=0.8, ls=':')
        ax[1].set_xlabel('Gauss-Newton step'); ax[1].set_ylim(1e-3, None)
        ax[1].legend(fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    return fig
