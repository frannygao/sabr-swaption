"""sabr calibration, pricing and greeks for interest rate swaptions"""

import datetime as dt
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

DARK_THEME = {
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'grid.color': '#2a2a4a',
    'legend.facecolor': '#16213e',
    'legend.edgecolor': '#e0e0e0',
    'figure.figsize': (12, 6),
    'font.size': 11,
}

DEFAULT_OFFSETS_BP = np.array([-200, -100, -50, -25, 0, 25, 50, 100, 200])


# hagan implied vol

def hagan_implied_vol(strike, forward, expiry, alpha, beta, nu, rho):
    """hagan lognormal (black) implied vol"""
    K = np.asarray(strike, dtype=float)
    F = float(forward)
    T = float(expiry)
    eps = 1e-12

    FK = F * K
    FK_mid = np.sqrt(FK)
    log_FK = np.log(F / np.maximum(K, eps))

    one_minus_beta = 1.0 - beta
    FK_pow = FK_mid ** one_minus_beta

    atm_mask = np.abs(log_FK) < 1e-7

    z = (nu / alpha) * FK_pow * log_FK
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho + eps) + eps)
    zx_ratio = np.where(atm_mask, 1.0, z / (x + eps))

    denom_term = (one_minus_beta * log_FK)**2
    denom = FK_pow * (1 + denom_term / 24 + denom_term**2 / 1920)

    term1 = (one_minus_beta * alpha)**2 / (24 * FK_mid**(2 * one_minus_beta))
    term2 = 0.25 * rho * beta * nu * alpha / FK_pow
    term3 = (2 - 3 * rho**2) * nu**2 / 24
    numerator = 1 + (term1 + term2 + term3) * T

    sigma = (alpha / denom) * zx_ratio * numerator
    return np.maximum(sigma, eps)


def hagan_normal_vol(strike, forward, expiry, alpha, beta, nu, rho):
    """hagan normal (bachelier) implied vol, in rate units"""
    K = np.asarray(strike, dtype=float)
    F = float(forward)
    T = float(expiry)
    eps = 1e-12

    FK_mid = np.sqrt(np.maximum(F * K, eps))
    log_FK = np.log(F / np.maximum(K, eps))
    one_minus_beta = 1.0 - beta

    atm_mask = np.abs(log_FK) < 1e-7

    z = (nu / alpha) * FK_mid ** one_minus_beta * log_FK
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho + eps) + eps)
    zx_ratio = np.where(atm_mask, 1.0, z / (x + eps))

    denom = 1 + (one_minus_beta * log_FK)**2 / 24 + (one_minus_beta * log_FK)**4 / 1920

    term1 = -beta * (2 - beta) * alpha**2 / (24 * FK_mid ** (2 * one_minus_beta))
    term2 = 0.25 * rho * beta * nu * alpha / FK_mid ** one_minus_beta
    term3 = (2 - 3 * rho**2) * nu**2 / 24
    numerator = 1 + (term1 + term2 + term3) * T

    sigma = alpha * FK_mid ** beta * (zx_ratio / denom) * numerator
    return np.maximum(sigma, eps)


VOL_MODELS = {'lognormal': hagan_implied_vol, 'normal': hagan_normal_vol}


# calibration

def sabr_objective(params, strikes, forward, expiry, market_vols, beta, vol_model, scale):
    """squared vol error, scaled to keep the objective away from optimiser tolerances"""
    alpha, nu, rho = params
    if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
        return 1e10
    model_vols = VOL_MODELS[vol_model](strikes, forward, expiry, alpha, beta, nu, rho)
    return float(np.sum(((model_vols - market_vols) / scale) ** 2))


def calibrate_sabr(strikes, forward, expiry, market_vols, beta=0.5, vol_model='lognormal'):
    """fit alpha, nu, rho to a single smile with beta fixed

    residuals are normalised by the mean market vol, so normal and lognormal quotes
    present the optimiser with the same scale. seeds are tried from a small grid
    because a single seed stalls on long expiries.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    scale = float(np.mean(market_vols))

    atm_vol_guess = float(np.interp(forward, strikes, market_vols))
    if vol_model == 'normal':
        alpha0 = atm_vol_guess / forward ** beta
    else:
        alpha0 = atm_vol_guess * forward ** (1 - beta)

    bounds = [(1e-6, 1.0), (1e-4, 5.0), (-0.999, 0.999)]
    seeds = [(alpha0, nu0, rho0)
             for nu0 in (0.15, 0.4, 1.0, 2.0)
             for rho0 in (-0.3, 0.0, 0.3)]

    best = None
    for x0 in seeds:
        result = minimize(
            sabr_objective, x0,
            args=(strikes, forward, expiry, market_vols, beta, vol_model, scale),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12},
        )
        if best is None or result.fun < best.fun:
            best = result

    alpha_cal, nu_cal, rho_cal = best.x
    model_vols = VOL_MODELS[vol_model](strikes, forward, expiry, alpha_cal, beta, nu_cal, rho_cal)
    errors = model_vols - market_vols

    return {
        'alpha': alpha_cal, 'beta': beta, 'nu': nu_cal, 'rho': rho_cal,
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'max_err': float(np.max(np.abs(errors))),
        'model_vols': model_vols, 'converged': bool(best.success),
        'vol_model': vol_model,
    }


# pricing

def black_price(forward, strike, expiry, vol, is_payer=True):
    """undiscounted black-76 price, one unit of annuity"""
    if vol <= 0 or expiry <= 0:
        return max(forward - strike, 0) if is_payer else max(strike - forward, 0)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * expiry) / (vol * np.sqrt(expiry))
    d2 = d1 - vol * np.sqrt(expiry)
    if is_payer:
        return forward * norm.cdf(d1) - strike * norm.cdf(d2)
    return strike * norm.cdf(-d2) - forward * norm.cdf(-d1)


def bachelier_price(forward, strike, expiry, vol, is_payer=True):
    """undiscounted normal-model price, one unit of annuity"""
    if vol <= 0 or expiry <= 0:
        return max(forward - strike, 0) if is_payer else max(strike - forward, 0)
    sign = 1.0 if is_payer else -1.0
    d = sign * (forward - strike) / (vol * np.sqrt(expiry))
    return sign * (forward - strike) * norm.cdf(d) + vol * np.sqrt(expiry) * norm.pdf(d)


def sabr_price(forward, strike, expiry, alpha, beta, nu, rho, is_payer=True):
    vol = hagan_implied_vol(strike, forward, expiry, alpha, beta, nu, rho)
    return black_price(forward, strike, expiry, float(vol), is_payer)


def compute_greeks(forward, strike, expiry, alpha, beta, nu, rho, is_payer=True):
    """vega per 1bp of alpha, gamma and vanna by finite difference"""
    dF = forward * 1e-4
    dVol = 1e-4

    price_base = sabr_price(forward, strike, expiry, alpha, beta, nu, rho, is_payer)

    p_up_v = sabr_price(forward, strike, expiry, alpha + dVol, beta, nu, rho, is_payer)
    p_dn_v = sabr_price(forward, strike, expiry, alpha - dVol, beta, nu, rho, is_payer)
    vega = (p_up_v - p_dn_v) / (2 * dVol) * 0.0001

    p_up_F = sabr_price(forward + dF, strike, expiry, alpha, beta, nu, rho, is_payer)
    p_dn_F = sabr_price(forward - dF, strike, expiry, alpha, beta, nu, rho, is_payer)
    gamma = (p_up_F - 2 * price_base + p_dn_F) / (dF ** 2)

    vega_up = (sabr_price(forward + dF, strike, expiry, alpha + dVol, beta, nu, rho, is_payer)
               - sabr_price(forward + dF, strike, expiry, alpha - dVol, beta, nu, rho, is_payer)) / (2 * dVol)
    vega_dn = (sabr_price(forward - dF, strike, expiry, alpha + dVol, beta, nu, rho, is_payer)
               - sabr_price(forward - dF, strike, expiry, alpha - dVol, beta, nu, rho, is_payer)) / (2 * dVol)
    vanna = (vega_up - vega_dn) / (2 * dF)

    return {'price': price_base, 'vega': vega, 'gamma': gamma, 'vanna': vanna}


# discount curve and annuity

FRED_PAR_SERIES = {
    1: 'DGS1', 2: 'DGS2', 3: 'DGS3', 5: 'DGS5',
    7: 'DGS7', 10: 'DGS10', 20: 'DGS20', 30: 'DGS30',
}


def fetch_fred_par_curve(lookback_days=30):
    """latest par rates from fred, as {maturity_years: rate}. returns {} on failure"""
    try:
        import pandas_datareader.data as pdr
    except ImportError:
        print('[curve] pandas-datareader not installed')
        return {}

    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    par = {}
    for maturity, series_id in FRED_PAR_SERIES.items():
        try:
            data = pdr.DataReader(series_id, 'fred', start, end).dropna()
            if len(data):
                par[maturity] = float(data.iloc[-1, 0]) / 100
        except Exception:
            continue
    if par:
        print(f'[curve] fred par rates: {len(par)} points, '
              f'{min(par)}Y={par[min(par)]*100:.2f}% to {max(par)}Y={par[max(par)]*100:.2f}%')
    else:
        print('[curve] fred unavailable')
    return par


def load_par_curve(path='data/sofr_curve.csv'):
    """par swap curve from file as {maturity_years: decimal rate}. empty if missing"""
    if not os.path.exists(path):
        print(f'[curve] no curve at {path}')
        return {}
    frame = pd.read_csv(path, comment='#')
    par = {float(r['tenor_years']): float(r['rate_pct']) / 100 for _, r in frame.iterrows()}
    print(f'[curve] loaded {len(par)} par pillars from {path}, '
          f'{min(par)}Y={par[min(par)]*100:.3f}% to {max(par)}Y={par[max(par)]*100:.3f}%')
    return par


def bootstrap_discount_curve(par_rates, freq=2, day_count=360.0, basis_days=365.0):
    """bootstrap discount factors from par swap rates

    pillars below the first coupon date are treated as simple-compounded deposits,
    the rest are bootstrapped on a semi-annual fixed leg. returns (times, dfs).
    """
    if not par_rates:
        return None

    step = 1.0 / freq
    accrual = (basis_days / freq) / day_count
    pillars = sorted(par_rates)

    short = [(t, 1.0 / (1.0 + par_rates[t] * t)) for t in pillars if t < step]

    times, dfs = [0.0], [1.0]
    annuity = 0.0
    n_steps = int(round(max(pillars) * freq))
    for i in range(1, n_steps + 1):
        t = i * step
        par = float(np.interp(t, pillars, [par_rates[m] for m in pillars]))
        df = (1.0 - par * annuity) / (1.0 + par * accrual)
        if df <= 0:
            break
        annuity += accrual * df
        times.append(t)
        dfs.append(df)

    merged = sorted(set(short + list(zip(times, dfs))))
    times = np.array([t for t, _ in merged])
    dfs = np.array([d for _, d in merged])
    if len(times) < 3:
        return None
    return times, dfs


def discount_factor(curve, t):
    """log-linear interpolation on the bootstrapped curve

    beyond the last pillar the instantaneous forward of the final segment is held
    flat, so long-dated swaps extrapolate instead of clamping to a constant df.
    """
    times, dfs = curve
    t = np.asarray(t, dtype=float)
    log_dfs = np.log(dfs)
    log_df = np.interp(t, times, log_dfs)

    slope = (log_dfs[-1] - log_dfs[-2]) / (times[-1] - times[-2])
    beyond = t > times[-1]
    if np.any(beyond):
        log_df = np.where(beyond, log_dfs[-1] + (t - times[-1]) * slope, log_df)
    return np.exp(log_df)


def curve_max_time(curve):
    return float(curve[0][-1])


def swap_schedule(expiry_years, tenor_years, freq=2):
    """fixed-leg payment times measured from today"""
    n = int(round(tenor_years * freq))
    step = 1.0 / freq
    return np.array([expiry_years + (i + 1) * step for i in range(n)])


def annuity_from_curve(curve, expiry_years, tenor_years, freq=2,
                       day_count=360.0, basis_days=365.0):
    """act/360 day-count weighted sum of discount factors on the fixed leg"""
    pay_times = swap_schedule(expiry_years, tenor_years, freq)
    accrual = (basis_days / freq) / day_count
    return float(np.sum(accrual * discount_factor(curve, pay_times)))


def forward_swap_rate(curve, expiry_years, tenor_years, freq=2,
                      day_count=360.0, basis_days=365.0):
    """par forward swap rate implied by the curve"""
    df_start = float(discount_factor(curve, expiry_years))
    df_end = float(discount_factor(curve, expiry_years + tenor_years))
    ann = annuity_from_curve(curve, expiry_years, tenor_years, freq, day_count, basis_days)
    return (df_start - df_end) / ann


def flat_annuity(forward, tenor_years, freq=2):
    """fallback annuity under a flat discount rate, not a market curve"""
    n = int(round(tenor_years * freq))
    step = 1.0 / freq
    return float(np.sum([step / (1 + forward * step) ** (i + 1) for i in range(n)]))


def price_swaption(forward, strike, expiry_years, tenor_years, alpha, beta, nu, rho,
                   curve=None, notional=1.0, is_payer=True):
    """black-76 swaption price under the annuity measure

    uses the bootstrapped curve when given, otherwise a flat-discount annuity.
    """
    vol = float(hagan_implied_vol(strike, forward, expiry_years, alpha, beta, nu, rho))
    unit = black_price(forward, strike, expiry_years, vol, is_payer)
    if curve is not None:
        ann = annuity_from_curve(curve, expiry_years, tenor_years)
        annuity_label = 'bootstrapped curve'
    else:
        ann = flat_annuity(forward, tenor_years)
        annuity_label = 'approximate annuity (flat discount)'
    return {
        'vol': vol, 'unit_price': unit, 'annuity': ann,
        'price': unit * ann * notional, 'annuity_label': annuity_label,
    }


# data

def build_smile_from_atm(atm_forward, atm_vol, strike_offsets_bp=None,
                         skew_strength=-0.3, curvature_strength=0.8):
    """synthetic smile shape around a given atm level"""
    if strike_offsets_bp is None:
        strike_offsets_bp = DEFAULT_OFFSETS_BP
    strikes = atm_forward + np.asarray(strike_offsets_bp) / 10000
    moneyness = (strikes - atm_forward) / atm_forward
    skew = skew_strength * moneyness * atm_vol
    curvature = curvature_strength * moneyness**2 * atm_vol
    implied_vols = np.maximum(atm_vol + skew + curvature, 0.005)
    return strikes, implied_vols


def generate_sample_vol_surface(expiries=(1, 2, 5, 10), tenors=(2, 5, 10),
                                strike_offsets_bp=None):
    """fully synthetic lognormal vol surface"""
    if strike_offsets_bp is None:
        strike_offsets_bp = DEFAULT_OFFSETS_BP
    surface = {}
    for expiry in expiries:
        for tenor in tenors:
            atm_forward = 0.03 + 0.002 * tenor + 0.001 * expiry
            strikes = atm_forward + strike_offsets_bp / 10000

            atm_vol = 0.20 + 0.05 * np.exp(-0.1 * expiry) + 0.01 * tenor
            skew = -0.0015 * (strikes - atm_forward) / atm_forward
            curvature = 0.8 * ((strikes - atm_forward) / atm_forward) ** 2
            noise = np.random.RandomState(42 + expiry * 10 + tenor).normal(0, 0.002, len(strikes))

            implied_vols = np.maximum(atm_vol + skew + curvature + noise, 0.01)

            surface[(expiry, tenor)] = {
                'strikes': strikes, 'atm_forward': atm_forward,
                'implied_vols': implied_vols, 'strike_offsets_bp': strike_offsets_bp,
                'source': 'synthetic', 'vol_model': 'lognormal',
            }
    return surface, list(expiries), list(tenors)


def parse_term(label):
    """'3Mo' or '10Yr' to years"""
    text = str(label).strip()
    number = float(''.join(c for c in text if c.isdigit() or c == '.'))
    unit = ''.join(c for c in text if c.isalpha()).lower()
    if unit.startswith('m'):
        return number / 12.0
    if unit.startswith('d'):
        return number / 365.0
    if unit.startswith('w'):
        return number / 52.0
    return number


def _read_header_date(path):
    with open(path) as handle:
        for line in handle:
            if not line.startswith('#'):
                break
            if 'date:' in line:
                return line.split('date:', 1)[1].strip()
    return 'unknown'


def load_vcub_snapshot(path='data/vcub_snapshot.csv', strike_offsets_bp=None):
    """load a vcub snapshot keyed by (expiry, tenor)

    accepts a raw-quote file (expiry, tenor, strike, implied_vol, atm_forward, date)
    or a calibrated-parameter file (expiry, tenor, F_pct, alpha, rho, nu, ...).
    returns an empty dict if the file is missing.
    """
    if strike_offsets_bp is None:
        strike_offsets_bp = DEFAULT_OFFSETS_BP

    if not os.path.exists(path):
        print(f'[vcub] no snapshot at {path}, falling back to synthetic data')
        return {}

    frame = pd.read_csv(path, comment='#')
    snapshot_date = _read_header_date(path)
    columns = set(frame.columns)
    snapshot = {}

    if {'strike', 'implied_vol'} <= columns:
        # raw market quotes, used directly
        for (expiry, tenor), group in frame.groupby(['expiry', 'tenor'], sort=False):
            group = group.sort_values('strike')
            atm_forward = float(group['atm_forward'].iloc[0])
            date = str(group['date'].iloc[0]) if 'date' in columns else snapshot_date
            vol_model = str(group['vol_model'].iloc[0]) if 'vol_model' in columns else 'lognormal'
            snapshot[(str(expiry), str(tenor))] = {
                'strikes': group['strike'].to_numpy(float),
                'implied_vols': group['implied_vol'].to_numpy(float),
                'atm_forward': atm_forward,
                'strike_offsets_bp': (group['strike'].to_numpy(float) - atm_forward) * 10000,
                'date': date, 'source': 'VCUB', 'quote_kind': 'raw market quotes',
                'vol_model': vol_model,
                'expiry_years': parse_term(expiry), 'tenor_years': parse_term(tenor),
            }
        n_quotes = len(frame)
        print(f'[vcub] loaded {len(snapshot)} smiles, {n_quotes} raw quotes, date {snapshot_date}')
        return snapshot

    if {'alpha', 'nu', 'rho', 'F_pct'} <= columns:
        # calibrated parameters, smile reconstructed from them
        for _, row in frame.iterrows():
            if 'status' in columns and str(row['status']).lower() != 'ok':
                continue
            expiry, tenor = str(row['expiry']), str(row['tenor'])
            expiry_years = parse_term(expiry)
            atm_forward = float(row['F_pct']) / 100
            strikes = atm_forward + np.asarray(strike_offsets_bp) / 10000
            vols = hagan_normal_vol(strikes, atm_forward, expiry_years,
                                    float(row['alpha']), 0.5, float(row['nu']), float(row['rho']))
            snapshot[(expiry, tenor)] = {
                'strikes': strikes, 'implied_vols': np.asarray(vols, float),
                'atm_forward': atm_forward, 'strike_offsets_bp': np.asarray(strike_offsets_bp),
                'date': snapshot_date, 'source': 'VCUB',
                'quote_kind': 'reconstructed from vcub sabr parameters',
                'vol_model': 'normal',
                'expiry_years': expiry_years, 'tenor_years': parse_term(tenor),
                'ref_alpha': float(row['alpha']), 'ref_nu': float(row['nu']),
                'ref_rho': float(row['rho']),
                'ref_rmse_bp': float(row.get('rmse_bp', np.nan)),
                'ref_atm_vol_bp': float(row.get('atm_vol_bp', np.nan)),
                'density_viol': int(row.get('density_viol', 0)),
                'any_flag': bool(row.get('any_flag', False)),
            }
        print(f'[vcub] loaded {len(snapshot)} slices of calibrated parameters, date {snapshot_date}')
        print('[vcub] smiles are reconstructed from vcub sabr parameters, not raw strike quotes')
        return snapshot

    print(f'[vcub] unrecognised columns in {path}, falling back to synthetic data')
    return {}


def fred_atm_level(lookback_days=500):
    """latest 10y rate and its historical vol from fred, as a fallback atm proxy"""
    try:
        import pandas_datareader.data as pdr
    except ImportError:
        print('[fred] pandas-datareader not installed')
        return None

    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    for series_id, label in [('DSWP10', '10y swap rate'), ('DGS10', '10y treasury')]:
        try:
            data = pdr.DataReader(series_id, 'fred', start, end).dropna()
        except Exception:
            continue
        if len(data) <= 50:
            continue
        rates = data.iloc[:, 0].to_numpy(float) / 100
        rates = rates[rates > 0]
        hist_vol = float(np.std(np.diff(np.log(rates))) * np.sqrt(252))
        print(f'[fred] {label}: {rates[-1]*100:.2f}%, historical vol {hist_vol*100:.1f}%')
        return {'rate': float(rates[-1]), 'hist_vol': hist_vol, 'series': series_id}

    print('[fred] no usable series')
    return None


def build_fallback_slice(expiry_years, tenor_years, fred_level=None,
                         strike_offsets_bp=None):
    """derived or synthetic slice for an (expiry, tenor) with no vcub quotes"""
    if strike_offsets_bp is None:
        strike_offsets_bp = DEFAULT_OFFSETS_BP

    if fred_level is not None:
        atm_forward = fred_level['rate'] + 0.001 * expiry_years
        atm_vol = fred_level['hist_vol'] * (1 + 0.1 * np.exp(-0.15 * expiry_years))
        source = 'FRED'
    else:
        atm_forward = 0.03 + 0.002 * tenor_years + 0.001 * expiry_years
        atm_vol = 0.20 + 0.05 * np.exp(-0.1 * expiry_years) + 0.01 * tenor_years
        source = 'synthetic'

    strikes, vols = build_smile_from_atm(atm_forward, atm_vol, strike_offsets_bp)
    return {
        'strikes': strikes, 'implied_vols': vols, 'atm_forward': atm_forward,
        'strike_offsets_bp': np.asarray(strike_offsets_bp),
        'source': source, 'vol_model': 'lognormal',
        'quote_kind': 'real atm level, synthetic smile' if source == 'FRED' else 'fully synthetic',
        'date': 'derived', 'expiry_years': expiry_years, 'tenor_years': tenor_years,
    }


SOURCE_LABELS = {
    'VCUB': 'VCUB (market observed)',
    'FRED': 'FRED ATM level, synthetic smile (derived)',
    'synthetic': 'synthetic (not market data)',
}


def source_label(source):
    return SOURCE_LABELS.get(source, source)
