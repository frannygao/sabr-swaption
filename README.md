# sabr-swaption

SABR calibration and swaption pricing on real VCUB data.

## What it does

Calibrates SABR to swaption volatility smiles slice by slice with beta fixed at 0.5, validates the implementation against one day of real VCUB quotes, prices with Black-76 under an annuity built from the SOFR curve of the same date, and computes vega, gamma and vanna by finite difference. 

## Calibration

SABR dynamics, with beta fixed at 0.5:

    dF     = alpha * F^beta * dW1
    dalpha = nu * alpha * dW2
    corr(dW1, dW2) = rho

Hagan lognormal implied vol, non-ATM:

    sigma(F, K) = alpha / ( (FK)^((1-beta)/2) * (1 + ((1-beta)^2/24) * ln(F/K)^2 + ((1-beta)^4/1920) * ln(F/K)^4 ) )
                * z / x(z)
                * ( 1 + ( ((1-beta)^2/24) * alpha^2 / (FK)^(1-beta)
                         + (rho * beta * nu * alpha) / (4 * (FK)^((1-beta)/2))
                         + ((2 - 3*rho^2)/24) * nu^2 ) * T )

with

    z    = (nu / alpha) * (FK)^((1-beta)/2) * ln(F/K)
    x(z) = ln( ( sqrt(1 - 2*rho*z + z^2) + z - rho ) / (1 - rho) )

ATM case:

    sigma_ATM = alpha / F^(1-beta)
              * ( 1 + ( ((1-beta)^2/24) * alpha^2 / F^(2-2*beta)
                      + (rho * beta * nu * alpha) / (4 * F^(1-beta))
                      + ((2 - 3*rho^2)/24) * nu^2 ) * T )

Calibration objective, minimized over (alpha, rho, nu) with beta fixed:

    min_{alpha, rho, nu}  sum_i  w_i * ( sigma_SABR(K_i) - sigma_market(K_i) )^2

Fit quality:

    RMSE   = sqrt( (1/N) * sum_i ( sigma_SABR(K_i) - sigma_market(K_i) )^2 )
    MaxErr = max_i | sigma_SABR(K_i) - sigma_market(K_i) |

## Pricing

Forward swap rate from the discount curve:

    S = ( P(T_start) - P(T_end) ) / sum_j tau_j * P(T_j)

Annuity:

    A = sum_j tau_j * P(T_j)

Black-76 payer swaption price under the annuity measure:

    V_payer    = A * ( S * N(d1) - K * N(d2) )
    V_receiver = A * ( K * N(-d2) - S * N(-d1) )

with

    d1 = ( ln(S/K) + 0.5 * sigma^2 * T ) / ( sigma * sqrt(T) )
    d2 = d1 - sigma * sqrt(T)

## Greeks

Computed by finite difference:

    vega  = dV/dsigma       ~  ( V(sigma + h) - V(sigma - h) ) / (2h)
    gamma = d^2V/dS^2       ~  ( V(S + h) - 2*V(S) + V(S - h) ) / h^2
    vanna = d^2V/dS dsigma  ~  ( vega(S + h) - vega(S - h) ) / (2h)


## Running it

Install dependencies, open the notebook, run cells in order. It runs on the included VCUB snapshot only, one day, 2026-07-28, 91 expiry/tenor slices. There is no fallback source: a slice outside that grid is not priced, and a missing snapshot or curve file raises an error instead of substituting anything.
