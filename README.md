# SABR Swaptions

SABR calibration and pricing for interest rate swaptions.

## What it does

Calibrates SABR to swaption volatility smiles slice by slice with beta fixed at 0.5, validates the
implementation against one day of real VCUB quotes, prices with Black-76 under an annuity built
from the SOFR curve of the same date, and computes vega, gamma and vanna by finite difference. On
the included snapshot every one of the 91 smiles fits to under 5bp, worst case 1.22bp. Free
swaption vol data is close to nonexistent, so the notebook has exactly one day. Slices off that
grid are built from a FRED derived ATM level with a synthetic smile, or generated outright, and
each calibration result carries a source tag saying which it is.

## Files

`sabr_swaption.ipynb` is the walkthrough: synthetic warm up, VCUB validation, fallback data,
pricing, summary table.
`sabr_swaption.py` holds the model, calibration, pricing, Greeks, curve and data loading code that
the notebook imports.
`requirements.txt` lists the dependencies.
`data/vcub_snapshot.csv` is the VCUB snapshot, 819 quotes, with the column layout documented in its
header comment.
`data/sofr_curve.csv` is the SOFR par curve for the same date, used for discounting.
`LICENSE` is MIT.

## How to run it

```bash
pip install -r requirements.txt
jupyter notebook sabr_swaption.ipynb
```

Run the cells in order. The VCUB snapshot is a single day covering 13 expiries by 7 tenors, so
slices off that grid fall back to FRED or to synthetic data, labelled in every table and plot title.
Only the fallback path needs network access. Vols and discount curve both ship with the repo, so
validation and the VCUB price run offline.

## Data

Swaption vol surfaces are not freely available. What is here is one business day, USD SOFR mid for
2026-07-28, pulled from Bloomberg VCUB on the OTM Swaptions/SABR tab as absolute normal vol: 91
smiles of nine strikes each, 819 quotes, with forwards and a discount curve from IRSB for the same
date. Quotes are normal (Bachelier) vol in decimal rate units, which is why the module carries both
a normal and a lognormal Hagan expansion. The loader also accepts a per slice calibrated parameter
file, in which case the smile is reconstructed from those parameters and the validation section
reports itself as a self consistency check rather than a market fit test. Slices with no VCUB data
are tagged FRED or synthetic and are never shown as market observed.

## Limitations

There is no shift term, so the model is unusable at or below zero rates. Each expiry and tenor is
calibrated independently, so fitting every slice to under 5bp does not make the surface arbitrage
free across expiries, and nothing here checks calendar or butterfly conditions on the fitted grid.
Hagan's expansion degrades at very long expiries and far strikes. Discounting uses a single SOFR
curve with no basis or multi-curve adjustment, and swaps running past its last pillar at 30Y rely
on flat forward extrapolation, so those annuities are indicative rather than desk numbers. One day
of data validates the method and says nothing about parameter stability over time.

## License

MIT.
