# SABR Swaptions

SABR calibration and pricing for interest rate swaptions.

## What it does

Calibrates SABR to swaption volatility smiles slice by slice with beta fixed at 0.5, validates the
implementation against a one day VCUB snapshot, prices with Black-76 under the annuity measure
using a discount curve bootstrapped from FRED par rates, and computes vega, gamma and vanna by
finite difference. Free swaption vol data is close to nonexistent, so the notebook only has one day
of real quotes. Every other slice is built from a FRED derived ATM level with a synthetic smile, or
generated outright, and each calibration result carries a source tag that labels which it is.

## Files

`sabr_swaption.ipynb` is the walkthrough: synthetic warm up, VCUB validation, fallback data,
pricing, summary table.
`sabr_swaption.py` holds the model, calibration, pricing, Greeks, curve and data loading code that
the notebook imports.
`requirements.txt` lists the dependencies.
`data/vcub_snapshot.csv` is the VCUB snapshot, with the expected column layout documented in its
header comment.
`LICENSE` is MIT.

## How to run it

```bash
pip install -r requirements.txt
jupyter notebook sabr_swaption.ipynb
```

Run the cells in order. The VCUB snapshot is a single day, so slices outside it fall back to FRED
or to synthetic data, labelled in every table and plot title. The FRED calls need network access.
Without it the notebook still runs and prints an approximate annuity instead of a bootstrapped one.

## Data

Swaption vol surfaces are not freely available. The snapshot here is one business day of USD VCUB,
stored as per slice SABR parameters in normal vol with the arbitrage diagnostics from that
calibration, rather than as raw strike quotes. The loader accepts both layouts, and with raw quotes
the validation section is a true market fit test. With the parameter file it is a check that the
implementation recovers known parameters and reproduces the quoted ATM vol, which the notebook
states plainly where the result is reported. Slices with no VCUB data are tagged FRED or synthetic
and are never shown as market observed.

## Limitations

There is no shift term, so the model is unusable at or below zero rates. Each expiry and tenor is
calibrated independently, so the surface is not arbitrage free across expiries, and the snapshot's
own diagnostics already flag negative implied density at long expiries where nu squared times T is
large. Hagan's expansion degrades at very long expiries and far strikes. The discount curve is
bootstrapped from Treasury par rates on a single semi-annual grid with no basis or multi-curve
adjustment, so the annuity is indicative rather than a desk number. One day of data validates the
method and says nothing about parameter stability over time.

## License

MIT.
