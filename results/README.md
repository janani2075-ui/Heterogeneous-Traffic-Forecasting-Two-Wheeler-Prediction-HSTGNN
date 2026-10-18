# HSTGNN Experimental Results

This directory documents the experimental findings from my
MSc Data Science dissertation on heterogeneous traffic forecasting.

## Forecasting Performance

The research evaluates traffic prediction across 12 forecasting
horizons using the PEMS08 dataset.

| Evaluation Metric | Three-Feature Model | Four-Feature Model |
|:---|---:|---:|
| MAE | 14.3300 | 14.3466 |
| RMSE | 23.2007 | 23.4144 |
| MAPE | 0.1024 | 0.0966 |
| WMAPE | 0.0617 | 0.0618 |

## Research Findings

The four-feature configuration achieved an approximately 5.66%
relative reduction in MAPE.

However, MAE, RMSE and WMAPE were slightly lower for the
three-feature configuration.

These findings demonstrate that introducing a synthetic
two-wheeler-related feature influenced forecasting performance
but did not consistently improve every evaluation metric.

## Research Outputs

This directory will contain verified experimental outputs,
evaluation summaries and forecasting visualisations.
