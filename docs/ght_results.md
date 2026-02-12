# GHT Results Summary (Template)

## Forecasting Benchmarks

### L=96

| Dataset | Model Variant | MSE | MAE | Train Throughput (samples/s) | Infer Throughput (samples/s) | Peak VRAM (MB) | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | PatchTST |  |  |  |  |  |  |
| ETTh1 | GHT-Rule (single) |  |  |  |  |  |  |
| ETTh1 | GHT-EMA (single) |  |  |  |  |  |  |
| ETTh1 | GHT-Dual (trend+res) |  |  |  |  |  |  |
| ETTh2 | PatchTST |  |  |  |  |  |  |
| ETTh2 | GHT-Rule (single) |  |  |  |  |  |  |
| ETTh2 | GHT-EMA (single) |  |  |  |  |  |  |
| ETTh2 | GHT-Dual (trend+res) |  |  |  |  |  |  |
| ETTm1 | PatchTST |  |  |  |  |  |  |
| ETTm1 | GHT-Rule (single) |  |  |  |  |  |  |
| ETTm1 | GHT-EMA (single) |  |  |  |  |  |  |
| ETTm1 | GHT-Dual (trend+res) |  |  |  |  |  |  |
| Electricity | PatchTST |  |  |  |  |  |  |
| Electricity | GHT-Rule (single) |  |  |  |  |  |  |
| Electricity | GHT-EMA (single) |  |  |  |  |  |  |
| Electricity | GHT-Dual (trend+res) |  |  |  |  |  |  |
| Traffic | PatchTST |  |  |  |  |  |  |
| Traffic | GHT-Rule (single) |  |  |  |  |  |  |
| Traffic | GHT-EMA (single) |  |  |  |  |  |  |
| Traffic | GHT-Dual (trend+res) |  |  |  |  |  |  |
| Weather | PatchTST |  |  |  |  |  |  |
| Weather | GHT-Rule (single) |  |  |  |  |  |  |
| Weather | GHT-EMA (single) |  |  |  |  |  |  |
| Weather | GHT-Dual (trend+res) |  |  |  |  |  |  |
| BeijingAirQuality | PatchTST |  |  |  |  |  |  |
| BeijingAirQuality | GHT-Rule (single) |  |  |  |  |  |  |
| BeijingAirQuality | GHT-EMA (single) |  |  |  |  |  |  |
| BeijingAirQuality | GHT-Dual (trend+res) |  |  |  |  |  |  |

### L=1440

| Dataset | Model Variant | MSE | MAE | Train Throughput (samples/s) | Infer Throughput (samples/s) | Peak VRAM (MB) | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | PatchTST |  |  |  |  |  |  |
| ETTh1 | GHT-Rule (single) |  |  |  |  |  |  |
| ETTh1 | GHT-EMA (single) |  |  |  |  |  |  |
| ETTh1 | GHT-Dual (trend+res) |  |  |  |  |  |  |
| ETTh2 | PatchTST |  |  |  |  |  |  |
| ETTh2 | GHT-Rule (single) |  |  |  |  |  |  |
| ETTh2 | GHT-EMA (single) |  |  |  |  |  |  |
| ETTh2 | GHT-Dual (trend+res) |  |  |  |  |  |  |
| ETTm1 | PatchTST |  |  |  |  |  |  |
| ETTm1 | GHT-Rule (single) |  |  |  |  |  |  |
| ETTm1 | GHT-EMA (single) |  |  |  |  |  |  |
| ETTm1 | GHT-Dual (trend+res) |  |  |  |  |  |  |
| Electricity | PatchTST |  |  |  |  |  |  |
| Electricity | GHT-Rule (single) |  |  |  |  |  |  |
| Electricity | GHT-EMA (single) |  |  |  |  |  |  |
| Electricity | GHT-Dual (trend+res) |  |  |  |  |  |  |
| Traffic | PatchTST |  |  |  |  |  |  |
| Traffic | GHT-Rule (single) |  |  |  |  |  |  |
| Traffic | GHT-EMA (single) |  |  |  |  |  |  |
| Traffic | GHT-Dual (trend+res) |  |  |  |  |  |  |
| Weather | PatchTST |  |  |  |  |  |  |
| Weather | GHT-Rule (single) |  |  |  |  |  |  |
| Weather | GHT-EMA (single) |  |  |  |  |  |  |
| Weather | GHT-Dual (trend+res) |  |  |  |  |  |  |
| BeijingAirQuality | PatchTST |  |  |  |  |  |  |
| BeijingAirQuality | GHT-Rule (single) |  |  |  |  |  |  |
| BeijingAirQuality | GHT-EMA (single) |  |  |  |  |  |  |
| BeijingAirQuality | GHT-Dual (trend+res) |  |  |  |  |  |  |

## Ablation Plan

- no compression: `ght_ratio=1.0`
- random compression: TODO (implement random token selection)
- rule graph: `ght_graph_mode=rule`
- EMA graph: `ght_graph_mode=ema`
- single stream vs dual stream: `ght_dual_stream=True/False`
- local constraint: `ght_local_window` sweep
- ratio sweep: `ght_ratio`, `ght_ratio_trend`, `ght_ratio_residual`
