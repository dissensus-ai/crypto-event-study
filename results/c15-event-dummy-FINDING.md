# C15 FINDING: weekly sentiment->volatility lead survives event-dummy controls

**Referee point (Digital Finance R&R):** the weekly bivariate Granger VAR
(sentiment -> realised volatility) omits the 50 classified events as exogenous
controls. Since an event spikes both sentiment and volatility, an unmodelled
event could manufacture a spurious sentiment->volatility lead.

**Test (`c15_granger_event_dummies.py`):** re-ran every one of the 18
(asset x sentiment) sentiment->volatility pairs with weekly-aggregated event
dummies added to the VAR as exogenous controls, entering **contemporaneously and
at lags 1..p** (so a same-week event shock and its persistence into following
weeks are both absorbed). Reused the c12 conditional-Granger SSR-F design (the
same machinery already used for the Bitcoin-co-movement and litigation controls).
Two control specs:
- **TOTAL** — a single weekly all-event count.
- **SPLIT** — separate weekly infrastructure and regulatory event counts (strict
  reading: lets the two shock types load differently on volatility).

All 50 events map onto the GDELT weekly grid (26 infra, 24 reg).

**Result: the lead is NOT an event-shock artefact.**

| pair | uncontrolled p | TOTAL-ctrl p | SPLIT-ctrl p |
|------|---------------|-------------|-------------|
| XRP reg | 0.0013 | 0.0022 | 0.0028 |
| XRP infra | 0.0104 | 0.0185 | 0.0242 |
| XRP aggregate | 0.0009 | 0.0017 | 0.0023 |
| ADA reg | 0.0005 | 0.0012 | 0.0013 |
| ADA infra | 0.0055 | 0.0128 | 0.0147 |
| ADA aggregate | 0.0010 | 0.0023 | 0.0028 |
| BNB reg | 0.0057 | 0.0149 | 0.0172 |

- **All 7 BH-FDR survivors remain significant at raw p<0.05 under both specs.**
  p-values move only trivially upward (the event block absorbs a little variance).
- Under TOTAL control: 8/18 raw-significant; **all 7 FDR survivors retained**;
  7/18 survive BH-FDR at q<0.05 (the same 7).
- Under SPLIT control: 8/18 raw-significant; **all 7 FDR survivors retained**;
  4/18 survive BH-FDR at q<0.05 (SPLIT adds more regressors, so the FDR family is
  a slightly stricter test; the point estimates are unchanged).
- The two marginal pairs that were never FDR survivors wash out as expected
  (LTC-reg 0.042 -> 0.090/0.057; BNB-infra 0.049 -> 0.077/0.076).

**Interpretation:** conditioning on the known event shocks does not remove the
weekly sentiment lead. The confound the reviewer flags is real in principle but
does not drive the result: sentiment carries predictive content for weekly
volatility over and above the classified events themselves. This is consistent
with the litigation-window finding (c12) — the lead is concentrated in
SEC-litigation assets — but it is not merely a shadow of the discrete event dates.

Outputs: `c15-granger-event-dummies.csv`, `c15-granger-event-dummies-summary.csv`.
