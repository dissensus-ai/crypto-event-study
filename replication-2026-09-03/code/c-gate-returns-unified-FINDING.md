# First-moment unified-basis result

Authoritative outputs are `results/c-gate-returns-unified-results.csv` and its
byte-identical release alias, `results/c11-returns-results.csv`. The run uses
5,000 event-block bootstrap draws with seed 42.

## Coverage rule

For the unified gates, an asset-event cell is usable only if the daily
`[-5,+30]` event window contains all 36 observations. This removes the
truncated BNB cell for regulatory event 50 (2025-08-07), whose source series
ends on 2025-08-31. Six earlier BNB cells (events 1--6) were already unusable
because they fail the predecessor engine's minimum-estimation-data rule. Thus:

- six-asset gate A uses 293 of 300 asset-event cells and omits
  `1:BNB;2:BNB;3:BNB;4:BNB;5:BNB;6:BNB;50:BNB`;
- five-asset gate B uses 243 of 250 cells and omits the same seven cells.

The legacy estimation slice is unchanged: it remains the inclusive interval
from event date minus 281 calendar days through event date minus 31 calendar
days, giving 251 observations when complete. The gate does not newly require
a complete estimation slice.

## Unified constant-mean results

On the six-asset, 50-event basis, mean CAR is -0.639469% for infrastructure
events and -9.332631% for regulatory events, a difference of +8.693162
percentage points. The event-block bootstrap gives two-sided p=0.202000 and
95% CI [-4.504648%, 22.222333%]. The event-level Welch test gives p=0.216434
and 95% CI [-5.261478%, 22.647802%].

Dropping XRP gives mean CARs of +0.439147% and -7.078012%, a difference of
+7.517159 percentage points. The event-block bootstrap gives two-sided
p=0.297600 and 95% CI [-6.527658%, 21.820609%]; the Welch test gives p=0.311797
and 95% CI [-7.268705%, 22.303024%].

Both bases therefore fail to reject equality at the first moment. The observed
effects remain below the approximately 19.5 and 20.6 percentage-point minimum
detectable effects at 80% power; non-rejection is not evidence of equivalence.

## Legacy negative-subset checks

The predecessor-sample smoke test remains on four assets
(BTC/ETH/SOL/ADA), eight valid negative infrastructure events and seven valid
negative regulatory events. For its constant-mean event-level difference of
+1.465221 percentage points, the exact two-sided event-label permutation test
enumerates all `C(15,8)=6,435` assignments; 5,961 are at least as extreme in
absolute value, so p=0.926340.

The market-model robustness uses the packaged
`src.event_study.MarketModel`, with BTC as the sole proxy. It estimates alpha
and beta separately for every non-BTC asset/event estimation window; BTC uses
the class's constant-mean fallback. Event-level CARs are then averaged across
available assets before comparing event classes. Mean CAR is -3.267100% for
infrastructure and -9.151367% for regulatory events, giving
delta=+5.884267 percentage points. The event-block bootstrap p-value is
0.523600, the Welch p-value is 0.583489, and the exact-permutation p-value is
0.574359. This robustness result also fails to reject equality.

The smoke test uses the predecessor snapshot
`code/data/events_reclassified.json`, including its historical local-date
Celsius coding of 2022-06-12, solely to reproduce that earlier analysis. The
unified gates use the corrected UTC date 2022-06-13 in
`code/data/events.csv` and `results/c1-dropout-census.csv`.
