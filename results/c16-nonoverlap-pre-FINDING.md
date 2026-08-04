# C16 FINDING: non-overlapping pre-event dummy (clean anticipation test)

**Referee points addressed:**
- **#3 (Major):** widening the regulatory pre-window (c8b) mechanically dilutes
  the average dummy, so a *falling* coefficient cannot rule out concentrated
  anticipation. Wants a separate non-overlapping pre-event dummy (e.g. [-10,-4]).
- **#4 (Minor):** c8b widened only the regulatory pre-window; infrastructure was
  held at [-3,+3]. Infrastructure events can also leak (on-chain rumours,
  pre-exploit liquidity drains). A symmetric treatment is warranted.

**Test (`c16_nonoverlap_pre_dummy.py`):** one GJR-GARCH-X per asset with FOUR
event regressors -- infrastructure and regulatory [-3,+3] event windows PLUS
infrastructure and regulatory [-10,-4] *non-overlapping* pre-windows -- plus the
three sentiment controls. FastTARCHX point estimates (canonical to ~3dp).

**Cross-asset results (naive one-sample t vs 0, descriptive; the copula bootstrap
remains the inference of record):**

| coefficient | cross-asset mean | median | naive p vs 0 |
|-------------|-----------------|--------|-------------|
| infra event [-3,+3] | +1.861 | +2.001 | 0.002 |
| reg event [-3,+3]   | +0.294 | +0.234 | 0.169 |
| **infra pre [-10,-4]** | **+0.230** | +0.133 | **0.212 (n.s.)** |
| **reg pre [-10,-4]**   | **+0.236** | +0.238 | **0.068 (marginal)** |

Event-window multiplier in this four-window spec: 1.861 / 0.294 = 6.33x.

**Reading (honest, two-sided):**
1. **Infrastructure: no pre-event leakage.** Pre-window coefficient +0.23,
   naive p=0.21. Consistent with infrastructure events being genuine surprises
   (the "more plausibly exogenous" leg). Point #4's worry is testable and comes
   out null.
2. **Regulatory: a modest, marginal pre-event bump the symmetric window misses.**
   Pre-window coefficient +0.24, marginal naive p=0.068. So *some* concentrated
   regulatory anticipation is present -- contrary to the widening sweep's
   apparent "no anticipation / multiplier grows to 8.77x" reading, which is a
   dilution artefact (referee #3 is correct).
3. **Crediting the regulatory pre-window** to the regulatory response lowers the
   event-window multiplier from ~6x to ~3.5-4x (reg total 0.294+0.236=0.530;
   infra event 1.861 -> 3.5x, or crediting both legs 2.091/0.530 = 3.9x). The
   asymmetry **attenuates but remains directional and infrastructure-dominated**,
   and -- as throughout -- is not significant under the copula bootstrap.

**Net effect on the paper:** the §6.2.2 claim "anticipation runs against the
asymmetry" (leaning on the widening sweep growing to 8.77x) is reframed. The
widening sweep's growth is a denominator-dilution artefact and is no longer read
as evidence. The clean non-overlapping test finds infrastructure unanticipated,
regulatory modestly anticipated, and the directional (insignificant) asymmetry
robust to crediting that anticipation. This is a concession that strengthens the
paper's credibility and directly answers both referee points.

Outputs: `c16-nonoverlap-pre-per-asset.csv`, `c16-nonoverlap-pre-summary.csv`.
