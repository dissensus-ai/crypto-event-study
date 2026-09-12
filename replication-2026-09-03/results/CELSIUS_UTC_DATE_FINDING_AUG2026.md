# Fifth date correction: Celsius freeze 2022-06-12 → 2022-06-13 (UTC convention)

**Found:** 7 Aug 2026, by an independent pre-publication referee report (round 7), which flagged
the Celsius memo as "published ≈02:00 UTC on 13 June by several accounts, but dated 2022-06-12".
**Verified and confirmed here by primary source.** This is the fifth error found in the 50-event
census and the first found by applying the paper's own stated dating convention as a test.

## The convention (§3.1.2, added in the 4-date correction)

> Events are dated by the UTC calendar day on which the defining information became public
> (filing or announcement), not by market reaction or procedural confirmation.

## Verification method: snowflake-ID decoding (primary, not secondary)

Twitter/X status IDs encode their creation time in milliseconds since the Twitter epoch
(1288834974657): `timestamp_ms = (id >> 22) + 1288834974657`. The timestamp is therefore
embedded in the URL itself and is not subject to the reporting errors that afflict secondary
time claims.

| event | status ID | decoded UTC | local (EDT) | recorded | verdict |
|---|---|---|---|---|---|
| Celsius pause announcement | 1536169010877739009 | **2022-06-13T02:10:07Z** | 2022-06-12 22:10 | 2022-06-12 | ⛔ **VIOLATION → 2022-06-13** |
| BNB Chain bridge pause | 1578148078636650496 | 2022-10-06T22:19:57Z | 2022-10-06 18:19 | 2022-10-06 | ✅ OK |

Corroboration for Celsius: CNBC filed its report under 2022-06-13. The Celsius Medium memo
("A Memo to the Celsius Community") and the @CelsiusNetwork announcement are the defining
public disclosure; there is no earlier official statement.

**Methodological note worth keeping:** a secondary source reported the BNB announcement as
"9:19 pm EDT" (which would have made it 01:19 UTC on 7 Oct — a second violation). The decoded
ID shows 6:19 pm EDT / 22:19 UTC on 6 Oct. **The secondary time report was wrong; the census
was right.** Decode, don't trust.

## Other events assessed

* **Bybit hack (43, 2025-02-21):** Ben Zhou livestream 17:15 UTC 21 Feb; ZachXBT attribution
  19:09 UTC 21 Feb. Same UTC day. ✅
* **Protocol upgrades (5, 8, 11, 17, 21, 26, 30, 38, 39, 42, 47):** block-timestamped, all
  comfortably within their recorded UTC day (e.g. Merge 06:42 UTC 15 Sep; Dencun 13:55 UTC
  13 Mar; Pectra ~10:00 UTC 7 May — the last is stated in the census title itself). Note
  BTC halving 2024 fired 00:09 UTC on 20 Apr, i.e. just *after* the UTC boundary, and is
  correctly dated 2024-04-20. ✅
* **US regulatory/judicial (2, 12, 20, 23, 31, 32, 33, 34, 37, 40, 44, 45, 46, 48, 49, 50):**
  ET business-hours filings/announcements = 13:00–21:00 UTC. Structurally low risk. ✅
* **Asian/EU-origin (6, 15, 19, 22, 35, 41):** local daytime = same UTC day. ✅
* **Binance hack '19 (3, 2019-05-07):** breach detected 17:15:24 UTC 7 May; CZ's public
  "unscheduled maintenance" tweet ≈2 h later (≈19:15 UTC), Binance's official statement same
  day. Same UTC day. ✅
* **USDC/SVB (29, 2023-03-10) — bundled event, dated to the primary disclosure. KEEP 03-10,
  but note the reasoning.** The event bundles two disclosures: SVB entering FDIC receivership
  (announced ≈16:00 UTC 10 Mar — the primary, macro-side shock and the first-named element of
  the census label) and Circle's $3.3bn-exposure statement, which went out ≈22:00 ET 10 Mar =
  **≈03:00 UTC 11 Mar** and triggered the depeg. Under the convention the event is anchored to
  the decisive public event, which for "SVB collapse" is the receivership announcement on
  10 March UTC. This is the same anchoring rule applied to FTX (Chapter 11 filing rather than
  the 6–8 Nov withdrawal-halt reports). The $[-3,+3]$ window covers both candidate anchors.
  Recommend the manuscript's dating sentence name SVB alongside FTX so the rule is visibly
  applied twice rather than once. ✅ (defensible; documented)
* **Poly hack (18, 2021-08-10):** Poly Network's public disclosure was during UTC daytime on
  10 Aug. ✅
* **Terra (24) and Black Thursday (7):** market processes rather than point-in-time
  announcements; the convention is applied to the initiating disclosure/session. No clock-time
  violation is definable. ✅

## Impact on results — SMALL, and again toward the null

The [−3,+3] variance windows for 12 vs 13 June share six of seven days, and mid-June 2022 was
uniformly turbulent, so the swap (drop 06-09, add 06-16) barely moves the second moment.

| quantity | Celsius 06-12 | Celsius 06-13 (corrected) |
|---|---|---|
| δ̄_infra | 2.0440 | **2.0443** |
| δ̄_reg | 0.5854 | **0.5861** |
| **multiplier** | 3.4916× | **3.4879×** (both print **3.49×**) |
| d̄_obs | 1.4586 | 1.4582 |
| 6-asset CAR diff | +8.78 pp | **+8.69 pp** |
| block p / Welch p | 0.199 / 0.211 | **0.202 / 0.216** |
| 5-asset CAR diff | +7.66 pp (p 0.286/0.301) | **+7.52 pp (p 0.298/0.312)** |
| CAR_infra / CAR_reg | −0.46% / −9.24% | **−0.64% / −9.33%** |
| MDE (6-asset) | 19.4 pp | **19.5 pp** |
| census: fails screen / zero-movers / dropped passers / big movers | 13 / 4 / 57 / 29 | **unchanged** |
| pass rates, retention z | 64.6% / 77.4%, z=−1.5951 | **unchanged** |

**The headline multiplier is unchanged at printed precision.** Per-asset coefficients do move at
the third decimal (XRP +0.039, LTC −0.035, BNB +0.012, BTC −0.009, ETH −0.003, ADA −0.001),
so Tables 4/6 (per-asset δ and ratios) need regeneration; the mean row prints 2.044 / **0.586**
(was 0.585).

Also affected: Celsius crosses a GDELT week boundary (12 Jun is a Sunday in week 2022-06-06;
13 Jun opens week 2022-06-13), so the c15 event-dummy Granger control re-maps that event to the
following week — the correct mapping for a 02:10 UTC Monday announcement.

The reconstructed candidate-pool source, curated event census, and committed outputs now all use
the corrected 2022-06-13 UTC date. The earlier hardcoded-date release blocker is closed.
