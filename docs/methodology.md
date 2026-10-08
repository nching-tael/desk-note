# Methodology and validation

Every number Desk Note reports comes from `app/analytics/`. This document gives the formula behind each one, what it assumes, and how it's validated. The validation tests are in [`tests/test_math.py`](../tests/test_math.py). They check against answers worked out by hand or planted in the data, not against the code's own output.

## The worked example

Most of the basics are checked on this four-day dummy portfolio:

| Day | Price of A | SPY | Shares at close | Daily P/L | Value |
|---|---|---|---|---|---|
| d0 | $100 | 400 | 10 (opening, cost $90) | — | $1,000 |
| d1 | $102 | 404 | 15 (bought 5 @ $102) | 10 × $2 = **+$20** | $1,530 |
| d2 | $99 | 400 | 15 | 15 × −$3 = **−$45** | $1,485 |
| d3 | $105 | 408 | 15 | 15 × $6 = **+$90** | $1,575 |

- **Gain:** +$65. The value rose by $575, of which $510 was the purchase and isn't counted as profit.
- **Time-weighted return:** 1.02 × (1,485 / 1,530) × (1,575 / 1,485) − 1 = **+5.0%**. SPY returned +2.0%, so A beat the market by 3.0 points.
- **Average cost:** (10 × $90 + 5 × $102) / 15 = **$94**. Unrealised gain: 15 × ($105 − $94) = **+$165**.

Desk Note reproduces every one of these figures exactly.

## Formulas

### Daily profit and loss
```
P/L(t) = Σ shares held at close(t−1) × (price(t) − price(t−1))
```
Trades take effect at the close of their date, so buying or selling never counts as gain or loss.

### Period return (time-weighted)
```
r(t)   = P/L(t) / portfolio value at close(t−1)
return = Π (1 + r(t)) − 1
```
Unaffected by buying or selling during the period. Without trades, it equals end value ÷ start value − 1.

### Attribution (market / sector / stock-specific)
Fit by ordinary least squares on the 252 sessions before the period, skipping any days before the stock listed:
```
r_stock = a + b_mkt · r_SPY + b_sec · (r_sectorETF − r_SPY) + ε
```
Then, for each day in the period:
```
market   = b_mkt · r_SPY
sector   = b_sec · (r_sectorETF − r_SPY)
specific = r_stock − market − sector          (includes the intercept a)
$ component = component × position value at close(t−1)
```
The three components add up to the dollar P/L by construction. That makes the "sums to total" test a bookkeeping check only. The real validation is whether the betas are right (see below).

### ETFs and funds
Each fund is classified from its published sector weights:

| Fund | Treated as | Factor |
|---|---|---|
| At least 50% in one sector (SMH, XLE, QQQ) | that sector | the sector's SPDR ETF |
| Mostly bonds (BND) | Bonds | market only |
| Anything else (VOO, VXUS, SCHD) | Diversified fund | market only |

For a fund, the stock-specific part means how the fund did beyond what the market (and its sector) explain, e.g. international stocks lagging the US.

Look-through splits each fund's value across its top holdings by weight and adds direct holdings on top:
```
exposure(stock) = direct value + Σ over funds (fund value × stock's weight in the fund)
```
Sector exposure splits each fund by its full sector weights (bonds and cash separately). Only the top holdings are published, so the rest of each fund isn't broken down by stock.

### Target allocation
```
drift (points)      = current weight − target weight
to rebalance ($)    = target weight × total value − current value     (+ buy, − sell)
```
A position is flagged when its drift is bigger than its band (default 5 points for targets of 20% or more, 2 otherwise). New money is split by each position's gap to target after the money arrives, so it never requires selling; a whole-share plan buys one share at a time of whichever position is furthest below target and still affordable.

### Satellite scorecard
```
same money in core = Σ satellite value at close(t−1) × core return(t)     over the days held
added vs core      = satellite's actual gain − same money in core
```
The core return is the value-weighted return of the holdings marked core (the S&P 500 if none are). Following the actual value day by day means buying or selling during the period is handled.

### Risk (today's holdings at today's weights)
| Measure | How it's calculated |
|---|---|
| Volatility | √(wᵀ Σ w), with Σ = sample covariance of the last year of daily returns × 252 |
| Recent volatility | The same, with each day weighted by 0.94 per day of age (RiskMetrics), so a calm or rough patch shows up quickly |
| Beta / downside beta | Slope of the portfolio's daily returns on the S&P 500's; downside beta uses only days the market fell |
| Share of risk | wᵢ (Σw)ᵢ / wᵀ Σ w (sums to 100%) |
| Effective positions | 1 / Σ wᵢ² |
| Bad-day and bad-week losses | Historical: over n days (or overlapping 5-day weeks), "1 in 20" is the k-th worst with k = n/20 rounded up, plus the average of those k worst. "1 in 100" likewise. No bell curve is assumed |
| Worst day / week / month | The worst 1, 5 and 21-session stretch in the loaded history, in today's dollars |
| Drawdown | Largest peak-to-trough fall of today's holdings over the past year and the full loaded history, sessions to recover, and what a fall that size would cost at today's value |
| Market drops | Each holding moves by downside beta × the drop (−10%, −20%, −35%), capped at a total loss |
| Crash replays | 2008, late 2018, Covid 2020 and 2022 replayed on today's holdings (below) |

Holdings that started trading partway through the history have their earlier days filled with beta × the market's return, rather than treated as flat, so a young fund doesn't look calmer than it is. Those holdings are listed, along with any price that hasn't changed for three sessions while the market moved.

**Crash replays.** Each holding uses, in order: its own actual move over the crash (close on or before the start to close on or before the end); the actual move of what it tracks if it's younger than the crash (SPY for VOO, bitcoin for IBIT, QQQ for QQQM, gold for GLDM...); or downside beta × the market's move, marked as estimated. A beta estimate only captures how a holding moves with stocks, so for crypto or narrow themes it understates the loss: in the 2022 replay IBIT is −58.8% via bitcoin, where a beta estimate gave about −22%. If the market's own prices for a crash aren't available, the S&P 500 index's peak-to-trough move is used and labelled as such.

### Options-implied earnings move
```
implied move = (at-the-money call mid + at-the-money put mid) / spot
```
Uses the first expiry within 10 days after the earnings date and the strike nearest the spot price.

### Cost basis
Average cost: a buy updates the average, and a sell books (price − average cost) × shares as realised P/L.

### Journal review
```
market-adjusted move = stock return − β × SPY return     (since the decision)
```
β is fit on the year before the decision.

## Validation

| What | How it's checked | Test |
|---|---|---|
| Daily P/L, value, gain vs purchases | Worked example above, exact | `test_daily_pnl_by_hand`, `test_performance_by_hand` |
| Time-weighted return | Worked example (5.0%); equals simple return with no trades | `test_performance_by_hand`, `test_without_trades_return_is_simple_value_change` |
| Invariances | Doubling positions doubles $ and keeps %; buying and selling the same shares at the same close changes nothing | `test_scaling_positions_…`, `test_round_trip_trade_changes_nothing` |
| Average cost and realised P/L | Hand example: $94 average, $66 realised | `test_average_cost_and_realised_by_hand` |
| Betas, noise-free | Data built as 0.0002 + 1.5·SPY + 0.8·sector; fit returns exactly 1.5 and 0.8 (to 10⁻⁹), and stock-specific is only the intercept | `test_regression_recovers_exact_planted_betas` |
| Betas, noisy | Planted betas recovered within tolerance | `test_regression_recovers_noisy_planted_betas` |
| Betas on the mock market | Each of the 11 holdings within 3 standard errors of the value the mock was generated with | `test_regression_recovers_mock_generator_betas` |
| Beta fit uses all real days | Matches an independent normal-equations solution on a stock that is unchanged on half its days | `test_fit_matches_independent_ols_including_flat_days` |
| Pre-listing days excluded | Stock listed mid-history: exact beta recovered | `test_fit_ignores_days_before_listing` |
| Share of risk, effective positions | Two uncorrelated, equally volatile, equal-weight stocks give 50% / 50% and 2.0; four equal positions give 4.0 | `test_risk_shares_split_evenly…`, `test_effective_positions_equal_weights` |
| Volatility | Equals the sample standard deviation of weighted daily returns × √252 | `test_volatility_by_hand` |
| Portfolio beta | Equals the weighted sum of holding betas (an identity) | `test_portfolio_beta_is_weighted_sum_of_betas` |
| Max drawdown | 100 → 120 → 90 → 110 gives −25%; at today's $110 that's −$28 | `test_max_drawdown_by_hand` |
| Stress test | β = 1.5, $10,000 position gives −$1,500 | `test_stress_test_by_hand` |
| Bad-day losses | 100 days with four −5% and one −4%: 1 in 20 = −4%, average of those = −4.8% | `test_bad_day_losses_by_hand` (tests/test_risk.py) |
| Downside beta | A stock that moves 2× on down days and 0.5× on up days has downside beta exactly 2 | `test_downside_beta_only_uses_falling_days` |
| Young holdings | Days before listing are filled with beta × market, not flat | `test_days_before_listing_are_filled_from_beta` |
| Crash replays | Actual moves where available, then what a fund tracks, then beta (marked) | `test_crash_replay_uses_actual_moves_where_it_can`, `test_crash_replay_uses_what_a_young_fund_tracks` |
| Implied move | Call mid 3.1 + put mid 2.9 on spot 100.4 gives 6.0% | `test_implied_move_by_hand` |
| Journal review | A stock that moves exactly with the market shows 0% market-adjusted | `test_journal_market_adjusted_move_by_hand` |
| Drift and rebalance | 80/20 against 70/30 on $10,000: sell $1,000 of A, buy $1,000 of B | `test_drift_and_rebalance_by_hand` (tests/test_allocation.py) |
| New money | $1,000 goes entirely to the position that's short; $5,000 lands exactly on target | `test_new_money_goes_to_the_gap_first`, `test_enough_new_money_lands_exactly_on_target` |
| Satellite scorecard | Gain $200 vs $0 in the core; with a mid-period buy, $400 vs −$100 | `test_scorecard_by_hand`, `test_scorecard_follows_buys_during_the_period` |
| Look-through | NVDA = direct value + 8.1% of VOO + 19.3% of SMH, checked by hand; sector exposure sums to 100% | `test_look_through_nvidia_by_hand` (tests/test_funds.py) |

**Do the tests catch real bugs?** Re-introducing a bug I found during this review (dropping zero-return days from the beta fit) makes `test_fit_matches_independent_ols_including_flat_days` fail. Counting purchases as gains makes the two worked-example tests fail.

**Independent cross-check (one-off).** Refitting all 11 mock holdings with `statsmodels` OLS matched Desk Note's betas to within 1.3 × 10⁻¹⁵. Volatility (19.1% vs 19.09%) and portfolio beta (1.17 vs 1.172) matched pandas and statsmodels to rounding.

## Known limitations

- **Dividends.** Yahoo's adjusted prices fold dividends into the price, so dividends count as reinvested gains. Historical dollar values are slightly below what actually traded, and won't match a broker statement exactly.
- **Average cost, not FIFO.** US brokers usually report realised gains first-in-first-out, so realised P/L here won't match tax forms.
- **Trades at the close.** Gains on the day of a trade, between the fill and the close, aren't counted.
- **The sector ETF contains the stock.** NVDA is a large part of XLK, so on a big NVDA-specific day some of its own move shows up as "sector".
- **No industry factor.** A move across the whole semiconductor industry is counted as stock-specific for each chip stock.
- **Betas are estimates.** With one year of daily data, a typical standard error is ±0.1 for b_mkt and ±0.2 for b_sec.
- **History-based risk** only knows the past it has: bad-day losses and drawdowns come from about two years of data, so they won't include a crash bigger than anything in that window. The crash replays exist to fill that gap.
- **Implied move** includes some ordinary volatility when the first usable expiry is several days after earnings.
- **The journal review** applies one beta to the whole move since a decision. Compounding makes this approximate over long periods.
