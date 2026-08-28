# NCR empirical vs 6N champion — ranking evaluation

Official fantasy points are the outcome. Predictions are compared as rankings, so the models' different point scales do not matter.

## Two-round average

| Top N | NCR overlap | Champion overlap | NCR point capture | Champion point capture |
|---:|---:|---:|---:|---:|
| 10 | **25.0%** | 10.0% | **61.6%** | 45.2% |
| 25 | **39.0%** | 28.0% | **70.1%** | 64.2% |
| 50 | **48.5%** | 41.0% | **72.9%** | 66.9% |
| 100 | **62.4%** | 57.4% | **78.8%** | 73.9% |

## Round-by-round

### GW1

Spearman rank correlation: **NCR 0.614**, champion 0.573.

| Top N | NCR hits | Champion hits | NCR point capture | Champion point capture |
|---:|---:|---:|---:|---:|
| 10 | **3.0** | 1.0 | **62.6%** | 45.5% |
| 25 | **10.5** | 8.0 | **67.7%** | 65.0% |
| 50 | **24.5** | 21.5 | **71.3%** | 67.6% |
| 100 | **65.8** | 59.8 | **83.5%** | 77.8% |

_Champion prediction is a point-in-time replay; its original GW1 CSV was not retained._

### GW2

Spearman rank correlation: **NCR 0.547**, champion 0.466.

| Top N | NCR hits | Champion hits | NCR point capture | Champion point capture |
|---:|---:|---:|---:|---:|
| 10 | **2.0** | 1.0 | **60.7%** | 44.8% |
| 25 | **9.0** | 6.0 | **72.4%** | 63.4% |
| 50 | **24.0** | 19.5 | **74.5%** | 66.2% |
| 100 | **59.0** | 55.0 | **74.1%** | 70.0% |

_Both prediction CSVs are the saved pre-match artifacts._

## Data quality and interpretation

| GW | NCR predictions | Champion predictions | Common cohort | Official pool | Non-zero scorers missing from both |
|---:|---:|---:|---:|---:|---:|
| 1 | 270 | 274 | 270 | 445 | 4 |
| 2 | 262 | 262 | 262 | 466 | 16 |

The headline overlap uses the full official leaderboard. Players absent from both prediction pools therefore count as misses. Spearman correlation uses only the common prediction cohort, isolating model ranking quality from shared lineup-input omissions.

Top-N hits are tie-adjusted at the actual cutoff. Point capture is the official points earned by a model's predicted top N divided by the points earned by the hindsight top N.

Only two rounds are available, so this can compare current ranking quality but cannot establish a learning curve or prove how the NCR model will improve with more rounds.

### Official top-100 players absent from both prediction pools

| GW | Player | Actual points | Status |
|---:|---|---:|---|
| 1 | Damian Penaud | 26 | P |
| 2 | Justo Piccardo | 57 | P |
| 2 | Romain Ntamack | 56 | P |
| 2 | Taira Main | 38 | P |
| 2 | Kalvin Gourgues | 26 | B |
| 2 | Florian Verhaeghe | 24 | P |
| 2 | Elrigh Louw | 18 | B |
