# btc-cent-forecast-01 entry: BTC 4-hour-ahead point forecast.
#
# Model: mean reversion toward a slow exponential moving average.
#   guess = last + 0.35 * (EMA1440 - last)
# where EMA1440 is the 24-hour exponential moving average of the case's
# 4,320 one-minute closes (alpha = 2 / (1440 + 1)) and `last` is the close
# at the checkpoint. The score is mean absolute error in cents, so the
# best single number is the honest conditional median; a shrunken move
# back toward the slow mean is the median estimate this rule encodes.
#
# Model selection (walk-forward inside the practice series, 528
# pseudo-cases: checkpoints every 30 min with >=1440 min of history,
# scored against the realised close +240 min):
#   persistence (last close)          MAE 43,821.74 cents
#   EMA1440 shrink k=0.35            MAE 41,993.91 cents  <- chosen
#   EMA1440 shrink k=0.40            MAE 42,020.18 cents
#   EMA720  shrink k=0.50            MAE 42,085.97 cents
#   72h-median shrink k=0.05         MAE 43,409.62 cents
#   momentum (EMA60) a=0.10          MAE 44,016.05 cents
# The chosen rule beat persistence on every one of the six practice
# series in walk-forward (per-series gains +700 to +2,800 cents).
# Caveat, recorded honestly: on the five recoverable practice cases
# alone (n=5, one steep down-trend leg) it scores 50,769 vs 47,055
# cents for persistence -- that 5-case gap is well inside its noise
# (per-case errors $145-$892), so the 528-case walk-forward decides.
#
# Pure standard library, no network, deterministic, well under the
# 8-second per-case cap (one linear pass over the prices).

import json
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Slow mean: 24-hour EMA.  Chosen by walk-forward grid over spans
# 720 / 1440 / 2160 / 2880 minutes.
EMA_SPAN_MINUTES = 1440
# Shrink toward the slow mean.  Chosen by walk-forward grid over
# k in 0.05..0.55; 0.35 is the walk-forward minimum.
SHRINK = 0.35

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999


def price5(x: float) -> float:
    """Keep a guess on the grid's 5 decimals, inside (0, 10,000,000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def ema_last(prices, span: int) -> float:
    """Final value of the exponential moving average over `prices`."""
    alpha = 2.0 / (span + 1.0)
    e = prices[0]
    for x in prices[1:]:
        e = alpha * x + (1.0 - alpha) * e
    return e


def forecast(prices) -> float:
    if not prices:
        raise ValueError("market has no prices")
    for x in prices:
        if not isinstance(x, (int, float)) or x != x or x in (float("inf"), float("-inf")):
            raise ValueError("market holds a non-finite price")
    last = prices[-1]
    slow = ema_last(prices, EMA_SPAN_MINUTES)
    return price5(last + SHRINK * (slow - last))


def main() -> None:
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with path.open(encoding="utf-8") as f:
            case = json.load(f)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        prices = markets[0].get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError(f"{path.name} has no prices")
        guess = forecast(prices)
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        with (out_dir / f"{case_id}.out").open("w", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
