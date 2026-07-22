"""Fan charts: history plus the projected point and prediction band per zone."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless / test-safe
import matplotlib.pyplot as plt
import pandas as pd

from src.forecast import compositional as comp


def fan_chart_data(df: pd.DataFrame, forecast_df: pd.DataFrame, zone: str,
                   target_year: int) -> dict:
    years, mat = comp.shares_matrix(df, [zone], value_col="share")
    row = forecast_df[forecast_df["zone"] == zone].iloc[0]
    return {
        "hist_years": [int(y) for y in years],
        "hist_shares": [float(v) for v in mat[:, 0]],
        "target_year": int(target_year),
        "point": float(row["projected_share"]),
        "lo": float(row["share_lo"]),
        "hi": float(row["share_hi"]),
    }


def plot_fan_charts(df: pd.DataFrame, forecast_df: pd.DataFrame, out_dir: str,
                    target_year: int) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for zone in forecast_df["zone"]:
        d = fan_chart_data(df, forecast_df, zone, target_year)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(d["hist_years"], d["hist_shares"], marker="o", label="history")
        ax.plot([d["hist_years"][-1], d["target_year"]],
                [d["hist_shares"][-1], d["point"]], "--", color="orange", label="forecast")
        ax.fill_between([d["hist_years"][-1], d["target_year"]],
                        [d["hist_shares"][-1], d["lo"]],
                        [d["hist_shares"][-1], d["hi"]],
                        color="orange", alpha=0.2, label="95% PI")
        ax.set_title(f"{zone} share of shots -> {d['target_year']}")
        ax.set_xlabel("Season"); ax.set_ylabel("Share of shots")
        ax.legend(); ax.grid(True)
        path = os.path.join(out_dir, f"fan_{zone}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_player_upside(upside_df: pd.DataFrame, out_dir: str, top_n: int = 20) -> str:
    """Horizontal bar chart ranking players by 2030 shot-diet opportunity delta.

    Bars use a diverging palette: orange (upside to capture) / blue (already ahead).
    Each bar is annotated with the player's specific shot recommendation in muted ink.
    """
    data = upside_df.head(top_n).iloc[::-1].reset_index(drop=True)
    n = len(data)

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.44)))
    fig.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # Diverging colour: orange = room to grow, blue = ahead of curve
    _ORANGE = "#eb6834"
    _BLUE = "#2a78d6"
    colors = [_ORANGE if v >= 0 else _BLUE for v in data["opportunity_delta"]]

    bars = ax.barh(
        data["player_name"],
        data["opportunity_delta"],
        color=colors,
        height=0.65,      # ≤ 24px — breathing room between bars
        linewidth=0,      # no border; separation is white space between bars
    )

    # Specific rec annotations in muted ink — never wear the data colour
    if "specific_rec" in data.columns:
        x_range = data["opportunity_delta"].abs().max() or 0.01
        for bar, rec in zip(bars, data["specific_rec"]):
            if rec and rec not in ("already well-positioned", "insufficient data"):
                ax.text(
                    bar.get_width() + x_range * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    str(rec),
                    va="center", ha="left",
                    fontsize=7.5, color="#898781",
                )

    # Zero baseline — axis-ink token, 1 px
    ax.axvline(0, color="#c3c2b7", linewidth=1.0, zorder=3)

    # Recessive hairline grid; no top/right/left spines
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    ax.set_xlabel(
        "Opportunity delta (pts/shot vs 2030 league mix)",
        color="#52514e", fontsize=9,
    )
    ax.set_title(
        "Player shot-diet upside — 2030 forecast",
        color="#0b0b0b", fontsize=11, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="#52514e", labelsize=8.5)

    # Extra right margin so annotations don't clip
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 1.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "player_upside.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
