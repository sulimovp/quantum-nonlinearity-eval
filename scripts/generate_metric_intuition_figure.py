"""Generate the four-panel metric intuition figure for the NeurIPS paper.

The panels are illustrative schematics, not reference-run results. They
use synthetic curves chosen so each metric definition is recognizable
at a glance.

Output: neurips2026/figures/metric_intuition.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "neurips2026" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "metric_intuition.pdf"


def _panel_nlc(ax: plt.Axes) -> None:
    x = np.linspace(-np.pi, np.pi, 800)
    ks = [1, 2, 4, 8, 16]
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(ks):
        offset = -2.4 * i
        ax.plot(
            x,
            np.sin(k * x) + offset,
            color=cmap(i / (len(ks) - 1)),
            lw=1.2,
        )
        ax.text(
            -np.pi - 0.15,
            offset,
            f"k={k}",
            ha="right",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(-np.pi - 1.0, np.pi + 0.2)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([])
    ax.set_xlabel(r"input $x$")
    ax.set_title(
        r"NLC: $\sin(kx)$ ladder probes Fourier reach"
        "\n"
        r"higher $k$ = finer modes the encoding must compose",
        fontsize=10,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _panel_bppi(ax: plt.Axes) -> None:
    n = np.array([2.0, 4.0, 6.0, 8.0])
    healthy = -0.3 * n + 0.5
    barren = -1.4 * n + 2.0

    ax.plot(n, healthy, "o-", color="#2a9d8f", lw=2, ms=6, label=r"healthy ($\alpha\approx-0.3$)")
    ax.plot(n, barren, "s--", color="#e76f51", lw=2, ms=6, label=r"barren ($\alpha\approx-1.4$)")

    ax.set_xticks(n)
    ax.set_xlabel(r"qubit count $n$")
    ax.set_ylabel(r"$\log_2 \mathrm{Var}_{\theta}\!\left[\partial L/\partial \theta_0\right]$")
    ax.set_title(
        "BPPI: slope of gradient-variance decay\n"
        r"steeper negative $\alpha$ $\Rightarrow$ closer to a barren plateau",
        fontsize=10,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)


def _panel_nss(ax: plt.Axes) -> None:
    p_grid = np.array([0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    p_plot = np.maximum(p_grid, 1e-4)

    robust = np.array([0.95, 0.92, 0.88, 0.85, 0.74, 0.55])
    fragile = np.array([0.95, 0.78, 0.55, 0.40, 0.18, 0.05])

    ax.plot(p_plot, robust, "o-", color="#2a9d8f", lw=2, ms=6, label="robust mechanism")
    ax.fill_between(p_plot, 0, robust, color="#2a9d8f", alpha=0.15)
    ax.plot(p_plot, fragile, "s--", color="#e76f51", lw=2, ms=6, label="fragile mechanism")

    ax.set_xscale("log")
    ax.set_xlim(p_plot[0], p_plot[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"depolarizing rate $p$ (log scale)")
    ax.set_ylabel(r"bounded skill $s(R^2(p))$")
    ax.set_title(
        "NSS: skill of a TRAINED model under noise\n"
        "shaded area = NSS (averaged over noise ladder)",
        fontsize=10,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def _panel_ter(ax: plt.Axes) -> None:
    steps = np.linspace(0, 200, 200)
    train = np.clip(0.95 * (1 - np.exp(-steps / 40.0)) + 0.02 * np.sin(steps / 8.0), 0, 1)

    b = float(np.max(train))
    t_final = float(train[-1]) - 0.05
    A_train = float(np.trapezoid(train, steps) / (steps[-1] - steps[0]))

    ax.plot(steps, train, color="#264653", lw=2, label="train skill")
    ax.fill_between(
        steps,
        0,
        train,
        color="#264653",
        alpha=0.15,
        label=fr"$A_{{\mathrm{{train}}}}\approx{A_train:.2f}$",
    )
    ax.axhline(b, color="#e9c46a", ls="--", lw=1.2, label=fr"best $b\approx{b:.2f}$")
    ax.scatter(
        [steps[-1]],
        [t_final],
        color="#e76f51",
        s=60,
        zorder=5,
        label=fr"final test $t\approx{t_final:.2f}$",
    )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("skill (clipped to [0,1])")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        r"TER: $0.50\,A_{\mathrm{train}} + 0.30\,b + 0.20\,t$, then $\sqrt{n^{\min}_{\mathrm{params}}/n_{\mathrm{params}}}$",
        fontsize=10,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _panel_nlc(axes[0, 0])
    _panel_bppi(axes[0, 1])
    _panel_nss(axes[1, 0])
    _panel_ter(axes[1, 1])
    fig.suptitle(
        "What each protocol metric probes (illustrative schematic, not reference-run data)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
