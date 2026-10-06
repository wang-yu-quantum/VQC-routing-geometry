from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(rows, FIGURES):
    colors = {2: "#3C6EAF", 3: "#D6675B", 4: "#2A9D8F", 5: "#A06DA8"}
    offsets = {2: -0.12, 3: -0.04, 4: 0.04, 5: 0.12}
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.05), constrained_layout=True, gridspec_kw={"wspace": 0.08}
    )
    rgrid = np.arange(0, 6)
    axes[0].plot(
        rgrid,
        1 - 2.0 ** (-rgrid),
        color="#30343B",
        lw=1.65,
        label="Theory $1-2^{-r}$",
        zorder=2,
    )
    for m in range(2, 6):
        subset = [r for r in rows if r["m"] == m]
        deficit = np.array([m - r["k"] for r in subset], float) + offsets[m]
        axes[0].scatter(
            deficit,
            [r["constructor_loss"] for r in subset],
            color=colors[m],
            edgecolor="white",
            linewidth=0.55,
            s=38,
            label=f"$m={m}$",
            zorder=3,
        )
    axes[0].set(
        xlabel="Channel deficit $r=m-k$",
        ylabel="Minimum infidelity $\\mathcal{L}^\\star$",
        xlim=(-0.28, 5.28),
        ylim=(-0.035, 1.02),
        xticks=range(6),
    )
    axes[0].legend(
        frameon=False,
        ncol=2,
        fontsize=7.2,
        loc="lower right",
        handletextpad=0.45,
        columnspacing=0.85,
    )
    axes[0].set_title(
        "Tight capacity floor",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=6,
        color="#30343B",
    )
    for m in range(2, 6):
        subset = [r for r in rows if r["m"] == m and r["k"] < m]
        deficit = np.array([m - r["k"] for r in subset], float) + offsets[m]
        ambient = np.array([r["ambient_state_gradient_norm"] for r in subset])
        projected = np.array(
            [max(r["state_projected_gradient_norm"], 1e-16) for r in subset]
        )
        for x, y0, y1 in zip(deficit, projected, ambient):
            axes[1].plot(
                [x, x], [y0, y1], color=colors[m], alpha=0.16, lw=0.8, zorder=1
            )
        axes[1].scatter(
            deficit,
            ambient,
            color=colors[m],
            edgecolor="white",
            linewidth=0.4,
            marker="o",
            s=34,
            zorder=3,
        )
        axes[1].scatter(
            deficit,
            projected,
            facecolor="white",
            edgecolor=colors[m],
            marker="D",
            s=31,
            linewidth=1.0,
            zorder=3,
        )
    axes[1].scatter(
        [], [], color="#4A4A4A", marker="o", s=30, label="Ambient $\\|G_{\\rm st}\\|$"
    )
    axes[1].scatter(
        [],
        [],
        facecolor="white",
        edgecolor="#4A4A4A",
        marker="D",
        s=28,
        label="Visible $\\|\\Pi_{\\mathcal{T}^{\\rm st}}G_{\\rm st}\\|$",
    )
    axes[1].set(
        yscale="log",
        ylim=(3e-17, 2.2),
        yticks=[1e-16, 1e-12, 1e-08, 0.0001, 1],
        xlim=(0.58, 5.42),
        xticks=[1, 2, 3, 4, 5],
        xlabel="Channel deficit $r=m-k$",
        ylabel="State-gradient norm",
    )
    axes[1].legend(frameon=False, fontsize=7.2, loc="center right", handletextpad=0.55)
    axes[1].text(
        0.035,
        0.055,
        "visible values $<10^{-16}$ shown at $10^{-16}$",
        transform=axes[1].transAxes,
        fontsize=6.7,
        color="#555B63",
    )
    axes[1].set_title(
        "Ambient slope remains hidden",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=6,
        color="#30343B",
    )
    for ax, label in zip(axes, "ab"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18, lw=0.65, color="#7A7F87")
        ax.tick_params(labelsize=8.2)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.text(
            -0.13,
            1.1,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
        )
    pdf = FIGURES / "fig_bell_floor.pdf"
    fig.savefig(
        pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)
