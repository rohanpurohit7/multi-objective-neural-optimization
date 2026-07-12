"""Generate the overall map from model optimization to cyber governance."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, width, height, title, body, color):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.025",
                           facecolor=color, edgecolor="#263238", linewidth=1.3)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * .72, title, ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(x + width / 2, y + height * .37, body, ha="center", va="center", fontsize=9)


def arrow(ax, start, end, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.5, color="#455A64"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .025,
                label, ha="center", fontsize=8, color="#37474F")


def main():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .04, .69, .25, .20, "1. Non-convex model space",
        "Multiple basins\nRNN · Transformer · Vector context", "#D9EAF7")
    box(ax, .375, .69, .25, .20, "2. Pareto frontier",
        "Missed attacks · False positives\nLatency · Complexity · Cost", "#DDF2E0")
    box(ax, .71, .69, .25, .20, "3. Confidence region",
        "Bootstrap intervals\nOverlap · uncertainty · robustness", "#FFF0CF")
    arrow(ax, (.29, .79), (.375, .79), "evaluate")
    arrow(ax, (.625, .79), (.71, .79), "quantify")

    box(ax, .10, .36, .23, .20, "4. Policy enforcement",
        "Allow · step-up MFA · block\nAsset criticality · risk appetite", "#FCE0DE")
    box(ax, .385, .36, .23, .20, "5. Cyber economics",
        "Expected-loss reduction\nLifecycle cost · analyst capacity", "#E8DFF2")
    box(ax, .67, .36, .23, .20, "6. Security strength",
        "Prevent · detect · delay\nContain · recover · learn", "#DDECF0")
    arrow(ax, (.82, .69), (.785, .56), "condition choice")
    arrow(ax, (.71, .69), (.50, .56), "price trade-off")
    arrow(ax, (.74, .69), (.215, .56), "set threshold")

    box(ax, .25, .07, .50, .19, "7. Defense-in-depth portfolio",
        "Identity → Endpoint → Network → Application → Data → Recovery\n"
        "Diversify failure modes; test common dependencies; monitor residual risk", "#E3F2E7")
    arrow(ax, (.215, .36), (.38, .26))
    arrow(ax, (.50, .36), (.50, .26))
    arrow(ax, (.785, .36), (.62, .26))

    ax.text(.5, .965, "Overall decision map: evidence informs policy—it does not dictate it",
            ha="center", va="center", fontsize=17, fontweight="bold")
    ax.text(.5, .925,
            "A preferred solution is conditional on objectives, confidence, economics, mission, and fallback layers",
            ha="center", va="center", fontsize=11, color="#455A64")
    output = Path(__file__).resolve().parents[1] / "artifacts" / "overall_cybersecurity_decision_map.svg"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

