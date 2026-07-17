import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np

from src.rfb_exact import ExactRFBubble1D

fig, axes = plt.subplots(1, 3, figsize=(8, 2.4))

pairs = [
    (0.2,  r"$\mathrm{Pe} = 0.2$",  1.10),
    (1.5,  r"$\mathrm{Pe} = 1.5$",  1.15),
    (6.0,  r"$\mathrm{Pe} = 6.0$",  1.55),
]

xi = np.linspace(0, 1, 1001)

for ax, (pe, label, ymax) in zip(axes, pairs):
    bubble = ExactRFBubble1D(
        eps=1.0,
        beta=2.0 * pe,
        sigma=0.0,
        h=1.0,
        residual_mode="constant",
        n_points=2000,
    )
    b, _ = bubble.value_grad_numpy(xi)
    ax.plot(xi, b, color="#1F4E79", linewidth=2.0)
    ax.set_ylim(-0.05, ymax)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\xi$", fontsize=10)
    if ax == axes[0]:
        ax.set_ylabel(r"$b(\xi)$", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.tick_params(labelsize=8)

fig.tight_layout()

figure_dir = __file__.rsplit("/", 1)[0] + "/figures"
fig.savefig(f"{figure_dir}/three_bubbles.pdf", bbox_inches="tight")
fig.savefig(f"{figure_dir}/three_bubbles.png", dpi=150, bbox_inches="tight")
plt.close(fig)
