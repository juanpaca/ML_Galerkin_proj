#!/usr/bin/env python3
"""Standalone convergence study script.

Calls src/convergence.py functions.
Usage:
    python convergence_study.py              # Classical + Exact RFB only
    python convergence_study.py --train-kan  # Also train KAN on-the-fly
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.convergence import convergence_study, print_table, plot_convergence


TEST_CASES = [
    (1e-3, 1.0, 0.0,  "Advection-dominated (Pe=500)"),
    (1e-3, 1.0, 10.0, "Reaction-dominated (Pe=500, rho=0.16)"),
    (1e-2, 1.0, 0.0,  "Moderate advection (Pe=50)"),
]

MODEL_PATH = "models/multi_bubble_model_1k.pt"


def load_or_train_kan():
    """Try to load saved model, or train from scratch."""
    from src.rfb_bubble import MultiKANBubble1D
    from src.rfb_training import generate_rfb_training_data_by_mode, train_multi_bubble_model

    if os.path.exists(MODEL_PATH):
        bubble = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
        try:
            import torch
            sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            bubble.load_state_dict(sd)
            bubble.eval()
            print(f"  Loaded KAN model from {MODEL_PATH}")
            return bubble
        except RuntimeError:
            print(f"  Saved model incompatible (old architecture). Retraining...")

    print(f"  Training KAN from scratch...")
    multi = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
    samples = generate_rfb_training_data_by_mode(
        n_samples=1000, h=1/16,
        eps_range=(1e-6, 1.0), sigma_range=(0.0, 10.0),
        residual_modes=("constant", "xi"),
    )
    train_multi_bubble_model(multi, samples, n_epochs=500, lr=1e-3, grad_weight=0.0, verbose=True)
    import torch
    torch.save(multi.state_dict(), MODEL_PATH)
    print(f"  Saved to {MODEL_PATH}")
    return multi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-kan", action="store_true",
                        help="Train KAN model on-the-fly (slow)")
    args = parser.parse_args()

    kan_model = None
    if args.train_kan:
        kan_model = load_or_train_kan()

    all_results = []
    all_labels = []
    for eps, beta, sigma, label in TEST_CASES:
        results = convergence_study(eps=eps, beta=beta, sigma=sigma,
                                    mesh_sizes=[4, 8, 16, 32, 64],
                                    kan_model=kan_model)
        print_table(results, title=label)
        all_results.append(results)
        all_labels.append(label)

    plot_convergence(all_results, all_labels, save_path="convergence_study.png")

    print(f"\n{'='*70}")
    print("  MESH-INDEPENDENCE PROOF")
    print(f"{'='*70}")
    print("  The log-log slopes (= convergence rates) should be the same")
    print("  across all mesh sizes for each method.")
    print("  Key: if KAN-RFB slope ~ Exact RFB slope, the learned bubbles")
    print("  are robust to mesh refinement.")
    print(f"{'='*70}")
