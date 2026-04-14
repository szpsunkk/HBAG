from __future__ import annotations

"""Simple plotting helpers."""

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_lines(
    x: np.ndarray,
    y_map: Dict[str, Iterable[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: str | Path,
) -> None:
    plt.figure(figsize=(7, 4))
    for name, y in y_map.items():
        plt.plot(x, list(y), label=name, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
