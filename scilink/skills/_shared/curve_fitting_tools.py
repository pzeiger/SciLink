import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import logging
import os

logger = logging.getLogger(__name__)


def _try_orient_xy(data: np.ndarray) -> np.ndarray:
    """Return a 2-column curve array with the monotonic (X-like) column first.

    Monotonicity is the strongest portable cue that a column is the
    independent variable (time, wavelength, energy). When exactly one
    column is monotonic we use it to disambiguate; when both or neither
    are, the heuristic is ambiguous and we leave the input untouched so
    well-formed data is not perturbed. Non-2D inputs and shapes other
    than (N, 2) pass through unchanged.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        return data
    col0, col1 = data[:, 0], data[:, 1]
    if not (np.isfinite(col0).all() and np.isfinite(col1).all()):
        return data

    def _mono(a: np.ndarray) -> bool:
        if a.size < 2:
            return False
        diffs = np.diff(a)
        return bool(np.all(diffs > 0) or np.all(diffs < 0))

    if _mono(col1) and not _mono(col0):
        logger.info(
            "load_curve_data: swapped columns — col 1 is monotonic but col 0 is not, "
            "so col 1 is treated as X."
        )
        return np.ascontiguousarray(data[:, ::-1])
    return data


def load_curve_data(data_path: str, auto_orient: bool = True) -> np.ndarray:
    """
    Robustly loads curve data (X, Y) from various file formats.
    Handles .npy, .h5/.hdf5/.nxs (NeXus), CSV, TSV, and whitespace separation
    automatically.

    When ``auto_orient`` is True (default), a 2-column result whose first
    column is non-monotonic but second column is monotonic is reoriented so
    X comes first — fixing inputs where intensity precedes the independent
    variable (e.g. CSVs whose header is ``intensity, time``). Pass
    ``auto_orient=False`` for legacy raw-layout behavior.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    # Native numpy format
    if data_path.endswith('.npy'):
        data = np.load(data_path)
        return _try_orient_xy(data) if auto_orient else data

    # NeXus / HDF5 — pull the signal and (when present) its axis so we
    # can return (X, Y) pairs for callers that expect a 2D layout.
    if data_path.lower().endswith(('.h5', '.hdf5', '.nxs')):
        from scilink.utils.hdf5_utils import load_hdf5_signal
        signal, axes = load_hdf5_signal(data_path, return_axes=True)
        if signal.ndim == 1 and axes and axes[0] is not None and axes[0].size == signal.size:
            return np.column_stack([axes[0], signal])
        return _try_orient_xy(signal) if auto_orient else signal

    attempts = [
        dict(),                                # whitespace-delimited, no header
        dict(skiprows=1),                      # whitespace-delimited, skip header
        dict(delimiter=','),                   # CSV, no header
        dict(delimiter=',', skiprows=1),       # CSV, skip header
    ]

    for kw in attempts:
        try:
            data = np.loadtxt(data_path, **kw)
            if data.size > 0:
                return _try_orient_xy(data) if auto_orient else data
        except Exception:
            pass

    # If all fail, raise descriptive error
    raise ValueError(f"Unsupported file format or invalid data structure in {data_path}.")

def plot_curve_to_bytes(curve_data: np.ndarray, system_info: dict, title_suffix: str = "") -> bytes:
    """
    Plots a 1D curve and returns the image as bytes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve_data[:, 0], curve_data[:, 1], 'b.', markersize=4)
    
    plot_title = system_info.get("title") or "Data"
    ax.set_title(plot_title + title_suffix)

    xlabel_text = system_info.get("xlabel") or "X-axis"
    ax.set_xlabel(xlabel_text)

    ylabel_text = system_info.get("ylabel") or "Y-axis"
    ax.set_ylabel(ylabel_text)
    
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)
    return image_bytes