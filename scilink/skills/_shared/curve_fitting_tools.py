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


# Column-name fragments that strongly suggest the independent (X) variable, used
# to pick X from a >2-column table when no explicit hint is given.
_X_AXIS_HINTS = (
    "2theta", "two_theta", "twotheta", "theta", "angle", "wavelength", "wavenumber",
    "energy", "ev", "raman", "shift", "time", "freq", "frequency", "temperature",
    "temp", "field", "position", "distance", "voltage", "bias", "delay", "q", "x",
)
# Fragments suggesting a column is NOT the signal (uncertainty/weight columns) —
# avoided when choosing Y.
_NON_Y_HINTS = ("err", "error", "sigma", "std", "uncert", "weight", "noise")


def _choose_xy_indices(arr, system_info, column_names, log):
    """Pick (x_index, y_index) from a >2-column array.

    Priority: (1) explicit ``x_column``/``y_column`` in system_info (name or
    index); (2) a column whose name matches an axis hint as X + first non-error
    column as Y; (3) the single monotonic column as X; (4) first two columns.
    """
    n = arr.shape[1]
    si = system_info if isinstance(system_info, dict) else {}
    names = column_names or si.get("columns")
    low = [str(c).strip().lower() for c in names[:n]] if names and len(names) >= n else None

    def resolve(spec):
        if isinstance(spec, bool):
            return None
        if isinstance(spec, int) and 0 <= spec < n:
            return spec
        if isinstance(spec, str) and low:
            for i, c in enumerate(low):
                if c == spec.strip().lower():
                    return i
        return None

    xi, yi = resolve(si.get("x_column")), resolve(si.get("y_column"))
    if xi is not None and yi is not None and xi != yi:
        return xi, yi

    if low:  # name-hint heuristic
        x_guess = next((i for i, c in enumerate(low)
                        if any(h in c for h in _X_AXIS_HINTS)), None)
        if x_guess is not None:
            y_guess = next((i for i in range(n) if i != x_guess
                            and not any(b in low[i] for b in _NON_Y_HINTS)),
                           next((i for i in range(n) if i != x_guess), None))
            return x_guess, y_guess

    for i in range(n):  # monotonic-column heuristic
        col = arr[:, i]
        if col.size >= 2 and np.all(np.isfinite(col)) and \
                (np.all(np.diff(col) > 0) or np.all(np.diff(col) < 0)):
            return i, next(j for j in range(n) if j != i)

    return 0, 1  # default: first two columns


def select_xy_columns(data, system_info=None, logger_=None, column_names=None) -> np.ndarray:
    """Reduce array-like curve data to a 2-column ``(x, y)`` array.

    Curve fitting is a 1D (x, y) operation, but real files arrive with extra
    columns (an error/weight column, multiple channels) or in row layout. This
    centralizes the reduction: 1D → ``(index, y)``; row-major ``(2, N)`` →
    transposed; ``(N, 2)`` → oriented; ``(N, M>2)`` → X/Y selected (see
    ``_choose_xy_indices``) and the rest dropped (logged). Column SELECTION is an
    analysis decision and so lives here — lossless file-prep keeps all columns.
    """
    log = logger_ or logger
    arr = np.asarray(data)
    if arr.ndim == 1:
        return np.column_stack([np.arange(arr.size), arr])
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D curve data, got {arr.ndim}D")
    # Orient to (n_points, n_cols): curve data has many more points than columns.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    n_cols = arr.shape[1]
    if n_cols == 1:
        return np.column_stack([np.arange(arr.shape[0]), arr[:, 0]])
    if n_cols == 2:
        return _try_orient_xy(np.ascontiguousarray(arr))
    xi, yi = _choose_xy_indices(arr, system_info, column_names, log)
    if log:
        cols_desc = f" ({column_names})" if column_names else ""
        log.warning(
            f"select_xy_columns: {n_cols}-column data{cols_desc} reduced to "
            f"(X=col {xi}, Y=col {yi}); other columns dropped for fitting."
        )
    return _try_orient_xy(np.column_stack([arr[:, xi], arr[:, yi]]))


def load_curve_data(data_path: str, auto_orient: bool = True,
                    system_info: dict = None) -> np.ndarray:
    """
    Robustly loads curve data (X, Y) from various file formats.
    Handles .npy, .h5/.hdf5/.nxs (NeXus), CSV, TSV, and whitespace separation
    automatically.

    When ``auto_orient`` is True (default), the result is normalized to a
    2-column ``(X, Y)`` array via :func:`select_xy_columns`: a 2-column array is
    oriented so the monotonic (X) column comes first, and a >2-column table (an
    extra error/weight column, multiple channels) is reduced to the chosen X/Y
    pair — guided by ``system_info`` (``x_column``/``y_column`` or ``columns``)
    when given, else by an axis-name / monotonicity heuristic. Pass
    ``auto_orient=False`` for legacy raw-layout behavior (no reduction).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    def _finish(arr):
        return select_xy_columns(arr, system_info=system_info) if auto_orient else arr

    # Native numpy format
    if data_path.endswith('.npy'):
        return _finish(np.load(data_path))

    # NeXus / HDF5 — pull the signal and (when present) its axis so we
    # can return (X, Y) pairs for callers that expect a 2D layout.
    if data_path.lower().endswith(('.h5', '.hdf5', '.nxs')):
        from scilink.utils.hdf5_utils import load_hdf5_signal
        signal, axes = load_hdf5_signal(data_path, return_axes=True)
        if signal.ndim == 1 and axes and axes[0] is not None and axes[0].size == signal.size:
            return np.column_stack([axes[0], signal])
        return _finish(signal)

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
                return _finish(data)
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