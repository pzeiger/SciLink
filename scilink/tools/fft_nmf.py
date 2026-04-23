import os
import numpy as np
from scipy import fftpack, ndimage
from sklearn.decomposition import NMF
from skimage.util import view_as_windows
from skimage import io, color
import warnings

# --- Helper Function for Standalone Usage ---
def load_image(file_path):
    """Simple wrapper to load images using skimage."""
    try:
        img = io.imread(file_path)
        # Handle RGBA/RGB
        if img.ndim == 3 and img.shape[2] in [3, 4]:
            img = color.rgb2gray(img[:, :, :3])
        return img
    except Exception as e:
        raise ValueError(f"Could not load image at {file_path}: {e}")

class SlidingFFTNMF:
    def __init__(self, window_size_x=None, window_size_y=None, 
                 window_step_x=None, window_step_y=None,
                 interpolation_factor=2, zoom_factor=2, 
                 hamming_filter=True, components=4):
        """
        Sliding Window FFT with NMF unmixing.
        Supports both Single Images (2D) and Time-Series Stacks (3D).
        """
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.window_step_x = window_step_x
        self.window_step_y = window_step_y
        
        self.interpol_factor = interpolation_factor
        self.zoom_factor = zoom_factor
        self.hamming_filter = hamming_filter
        self.n_components = components
        
        # Internal state
        self.hamming_window = None
        self.windows_shape = None # (n_frames, n_windows_y, n_windows_x)
        self.fft_size = None
        
    def _calculate_window_params(self, image_shape):
        """Auto-calculates window parameters based on the spatial dimensions (H, W)."""
        # image_shape is expected to be (Height, Width) here
        height, width = image_shape
        
        # 1. Defaults for Window Size
        if self.window_size_x is None:
            val = max(32, min(128, height // 8))
            self.window_size_x = 2 ** int(np.log2(val))
        if self.window_size_y is None:
            val = max(32, min(128, width // 8))
            self.window_size_y = 2 ** int(np.log2(val))
            
        # 2. Defaults for Step Size (25% overlap is standard)
        if self.window_step_x is None:
            self.window_step_x = max(1, self.window_size_x // 4)
        if self.window_step_y is None:
            self.window_step_y = max(1, self.window_size_y // 4)
            
        # 3. Create Hamming Window
        bw2d = np.outer(np.hamming(self.window_size_x), np.ones(self.window_size_y))
        self.hamming_window = np.sqrt(bw2d * bw2d.T)

    def _extract_windows_from_frame(self, frame):
        """Extracts sliding windows from a SINGLE 2D frame."""
        # Pad if necessary
        pad_h = max(0, self.window_size_x - frame.shape[0])
        pad_w = max(0, self.window_size_y - frame.shape[1])
        if pad_h > 0 or pad_w > 0:
            frame = np.pad(frame, ((0, pad_h), (0, pad_w)), mode='constant')

        window_shape = (self.window_size_x, self.window_size_y)
        step = (self.window_step_x, self.window_step_y)
        
        # Extract windows: Shape becomes (n_win_y, n_win_x, win_h, win_w)
        windows = view_as_windows(frame, window_shape, step=step)
        
        # Return flattened list of windows and the grid shape
        grid_shape = windows.shape[:2]
        windows_flat = windows.reshape(-1, self.window_size_x, self.window_size_y)
        return windows_flat, grid_shape

    def make_windows(self, data):
        """
        Extract windows from 2D (Single) or 3D (Series) data.
        Performs GLOBAL normalization to preserve relative intensity changes.
        """
        data = data.astype(float)
        
        # 1. Global Normalization (Crucial for Time Series)
        d_min, d_max = np.min(data), np.max(data)
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        else:
            data = data - d_min # Handle flat images

        # 2. Determine input type
        if data.ndim == 2:
            # Single Image: Add dummy time dimension -> (1, H, W)
            data = data[np.newaxis, :, :]
            self.is_series = False
        elif data.ndim == 3:
            # Time Series: (Time, H, W)
            self.is_series = True
        else:
            raise ValueError(f"Input must be 2D or 3D array. Got {data.ndim}D")

        # 3. Initialize Parameters based on first frame
        self._calculate_window_params(data.shape[1:])
        
        all_windows = []
        
        # 4. Extract windows from every frame
        for t in range(data.shape[0]):
            windows, grid_shape = self._extract_windows_from_frame(data[t])
            all_windows.append(windows)
            
        # Store grid shape for reconstruction: (Time, Grid_Y, Grid_X)
        self.grid_shape = (data.shape[0], grid_shape[0], grid_shape[1])
        
        # Stack all windows into one massive array: (Total_Windows, H, W)
        # Total_Windows = Time * Grid_Y * Grid_X
        return np.vstack(all_windows)

    def process_fft(self, windows):
        """Compute FFT for a batch of windows."""
        n_windows = windows.shape[0]
        fft_results = []
        
        # Pre-calculate zoom indices to avoid doing it inside the loop
        cx, cy = self.window_size_x // 2, self.window_size_y // 2
        zoom_sz = max(1, self.window_size_x // (2 * self.zoom_factor))
        x_sl = slice(max(0, cx - zoom_sz), min(self.window_size_x, cx + zoom_sz))
        y_sl = slice(max(0, cy - zoom_sz), min(self.window_size_y, cy + zoom_sz))

        for i in range(n_windows):
            w = windows[i]
            if self.hamming_filter:
                w = w * self.hamming_window
                
            # FFT
            fft_res = fftpack.fftshift(fftpack.fft2(w))
            fft_mag = np.log1p(np.abs(fft_res)) # Log magnitude
            
            # Zoom
            zoomed = fft_mag[x_sl, y_sl]
            
            # Interpolate (Optional)
            if self.interpol_factor > 1:
                zoomed = ndimage.zoom(zoomed, self.interpol_factor, order=1)
                
            fft_results.append(zoomed)
            
        # Stack and pad if necessary (though shapes should be uniform)
        # We assume uniform shapes for efficiency here
        self.fft_size = fft_results[0].shape
        return np.array(fft_results)

    def run_nmf(self, fft_data):
        """
        Run NMF on the stacked FFT data.
        Returns reshaped components and abundances.
        """
        # Flatten: (N_Samples, H*W)
        n_samples = fft_data.shape[0]
        flat_data = fft_data.reshape(n_samples, -1)
        
        # Clean data
        flat_data = np.nan_to_num(flat_data)
        flat_data = np.maximum(0, flat_data)
        
        # Safety check for components
        n_comps = min(self.n_components, n_samples, flat_data.shape[1])
        if n_comps != self.n_components:
            warnings.warn(f"Reduced components from {self.n_components} to {n_comps} due to data size.")
            
        # --- Run NMF ---
        nmf = NMF(n_components=n_comps, init='random', random_state=42, max_iter=500)
        W = nmf.fit_transform(flat_data) # Abundances (N_Samples, n_comps)
        H = nmf.components_            # Components (n_comps, Features)
        
        # --- Reshape Results ---
        
        # 1. Components: (n_comps, fft_h, fft_w)
        components_img = H.reshape(n_comps, self.fft_size[0], self.fft_size[1])
        
        # 2. Abundances: Reconstruct spatial/temporal structure
        # W is currently (Time * Grid_Y * Grid_X, n_comps)
        # We need to reshape it back to the grid structure
        t_steps, grid_y, grid_x = self.grid_shape
        
        # Reshape to (Time, Grid_Y, Grid_X, n_comps)
        abundances_grid = W.reshape(t_steps, grid_y, grid_x, n_comps)
        
        # Transpose to standard format: (Time, n_comps, Grid_Y, Grid_X)
        abundances_final = abundances_grid.transpose(0, 3, 1, 2)
        
        # If input was single image, squeeze out the time dimension
        if not self.is_series:
            abundances_final = abundances_final[0] # -> (n_comps, Grid_Y, Grid_X)
            
        return components_img, abundances_final

    def analyze(self, image_input, output_dir=None):
        """
        Main method. Handles file loading, analysis, and saving.
        
        Returns:
            components: (n_comps, h, w)
            abundances: (Time, n_comps, h, w) OR (n_comps, h, w) for single image
        """
        # 1. Load Data
        if isinstance(image_input, str):
            data = load_image(image_input)
            if output_dir is None:
                name = os.path.splitext(os.path.basename(image_input))[0]
                output_dir = f"{name}_results"
        elif isinstance(image_input, np.ndarray):
            data = image_input
            if output_dir is None:
                output_dir = "nmf_results"
        else:
            raise TypeError("Input must be path (str) or array")
            
        # 2. Execute Pipeline
        print(f"Processing data with shape {data.shape}...")
        all_windows = self.make_windows(data)
        
        print(f"Computed {len(all_windows)} total windows. Running FFT...")
        fft_data = self.process_fft(all_windows)
        
        print("Running NMF...")
        components, abundances = self.run_nmf(fft_data)
        
        # 3. Save Results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "components.npy"), components)
            np.save(os.path.join(output_dir, "abundances.npy"), abundances)
            print(f"Saved results to {output_dir}")
            
        return components, abundances


def run_fft_nmf_analysis(image_array, params=None):
    """Run sliding FFT + NMF decomposition on an image.

    Convenience wrapper around :class:`SlidingFFTNMF` that mirrors the
    ``run_sam_analysis`` API so generated scripts can call it directly.

    Args:
        image_array: 2D numpy array (grayscale image).
        params: Optional dict with:
            - window_size (int): Side length in pixels (default: auto).
            - n_components (int): Number of NMF components (default: 4).
            - step_fraction (float): Window step as fraction of window
              size — 0.25 means 75 % overlap (default: 0.25).

    Returns:
        dict with ``components`` (n, h, w), ``abundances`` (n, gh, gw),
        ``n_components``, ``window_size``, and ``grid_shape``.
    """
    params = params or {}
    window_size = params.get("window_size")
    n_components = params.get("n_components", 4)
    step_fraction = params.get("step_fraction", 0.25)

    kwargs = {"components": n_components}
    if window_size is not None:
        kwargs["window_size_x"] = window_size
        kwargs["window_size_y"] = window_size
        step = max(1, int(window_size * step_fraction))
        kwargs["window_step_x"] = step
        kwargs["window_step_y"] = step

    analyzer = SlidingFFTNMF(**kwargs)

    # Call pipeline steps directly to avoid stdout prints and
    # unwanted file saves that analyze() does by default.
    data = image_array.astype(float)
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        data = (data - d_min) / (d_max - d_min)
    windows = analyzer.make_windows(data)
    fft_data = analyzer.process_fft(windows)
    components, abundances = analyzer.run_nmf(fft_data)

    return {
        "components": components,
        "abundances": abundances,
        "n_components": int(components.shape[0]),
        "window_size": (analyzer.window_size_x, analyzer.window_size_y),
        "grid_shape": (
            analyzer.grid_shape[1],
            analyzer.grid_shape[2],
        ),
    }


# =============================================================================
# TOOL SPEC
# =============================================================================

from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="run_fft_nmf_analysis",
    description=(
        "Sliding-window FFT + NMF decomposition. Factorizes local frequency patterns "
        "across an image into a small set of basis spectra and their spatial abundance maps."
    ),
    import_line="from scilink.tools.fft_nmf import run_fft_nmf_analysis",
    signature="run_fft_nmf_analysis(image_array, params=None) -> dict",
    agents=["image_analysis"],
    when_to_use=(
        "Materials with defects, disorder, multiple phases, or spatially varying "
        "structure — or when the objective involves characterizing disorder, phase "
        "separation, or local symmetry variations. Also useful for any image where "
        "the goal is periodic patterns, symmetries, or electronic/lattice patterns "
        "rather than individual atom positions.\n"
        "\n"
        "With a window size tuned to the spatial scale of the repeating features and a "
        "simple post-processing step (e.g. inspecting components and abundance maps, "
        "thresholding or clustering abundances to localize distinct regions), this is "
        "already a complete Tier 1 pipeline — no extra processing steps are required.\n"
        "\n"
        "**What the method guarantees vs. does not:** FFT-NMF is a data-driven "
        "non-negative decomposition. It reliably produces (a) non-negative spectral "
        "components with low reconstruction error, (b) spatially coherent abundance "
        "maps when the image contains spatial variation, and (c) visually distinct "
        "components when the image contains distinct patterns. It does NOT assign "
        "semantic labels to components (e.g. one component = 'crystalline', another "
        "= 'disordered') — that is for the user to interpret. Plan's quality_criteria "
        "should target coherent, non-noise outputs rather than idealized textbook "
        "patterns or strict semantic separation the method cannot deliver. "
        "When signals co-vary spatially (lattice × LDOS envelope, topography × "
        "composition, etc.), components typically mix these signals rather than "
        "isolating each — interpret components as basis patterns, not "
        "physics-separated modes."
    ),
    parameters={
        "image_array": {
            "type": "ndarray",
            "description": "2D grayscale numpy array.",
        },
        "params": {
            "type": "dict",
            "description": (
                "Optional. Keys: "
                "window_size (int pixels, default auto — pick based on the spatial scale "
                "of repeating features), "
                "n_components (int, default 4 — number of distinct patterns expected), "
                "step_fraction (float, default 0.25 — window step as a fraction of window "
                "size; 0.25 = 75% overlap)."
            ),
        },
    },
    required=["image_array"],
    returns=(
        "dict with 'components' (ndarray, shape (n_components, fft_h, fft_w)) — each "
        "is a 2D FFT power spectrum for one dominant frequency pattern; 'abundances' "
        "(ndarray, shape (n_components, grid_h, grid_w)) — spatial maps of where each "
        "component is present; 'n_components' (int); 'window_size' (tuple of two ints, "
        "(width, height)); 'grid_shape' (tuple of two ints, (grid_h, grid_w))."
    ),
    example=(
        "result = run_fft_nmf_analysis(image_array, params={'n_components': 4})\n"
        "np.save('nmf_components.npy', result['components'])\n"
        "np.save('abundance_maps.npy', result['abundances'])"
    ),
)
