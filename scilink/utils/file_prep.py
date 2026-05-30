"""Codegen-backed file preparation: split a combined data+metadata file into
separate data + metadata files BEFORE delegation to a specialist.

STRICT SCOPE — this is the meta agent's ONLY code-generation surface, and it is
restricted to **lossless file repackaging**. The generated script may separate
existing data from metadata and reconstruct the original; it must NOT transform,
scale, filter, fit, resample, denoise, or analyze anything. Analysis is always
delegated.

Enforcement is layered:
  1. narrow prompt (read input → write data + metadata, values unchanged);
  2. ROUND-TRIP verification — the script must also reconstruct the original from
     its two outputs, and we check reconstruction ≈ original. This confirms a
     faithful reconstruction PATH exists; it does NOT, by itself, prove the data
     leg (`data_path`, the file actually delegated) is untransformed, because the
     reconstruction is produced by the same generated script and is not forced to
     derive solely from the two split outputs. The data leg's fidelity rests on
     the prompt + import guard; the round-trip is a strong corroborating signal,
     not a hermetic proof. If the round-trip can't be verified, the split is
     REJECTED;
  3. static import guard (no analysis libraries — note numpy/pandas are allowed
     and can transform, so this narrows but does not eliminate the surface);
  4. explicit documentation on the tool.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

# Modules that signal analysis/computation rather than IO/parsing. A generated
# prep script importing any of these (or a submodule of one) is rejected before
# execution. Entries are matched by dotted PREFIX, so `scipy.optimize` denies
# `scipy.optimize.*` but leaves IO-only `scipy.io` / `scipy.sparse` available —
# the granularity is deliberate (classic MATLAB .mat needs `scipy.io`).
_ANALYSIS_IMPORT_DENYLIST = (
    "sklearn", "scipy.optimize", "scipy.signal", "scipy.ndimage",
    "scipy.interpolate", "scipy.stats", "scipy.fft", "scipy.fftpack",
    "lmfit", "torch", "tensorflow", "cv2", "skimage", "statsmodels",
    "matplotlib",
)

_PROMPT = '''You prepare a scientific data file for downstream analysis by
SEPARATING its data from its metadata. This is a LOSSLESS REPACKAGING step ONLY —
you are NOT analyzing anything.

Input file: {path}

Structural probe (evidence — do not assume beyond it):
{probe}

Common layouts you may encounter (illustrative, not exhaustive — let the probe
decide):
- A `.dat`/`.txt`/`.csv` with a metadata HEADER block (comment lines, `key: value`
  or `key = value` pairs, a units row) ABOVE the numeric data rows.
- A MATLAB `.mat` (or `.npz`/HDF5) where some keys are arrays (data) and others
  are scalars/strings (metadata).
- An HDF5/NeXus dataset carrying metadata in attributes alongside the array.
- A `.tif`/`.tiff` carrying acquisition metadata in TIFF tags / ImageDescription
  (ImageJ `key=value` block or OME-XML) alongside the pixel array.
Identify which bytes/keys/lines are data vs metadata from the probe, then:

Write a Python script that:
1. Reads the input file.
2. Splits it into (a) the primary numerical DATA (the array / table / signal) and
   (b) the METADATA — everything that is NOT the data values: headers, attributes,
   comments, units, calibration, axes, acquisition parameters, etc.
3. Writes the DATA to a file whose path you choose under the output directory
   "{out_dir}" — use `.npy` for an array/cube or `.csv` for a table.
4. Writes the METADATA to "{metadata_out}" as a single JSON object. Put the
   HUMAN-MEANINGFUL scientific metadata at the TOP LEVEL with clean keys
   (technique, instrument, sample, units, axes, wavelength, ...). If you need
   extra bookkeeping to reconstruct the original byte-for-byte (line endings,
   dtypes, column order, key order), nest ALL of it under a single
   "_reconstruction" key so downstream sees clean metadata. A value can be both
   meaningful AND needed for reconstruction — when scientific metadata lives in a
   container's attributes/keys (HDF5 attrs, .npz/.mat keys), surface it at the
   TOP LEVEL and, if needed, also record its placement under "_reconstruction";
   do not relegate it to "_reconstruction" alone.
5. RECONSTRUCTS the original file from those two outputs and writes it to
   "{recon_out}" (same format as the input). We will verify it matches the input.
6. Prints exactly one line:
   PREP_RESULT_JSON:{{"data_out": "<abs path>", "metadata_out": "{metadata_out}", "recon_out": "{recon_out}"}}

HARD CONSTRAINTS (a violation fails verification):
- DO NOT transform, scale, normalize, filter, fit, resample, denoise, crop, or
  analyze the data. The data you write MUST be the original values, unchanged.
- The reconstruction MUST reproduce the original file's data and metadata.
- Allowed libraries ONLY: numpy, pandas, json, csv, struct, io, re, h5py,
  scipy.io (for MATLAB .mat read/write ONLY), PIL/Pillow (for TIFF tags /
  ImageDescription), os, pathlib. Do NOT import analysis libraries — sklearn,
  lmfit, torch, cv2, skimage, matplotlib, or any other scipy submodule
  (optimize, signal, ndimage, interpolate, stats, fft, ...).
- No network, no plotting.

Respond with a JSON object: {{"script": "<the full python script>"}}
'''


def _extract_script(text: str) -> str:
    """Pull the script string out of the model response (JSON or fenced)."""
    if not text:
        return ""
    # Prefer a {"script": "..."} JSON object.
    try:
        m = re.search(r'\{.*"script".*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0)).get("script", "")
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to a fenced code block.
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _denied_module(name: str) -> str | None:
    """Return the denylist entry a dotted module name matches by prefix, else None.

    ``scipy.optimize`` matches ``scipy.optimize`` and ``scipy.optimize.foo`` but
    NOT ``scipy.io``; ``sklearn`` matches ``sklearn`` and any ``sklearn.*``.
    """
    parts = name.split(".")
    for entry in _ANALYSIS_IMPORT_DENYLIST:
        ep = entry.split(".")
        if parts[: len(ep)] == ep:
            return entry
    return None


def _static_guard(script: str) -> str | None:
    """Return a rejection reason if the script imports a denied analysis module.

    Parses the script and inspects ``import``/``from`` statements so the dotted
    denylist is honored precisely (e.g. ``from scipy import optimize`` is caught
    while ``from scipy import io`` is allowed). Not a sandbox — dynamic imports
    (``__import__``) are out of scope; the round-trip check is the real net.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # Unparseable code can't run, so it can't do harm; let execution surface
        # the syntax error. Conservative root-level fallback just in case.
        for entry in _ANALYSIS_IMPORT_DENYLIST:
            root = entry.split(".")[0]
            if re.search(rf"^\s*(import|from)\s+{re.escape(root)}\b", script, re.MULTILINE):
                return f"Generated prep script imports a forbidden module ('{root}')."
        return None

    def reason(hit: str) -> str:
        return (
            f"Generated prep script imports a forbidden analysis module "
            f"('{hit}'). File prep must use IO/parsing libraries only."
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _denied_module(alias.name)
                if hit:
                    return reason(hit)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — not a third-party analysis lib
                continue
            mod = node.module or ""
            hit = _denied_module(mod)
            if hit:
                return reason(hit)
            for alias in node.names:  # from pkg import submod
                hit = _denied_module(f"{mod}.{alias.name}" if mod else alias.name)
                if hit:
                    return reason(hit)
    return None


def _members_equal(a, b) -> bool:
    """Compare two arrays type-appropriately: numeric within tolerance, else exact.

    Combined containers (e.g. ``.npz``) routinely hold non-numeric members —
    string/object metadata keys — alongside the numeric data, so a blanket
    ``np.allclose`` would raise on those and falsely fail the round-trip.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return bool(np.allclose(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _attrs_equal(a: dict, b: dict) -> bool:
    """Compare two attribute dicts (HDF5 attrs / .mat fields) member-wise."""
    if set(a) != set(b):
        return False
    return all(_members_equal(a[k], b[k]) for k in a)


def _h5_equal(original: Path, recon: Path) -> bool:
    """Content-aware HDF5 comparison: a faithful re-write is rarely byte-identical
    (chunking, compression, attribute/dataset order, timestamps all vary), so we
    compare the group/dataset tree and attributes instead. ``h5py`` is already an
    allowed prep library; if it is unavailable we cannot verify → reject.
    """
    try:
        import h5py
    except Exception:
        return False

    def collect(f):
        items = {"/": ("group", None, dict(f.attrs))}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items[name] = ("dataset", obj[()], dict(obj.attrs))
            else:  # group
                items[name] = ("group", None, dict(obj.attrs))

        f.visititems(visit)
        return items

    try:
        with h5py.File(original, "r") as fa, h5py.File(recon, "r") as fb:
            a, b = collect(fa), collect(fb)
            if set(a) != set(b):
                return False
            for k in a:
                kind_a, val_a, attrs_a = a[k]
                kind_b, val_b, attrs_b = b[k]
                if kind_a != kind_b or not _attrs_equal(attrs_a, attrs_b):
                    return False
                if kind_a == "dataset" and not _members_equal(val_a, val_b):
                    return False
            return True
    except Exception:
        return False


def _mat_equal(original: Path, recon: Path) -> bool:
    """Content-aware MATLAB .mat comparison. Classic v5/6/7 .mat go through
    ``scipy.io.loadmat`` (an IO module, allowed); v7.3 .mat are HDF5 and fall back
    to :func:`_h5_equal`. Comparison ignores loadmat's ``__header__`` /
    ``__version__`` / ``__globals__`` bookkeeping (the header carries a timestamp).
    """
    try:
        from scipy.io import loadmat
    except Exception:
        return _h5_equal(original, recon)  # scipy absent → try HDF5 (v7.3 .mat)
    try:
        a, b = loadmat(original), loadmat(recon)
    except NotImplementedError:
        return _h5_equal(original, recon)  # v7.3 .mat is HDF5 under the hood
    except Exception:
        return False
    skip = {"__header__", "__version__", "__globals__"}
    ka = {k for k in a if k not in skip}
    kb = {k for k in b if k not in skip}
    if ka != kb:
        return False
    return all(_members_equal(a[k], b[k]) for k in ka)


def _tif_equal(original: Path, recon: Path) -> bool:
    """Content-aware TIFF comparison: pixel array(s) numeric-tolerant, tags exact.

    A faithful TIFF rewrite is rarely byte-identical (encoder, strip/tile layout,
    tag order), so compare decoded pixels per page plus the tag dict. ``PIL`` is
    an allowed prep library; if unavailable we cannot verify → reject.
    """
    try:
        from PIL import Image
    except Exception:
        return False
    try:
        with Image.open(original) as ia, Image.open(recon) as ib:
            na, nb = getattr(ia, "n_frames", 1), getattr(ib, "n_frames", 1)
            if na != nb:
                return False
            for i in range(na):
                ia.seek(i)
                ib.seek(i)
                if not _members_equal(np.asarray(ia), np.asarray(ib)):
                    return False
                ta = dict(getattr(ia, "tag_v2", {}) or {})
                tb = dict(getattr(ib, "tag_v2", {}) or {})
                if set(ta) != set(tb):
                    return False
                # Tags must be reproduced exactly (metadata losslessness).
                if any(str(ta[k]) != str(tb.get(k)) for k in ta):
                    return False
            return True
    except Exception:
        return False


def _roundtrip_ok(original: Path, recon: Path) -> bool:
    """Tolerant lossless check: does ``recon`` reproduce ``original``?"""
    try:
        ob = original.read_bytes()
        rb = recon.read_bytes()
    except OSError:
        return False
    if ob == rb:
        return True  # byte-identical
    ext = original.suffix.lower()
    try:
        if ext == ".npy":
            return _members_equal(np.load(original, allow_pickle=True),
                                  np.load(recon, allow_pickle=True))
        if ext == ".npz":
            a, b = np.load(original, allow_pickle=True), np.load(recon, allow_pickle=True)
            return set(a.files) == set(b.files) and all(
                _members_equal(a[k], b[k]) for k in a.files
            )
        if ext in (".h5", ".hdf5", ".nxs", ".nx"):
            return _h5_equal(original, recon)
        if ext == ".mat":
            return _mat_equal(original, recon)
        if ext in (".tif", ".tiff"):
            return _tif_equal(original, recon)
        if ext in (".csv", ".tsv", ".txt", ".dat"):
            return _text_equal_tolerant(original.read_text(errors="replace"),
                                        recon.read_text(errors="replace"))
    except Exception:
        return False
    return False  # unknown/binary that wasn't byte-identical → cannot verify


def _text_equal_tolerant(a: str, b: str) -> bool:
    """Token-wise compare: numbers within tolerance, other tokens exact."""
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if np.isclose(float(x), float(y), equal_nan=True):
                continue
        except ValueError:
            pass
        return False
    return True


def _verify(input_path: Path, data_out: Path, meta_out: Path,
            recon_out: Path) -> tuple[bool, str]:
    """Verify the split is well-formed AND lossless. Returns (ok, reason).

    ``data_out`` is the model's chosen data path (pulled from PREP_RESULT_JSON);
    ``meta_out`` / ``recon_out`` are the template-fixed paths the caller owns, so
    we verify (and later return) those rather than trusting the model's echo.
    """
    if not data_out.exists():
        return False, f"data_out not written: {data_out}"
    try:
        if data_out.suffix.lower() == ".npy":
            arr = np.load(data_out, allow_pickle=False)
            if arr.size == 0:
                return False, "data_out array is empty"
        else:  # tabular
            if data_out.stat().st_size == 0:
                return False, "data_out file is empty"
    except Exception as e:
        return False, f"data_out does not load as data: {e}"

    if not meta_out.exists():
        return False, f"metadata_out not written: {meta_out}"
    try:
        meta = json.loads(meta_out.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return False, f"metadata_out is not valid JSON: {e}"
    # Reject empty metadata: a genuinely metadata-light file fails here and the
    # meta bounces it back to the user rather than delegating a no-op split.
    if not isinstance(meta, dict) or not meta:
        return False, "metadata_out is empty or not a JSON object"

    if not recon_out.exists():
        return False, f"recon_out (round-trip check) not written: {recon_out}"
    if not _roundtrip_ok(input_path, recon_out):
        return False, (
            "round-trip FAILED: reconstruction does not match the original — the "
            "split is not lossless (or the data was transformed). Rejected."
        )
    return True, "lossless split verified"


def prepare_inputs(input_path, model, executor, output_dir, probe=None,
                   logger=None, max_retries: int = 1) -> dict:
    """Split a combined data+metadata file into data + metadata files (codegen).

    Args:
        input_path: the combined file to split.
        model: LLM with ``.generate_content(prompt).text``.
        executor: a ``ScriptExecutor`` (runs the generated script in a sandbox).
        output_dir: directory for the outputs.
        probe: optional structural probe dict (e.g. from the meta's file probe);
            a compact fallback is used when omitted.
        logger: optional logger.
        max_retries: extra attempts after the first (default 1).

    Returns a dict: on success ``{status:"success", data_path, metadata_path,
    attempts}``; otherwise ``{status:"error", message, attempts}``. The
    reconstruction used for the round-trip check is deleted once verified.
    The caller decides how to handle an error — the meta surfaces it to the user
    and asks how to proceed rather than silently delegating an unsplit file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    stem = re.sub(r"[^0-9A-Za-z_-]", "_", input_path.stem) or "input"
    # Per-file subdirectory keyed on the absolute path: two uploads sharing a
    # basename (e.g. runA/scan.tif and runB/scan.tif) must not clobber each
    # other's metadata/recon/data outputs in a shared "prepared/" dir.
    key = hashlib.sha1(str(input_path.resolve()).encode()).hexdigest()[:8]
    file_out = output_dir / f"{stem}_{key}"
    file_out.mkdir(parents=True, exist_ok=True)
    metadata_out = file_out / f"{stem}_metadata.json"
    recon_out = file_out / f"{stem}_reconstructed{input_path.suffix}"

    probe_str = json.dumps(probe, indent=2, default=str) if probe else \
        f"(no probe) extension={input_path.suffix}, size={input_path.stat().st_size} bytes"
    base_prompt = _PROMPT.format(
        path=str(input_path), probe=probe_str, out_dir=str(file_out),
        metadata_out=str(metadata_out), recon_out=str(recon_out),
    )

    prompt = base_prompt
    last_reason = ""
    for attempt in range(1, max_retries + 2):
        if logger:
            logger.info(f"📦 File prep (attempt {attempt}): generating split script for {input_path.name}")
        try:
            script = _extract_script(model.generate_content(prompt).text)
        except Exception as e:
            last_reason = f"code generation failed: {e}"
            prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{last_reason}\nFix it."
            continue
        if not script:
            last_reason = "model returned no script"
            prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{last_reason}\nFix it."
            continue

        guard = _static_guard(script)
        if guard:
            last_reason = guard
            if logger:
                logger.warning(f"📦 File prep rejected: {guard}")
            prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{guard}\nUse IO/parsing libraries only."
            continue

        exec_res = executor.execute_script(script, working_dir=str(file_out))
        if exec_res.get("status") != "success":
            last_reason = f"script execution failed: {exec_res.get('message', '')[:600]}"
            prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{last_reason}\nFix the script."
            continue

        m = re.search(r"PREP_RESULT_JSON:(\{.*\})", exec_res.get("stdout", ""))
        if not m:
            last_reason = "script did not print PREP_RESULT_JSON with the output paths"
            prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{last_reason}\nPrint the result line."
            continue
        try:
            result = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            last_reason = f"PREP_RESULT_JSON not valid JSON: {e}"
            prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{last_reason}\nFix the output line."
            continue

        data_out = Path(result.get("data_out", ""))
        ok, reason = _verify(input_path, data_out, metadata_out, recon_out)
        if ok:
            # The reconstruction was only needed for the round-trip check; it is a
            # full duplicate of the input, so drop it rather than leave it behind.
            recon_out.unlink(missing_ok=True)
            if logger:
                logger.info(f"✅ File prep OK ({reason}): data={data_out}, metadata={metadata_out}")
            return {
                "status": "success",
                "data_path": str(data_out),
                "metadata_path": str(metadata_out),
                "attempts": attempt,
            }
        last_reason = reason
        if logger:
            logger.warning(f"📦 File prep verification failed: {reason}")
        prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{reason}\nProduce a lossless split."

    recon_out.unlink(missing_ok=True)  # drop any leftover from the last attempt
    return {
        "status": "error",
        "message": f"Could not produce a verified lossless split after "
                   f"{max_retries + 1} attempt(s): {last_reason}",
        "attempts": max_retries + 1,
    }
