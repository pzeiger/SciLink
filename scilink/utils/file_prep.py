"""Codegen-backed file preparation: split a combined data+metadata file into
separate data + metadata files BEFORE delegation to a specialist.

STRICT SCOPE — this is the meta agent's ONLY code-generation surface, and it is
restricted to **lossless file repackaging**. The generated script may separate
existing data from metadata and reconstruct the original; it must NOT transform,
scale, filter, fit, resample, denoise, or analyze anything. Analysis is always
delegated.

Enforcement is layered, the middle layer mechanical:
  1. narrow prompt (read input → write data + metadata, values unchanged);
  2. ROUND-TRIP verification — the script must also reconstruct the original from
     its two outputs, and we check reconstruction ≈ original. A script that
     transformed the data cannot round-trip, so "prep only" is enforced, not just
     requested. If losslessness can't be verified, the split is REJECTED (the
     caller falls back to delegating the file as-is);
  3. static import guard (no analysis libraries);
  4. explicit documentation on the tool.
"""

import json
import re
from pathlib import Path

import numpy as np

# Libraries that signal analysis/computation rather than IO/parsing. A generated
# prep script importing any of these is rejected before execution.
_ANALYSIS_IMPORT_DENYLIST = (
    "sklearn", "scipy.optimize", "scipy.signal", "scipy.ndimage",
    "scipy.interpolate", "scipy.stats", "lmfit", "torch", "tensorflow",
    "cv2", "skimage", "statsmodels", "matplotlib",
)

_PROMPT = '''You prepare a scientific data file for downstream analysis by
SEPARATING its data from its metadata. This is a LOSSLESS REPACKAGING step ONLY —
you are NOT analyzing anything.

Input file: {path}

Structural probe (evidence — do not assume beyond it):
{probe}

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
   "_reconstruction" key so downstream sees clean metadata.
5. RECONSTRUCTS the original file from those two outputs and writes it to
   "{recon_out}" (same format as the input). We will verify it matches the input.
6. Prints exactly one line:
   PREP_RESULT_JSON:{{"data_out": "<abs path>", "metadata_out": "{metadata_out}", "recon_out": "{recon_out}"}}

HARD CONSTRAINTS (a violation fails verification):
- DO NOT transform, scale, normalize, filter, fit, resample, denoise, crop, or
  analyze the data. The data you write MUST be the original values, unchanged.
- The reconstruction MUST reproduce the original file's data and metadata.
- Allowed libraries ONLY: numpy, pandas, json, csv, struct, io, re, h5py, os,
  pathlib. Do NOT import analysis libraries (sklearn, scipy.*, lmfit, torch,
  cv2, skimage, matplotlib, ...).
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


def _static_guard(script: str) -> str | None:
    """Return a rejection reason if the script imports an analysis library."""
    for lib in _ANALYSIS_IMPORT_DENYLIST:
        root = lib.split(".")[0]
        if re.search(rf"^\s*(import|from)\s+{re.escape(root)}\b", script, re.MULTILINE):
            return (
                f"Generated prep script imports a forbidden analysis library "
                f"('{root}'). File prep must use IO/parsing libraries only."
            )
    return None


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
            return np.allclose(np.load(original), np.load(recon), equal_nan=True)
        if ext == ".npz":
            a, b = np.load(original), np.load(recon)
            return set(a.files) == set(b.files) and all(
                np.allclose(a[k], b[k], equal_nan=True) for k in a.files
            )
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


def _verify(input_path: Path, result: dict) -> tuple[bool, str]:
    """Verify the split is well-formed AND lossless. Returns (ok, reason)."""
    data_out = Path(result.get("data_out", ""))
    meta_out = Path(result.get("metadata_out", ""))
    recon_out = Path(result.get("recon_out", ""))

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
    recon_path, attempts}``; otherwise ``{status:"error", message, attempts}``.
    The caller decides how to handle an error — the meta surfaces it to the user
    and asks how to proceed rather than silently delegating an unsplit file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^0-9A-Za-z_-]", "_", input_path.stem) or "input"
    metadata_out = output_dir / f"{stem}_metadata.json"
    recon_out = output_dir / f"{stem}_reconstructed{input_path.suffix}"

    probe_str = json.dumps(probe, indent=2, default=str) if probe else \
        f"(no probe) extension={input_path.suffix}, size={input_path.stat().st_size} bytes"
    base_prompt = _PROMPT.format(
        path=str(input_path), probe=probe_str, out_dir=str(output_dir),
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

        exec_res = executor.execute_script(script, working_dir=str(output_dir))
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

        ok, reason = _verify(input_path, result)
        if ok:
            if logger:
                logger.info(f"✅ File prep OK ({reason}): data={result['data_out']}, metadata={metadata_out}")
            return {
                "status": "success",
                "data_path": result["data_out"],
                "metadata_path": str(metadata_out),
                "recon_path": str(recon_out),
                "attempts": attempt,
            }
        last_reason = reason
        if logger:
            logger.warning(f"📦 File prep verification failed: {reason}")
        prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{reason}\nProduce a lossless split."

    return {
        "status": "error",
        "message": f"Could not produce a verified lossless split after "
                   f"{max_retries + 1} attempt(s): {last_reason}",
        "attempts": max_retries + 1,
    }
