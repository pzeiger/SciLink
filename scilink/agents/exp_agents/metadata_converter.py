import json
import logging
import os

from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params
from ...auth import get_internal_proxy_key


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the JSON Schema as a Dictionary
METADATA_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        # --- Core Identification ---
        "experiment_type": {
            "type": "string",
            "description": "General type of experiment (e.g., Microscopy, Spectroscopy, Diffraction, Curve Analysis)."
        },
        # --- Detailed Experiment Info ---
        "experiment": {
            "type": "object",
            "description": "Details about the specific experimental setup.",
            "properties": {
                "technique": {
                    "type": "string",
                    "description": "Specific experimental technique used (e.g., HAADF-STEM, TEM, Photoluminescence Spectroscopy, XRD)."
                },
                "date": {
                    "type": ["string", "null"], # Optional
                    "description": "Date of the experiment (if available, YYYY-MM-DD)."
                },
                "instrument": {
                    "type": ["string", "null"], # Optional
                    "description": "Instrument used for the experiment."
                },
                "details": {
                    "type": ["string", "null"], # Optional
                    "description": "Other relevant conditions or parameters (e.g., voltage, temperature, excitation source, probe current)."
                }
            },
            "required": ["technique"] # Technique is crucial
        },
        # --- Sample Info ---
        "sample": {
            "type": "object",
            "description": "Details about the sample.",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "Specific material name, composition, or formula (e.g., MoS2, Gallium Nitride (GaN))."
                },
                "description": {
                    "type": ["string", "null"], # Optional
                    "description": "Additional description (e.g., form, substrate, synthesis method)."
                }
            },
            "required": ["material"] # Specific material is crucial
        },
        # --- Microscopy Specific (Conditional) ---
        "spatial_info": {
            "type": ["object", "null"], # Optional object
            "description": "Spatial dimensions/scale (Primarily for Microscopy).",
            "properties": {
                "field_of_view_x": {"type": ["number", "null"]},
                "field_of_view_y": {"type": ["number", "null"]},
                "field_of_view_units": {"type": ["string", "null"], "description": "Units like 'nm', 'um', 'pixels'."}
            },
        },
        # --- Spectroscopy/Hyperspectral Specific (Conditional) ---
        "energy_range": {
            "type": ["object", "null"], # Optional object
            "description": "Spectral or energy range covered (Primarily for Spectroscopy/Hyperspectral).",
            "properties": {
                "start": {"type": ["number", "null"]},
                "end": {"type": ["number", "null"]},
                "units": {"type": ["string", "null"], "description": "Units like 'nm', 'eV', 'cm^-1'."}
            },
        },
        # --- 1D Curve Specific (Conditional) ---
        "title": {
            "type": ["string", "null"], # Optional
            "description": "A descriptive title for the data/plot (Primarily for 1D Curves)."
        },
        "data_columns": {
            "type": ["array", "null"], # Optional array
            "description": "Description of data columns, typically X and Y (Primarily for 1D Curves).",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name like 'Wavelength', 'Angle', 'Energy', 'Intensity', 'Counts'."},
                    "units": {"type": "string"}
                },
                "required": ["name", "units"]
            }
        },
        "xlabel": {
            "type": ["string", "null"], # Optional
            "description": "Suggested X-axis label, including units (Primarily for 1D Curves)."
        },
        "ylabel": {
            "type": ["string", "null"], # Optional
            "description": "Suggested Y-axis label, including units (Primarily for 1D Curves)."
        }
    },
    # Universally required fields at the top level
    "required": ["experiment_type", "experiment", "sample"]
}

# Convert schema dictionary to a JSON string for embedding in the prompt
schema_json_string_for_prompt = json.dumps(METADATA_SCHEMA_DICT, indent=2)

METADATA_GENERATION_PROMPT = f"""
You are an expert scientific assistant. Your task is to read the provided plain text description of a scientific experiment (which could be microscopy, spectroscopy/hyperspectral, or 1D curve data like PL/XRD) and extract key metadata into a structured JSON format.

Format your output STRICTLY as a valid JSON object conforming *exactly* to the following structure:

```json
{schema_json_string_for_prompt}
```

Instructions:

Identify Experiment Type: First, determine the general experiment_type (e.g., Microscopy, Spectroscopy, Diffraction, Curve Analysis) and the specific experiment.technique (e.g., HAADF-STEM, TEM, PL, XRD, Absorption). Also identify sample.material.

Fill Core Fields: Populate the required fields: experiment_type, experiment (including technique), and sample (including material). Extract other details like instrument or conditions into experiment.details or sample.description if available.

Fill Conditional Fields based on Type:

If Microscopy: Focus on extracting spatial_info (field of view and units). Omit or use null for spectroscopy/curve fields if not relevant.

If Spectroscopy/Hyperspectral: Focus on extracting energy_range (start, end, units). Omit or use null for microscopy/curve fields if not relevant.

If 1D Curve Data (PL, XRD, etc.): Focus on extracting title, data_columns (determining X and Y column names/units), xlabel, and ylabel. energy_range might also apply if the x-axis represents energy. Omit or use null for microscopy fields.

Handle Missing Info: Use null for any optional fields (like experiment.date) or omit entire optional objects (spatial_info, energy_range, data_columns) if the information is missing or clearly not applicable to the described experiment type.

Required Fields: For universally required fields (experiment_type, experiment.technique, sample.material) that are truly missing even after careful reading, use the string "N/A".

Strict Formatting: Only include fields defined in the schema. Output ONLY the valid JSON object without markdown formatting.

Ensure the output JSON accurately reflects the information present in the text description and adheres to the conditional logic based on the experiment type.
"""


def generate_metadata_json_from_text(
    input_text_filepath: str,           
    api_key: str | None = None,
    model_name: str = "gemini-3-flash-preview",
    base_url: str | None = None,
    # Deprecated
    google_api_key: str | None = None,
    local_model: str | None = None
) -> dict | None:
    """
    Reads a plain text experimental description and converts it to metadata JSON.
    """
    
    # 1. Normalize Parameters
    api_key, base_url = normalize_params(
        api_key=api_key,
        google_api_key=google_api_key,
        base_url=base_url,
        local_model=local_model,
        source="generate_metadata_json_from_text"
    )

    # 2. Input Validation
    if not input_text_filepath or not os.path.exists(input_text_filepath):
        logger.error(f"Input text file not found: {input_text_filepath}")
        return None
    try:
        with open(input_text_filepath, 'r', encoding='utf-8') as f:
            text_description = f.read()
        if not text_description.strip(): return None
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return None

    base_name = os.path.splitext(input_text_filepath)[0]
    output_json_filepath = f"{base_name}.json"

    # 3. Model Initialization
    model = None
    
    if base_url:
        if 'gguf' in base_url:
             # Support GGUF here for completeness
             from scilink.wrappers.llama_wrapper import LocalLlamaModel
             model = LocalLlamaModel(base_url)
        else:
            if api_key is None:
                api_key = get_internal_proxy_key()
            if not api_key:
                # Fallback or error based on preference, here we log warning
                logger.warning("No API key found for proxy, attempting connection anyway...")

            logger.info(f"🏛️ Using OpenAI-compatible agent: {base_url}")
            model = OpenAIAsGenerativeModel(model=model_name, api_key=api_key, base_url=base_url)
    else:
        logger.info(f"☁️ Using LiteLLM agent: {model_name}")
        model = LiteLLMGenerativeModel(model=model_name, api_key=api_key)

    # 4. Prepare Prompt
    prompt_parts = [
        METADATA_GENERATION_PROMPT,
        "\n--- Plain Text Description ---",
        text_description, 
        "\n--- Extracted JSON Metadata ---"
    ]

    # 5. Generate
    try:
        response = model.generate_content(contents=prompt_parts)
        
        # 6. Parse
        if not hasattr(response, 'text'):
            if hasattr(response, 'candidates'): raw_text = response.candidates[0].content.parts[0].text
            else: raw_text = str(response)
        else:
             raw_text = response.text

        # Clean markdown
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]
            
        metadata_dict = json.loads(raw_text.strip())
        
        # Save
        with open(output_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=4)
        logger.info(f"Saved metadata to: {output_json_filepath}")
        
        return metadata_dict

    except Exception as e:
        logger.error(f"Error generating metadata: {e}", exc_info=True)
        return None