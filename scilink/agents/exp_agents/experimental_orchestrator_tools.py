"""
Tool definitions and schemas for the ExperimentalAnalysisOrchestrator.
Supports both Google Gemini (function objects) and OpenAI (JSON schemas).

Follows the same pattern as planning_agents/orchestrator_tools.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from enum import Enum


class AgentType(Enum):
    """Available analysis agent types."""
    FFT_MICROSCOPY = "fft_microscopy"
    SAM_MICROSCOPY = "sam_microscopy"
    HYPERSPECTRAL = "hyperspectral"
    CURVE_FITTING = "curve_fitting"


class ExperimentalOrchestratorTools:
    """
    Manages tool definitions, schemas, and execution for the ExperimentalAnalysisOrchestrator.
    
    Follows the same pattern as OrchestratorTools from planning_agents.
    """
    
    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: Reference to the parent ExperimentalAnalysisOrchestrator
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build function map and schemas
        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []
        self.gemini_functions: list = []
        
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all tools with both OpenAI and Gemini formats."""
        
        # =====================================================================
        # 1. ANALYZE MICROSCOPY FFT
        # =====================================================================
        def analyze_microscopy_fft(data_path: str, metadata_json: str) -> str:
            """
            Analyze microscopy image using FFT/NMF for periodic structures.
            
            Args:
                data_path: Path to the microscopy image file
                metadata_json: JSON string with experiment metadata
            """
            return self._run_analysis(AgentType.FFT_MICROSCOPY, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_microscopy_fft,
            name="analyze_microscopy_fft",
            description=(
                "Analyze microscopy image using FFT/NMF for periodic structures, "
                "domains, phases, lattice fringes, and Moiré patterns. "
                "Best for crystalline materials, atomic-resolution images, oriented domains."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": "Path to the microscopy image file (.tif, .png, .jpg)"
                },
                "metadata_json": {
                    "type": "string",
                    "description": "JSON string with experiment metadata (technique, material, etc.)"
                }
            },
            required=["data_path", "metadata_json"]
        )
        
        # =====================================================================
        # 2. ANALYZE MICROSCOPY SAM
        # =====================================================================
        def analyze_microscopy_sam(data_path: str, metadata_json: str) -> str:
            """
            Analyze microscopy image using SAM for particle detection.
            
            Args:
                data_path: Path to the microscopy image file
                metadata_json: JSON string with experiment metadata
            """
            return self._run_analysis(AgentType.SAM_MICROSCOPY, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_microscopy_sam,
            name="analyze_microscopy_sam",
            description=(
                "Analyze microscopy image using Segment Anything Model (SAM) for "
                "particle detection, object counting, and morphological analysis. "
                "Best for discrete particles, pores, cells, droplets, voids."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": "Path to the microscopy image file (.tif, .png, .jpg)"
                },
                "metadata_json": {
                    "type": "string",
                    "description": "JSON string with experiment metadata (technique, material, etc.)"
                }
            },
            required=["data_path", "metadata_json"]
        )
        
        # =====================================================================
        # 3. ANALYZE HYPERSPECTRAL
        # =====================================================================
        def analyze_hyperspectral(data_path: str, metadata_json: str) -> str:
            """
            Analyze hyperspectral/spectroscopic data with NMF unmixing.
            
            Args:
                data_path: Path to the hyperspectral data file (.npy)
                metadata_json: JSON string with experiment metadata
            """
            return self._run_analysis(AgentType.HYPERSPECTRAL, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_hyperspectral,
            name="analyze_hyperspectral",
            description=(
                "Analyze hyperspectral/spectroscopic data (3D datacube) with NMF unmixing. "
                "Best for EELS, EDS, hyperspectral imaging where you need to extract "
                "component spectra and spatial abundance maps."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": "Path to the hyperspectral data file (.npy format, shape: [height, width, channels])"
                },
                "metadata_json": {
                    "type": "string",
                    "description": "JSON string with experiment metadata (technique, material, etc.)"
                }
            },
            required=["data_path", "metadata_json"]
        )
        
        # =====================================================================
        # 4. ANALYZE CURVE
        # =====================================================================
        def analyze_curve(data_path: str, metadata_json: str) -> str:
            """
            Analyze 1D curve data with peak fitting.
            
            Args:
                data_path: Path to the curve data file
                metadata_json: JSON string with experiment metadata
            """
            return self._run_analysis(AgentType.CURVE_FITTING, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_curve,
            name="analyze_curve",
            description=(
                "Analyze 1D curve data (spectra, diffractograms) with peak fitting. "
                "Best for Raman, XRD, PL, IR, UV-Vis, absorption spectra. "
                "Identifies peaks, fits profiles, extracts positions and intensities."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": "Path to the curve data file (.csv, .txt, .npy, .xlsx)"
                },
                "metadata_json": {
                    "type": "string",
                    "description": "JSON string with experiment metadata (technique, material, etc.)"
                }
            },
            required=["data_path", "metadata_json"]
        )
        
        # =====================================================================
        # 5. SELECT MICROSCOPY AGENT (Visual)
        # =====================================================================
        def select_microscopy_agent(image_path: str, analysis_goal: str = None) -> str:
            """
            Visually examine a microscopy image to choose between FFT and SAM analysis.
            
            Args:
                image_path: Path to the microscopy image
                analysis_goal: Optional analysis goal for context
            """
            print(f"  ⚡ Tool: Visual microscopy agent selection for {image_path}...")
            
            # Try to load and examine the image
            try:
                from ...tools.image_processor import load_image, preprocess_image, convert_numpy_to_jpeg_bytes
            except ImportError:
                try:
                    from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes
                except ImportError:
                    return json.dumps({
                        "status": "fallback",
                        "selected_agent": "fft_microscopy",
                        "reasoning": "Image processing not available, defaulting to FFT",
                        "recommendation": "Use analyze_microscopy_fft for this image"
                    })
            
            if not Path(image_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Image not found: {image_path}"
                })
            
            # For now, use heuristics based on filename/goal
            # In full implementation, this would use LLM vision
            selected = "fft_microscopy"
            reasoning = "Default selection - FFT analysis for structural features"
            
            if analysis_goal:
                goal_lower = analysis_goal.lower()
                if any(kw in goal_lower for kw in ['particle', 'count', 'size', 'distribution', 'pore', 'cell']):
                    selected = "sam_microscopy"
                    reasoning = f"SAM selected based on goal keywords: {analysis_goal}"
                elif any(kw in goal_lower for kw in ['periodic', 'lattice', 'fft', 'domain', 'phase', 'moiré']):
                    selected = "fft_microscopy"
                    reasoning = f"FFT selected based on goal keywords: {analysis_goal}"
            
            return json.dumps({
                "status": "success",
                "selected_agent": selected,
                "reasoning": reasoning,
                "recommendation": f"Use analyze_microscopy_{selected.replace('_microscopy', '')} for this image"
            })
        
        self._register_tool(
            func=select_microscopy_agent,
            name="select_microscopy_agent",
            description=(
                "Examine a microscopy image to choose between FFT analysis (for periodic structures) "
                "and SAM analysis (for particle detection). Use this when unsure which analysis is appropriate."
            ),
            parameters={
                "image_path": {
                    "type": "string",
                    "description": "Path to the microscopy image to examine"
                },
                "analysis_goal": {
                    "type": "string",
                    "description": "Optional: specific analysis goal (e.g., 'count particles', 'analyze lattice')"
                }
            },
            required=["image_path"]
        )
        
        # =====================================================================
        # 6. READ FILE
        # =====================================================================
        def read_file(file_path: str) -> str:
            """
            Read contents of a file (metadata, config, etc.).
            
            Args:
                file_path: Path to the file to read
            """
            print(f"  ⚡ Tool: Reading file {file_path}...")
            
            try:
                path = Path(file_path)
                if not path.exists():
                    return json.dumps({
                        "status": "error",
                        "message": f"File not found: {file_path}"
                    })
                
                # Check file size (limit to 100KB)
                if path.stat().st_size > 100 * 1024:
                    return json.dumps({
                        "status": "error",
                        "message": "File too large (>100KB)"
                    })
                
                ext = path.suffix.lower()
                
                if ext == '.json':
                    with open(path, 'r') as f:
                        content = json.load(f)
                    
                    # Store as current metadata if it looks like metadata
                    if isinstance(content, dict):
                        self.orch.current_metadata = content
                    
                    return json.dumps({
                        "status": "success",
                        "file_type": "json",
                        "content": content
                    })
                    
                elif ext in ['.txt', '.md', '.csv', '.yaml', '.yml']:
                    with open(path, 'r') as f:
                        content = f.read()
                    return json.dumps({
                        "status": "success",
                        "file_type": "text",
                        "content": content[:5000]  # Limit text length
                    })
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"Unsupported file type: {ext}",
                        "supported": [".json", ".txt", ".md", ".csv", ".yaml", ".yml"]
                    })
                    
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=read_file,
            name="read_file",
            description=(
                "Read contents of a file (JSON metadata, text descriptions, configs). "
                "Use this to load metadata files before running analysis."
            ),
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            required=["file_path"]
        )
        
        # =====================================================================
        # 7. LIST DIRECTORY
        # =====================================================================
        def list_directory(directory_path: str) -> str:
            """
            List files in a directory.
            
            Args:
                directory_path: Path to the directory to list
            """
            print(f"  ⚡ Tool: Listing directory {directory_path}...")
            
            try:
                path = Path(directory_path)
                if not path.exists():
                    return json.dumps({
                        "status": "error",
                        "message": f"Directory not found: {directory_path}"
                    })
                if not path.is_dir():
                    return json.dumps({
                        "status": "error",
                        "message": f"Not a directory: {directory_path}"
                    })
                
                files = []
                for f in sorted(path.iterdir()):
                    if f.name.startswith('.'):
                        continue  # Skip hidden files
                    
                    size = f.stat().st_size if f.is_file() else 0
                    files.append({
                        "name": f.name,
                        "type": "directory" if f.is_dir() else "file",
                        "size": size,
                        "extension": f.suffix if f.is_file() else None
                    })
                
                return json.dumps({
                    "status": "success",
                    "directory": str(path),
                    "files": files,
                    "count": len(files)
                })
                
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=list_directory,
            name="list_directory",
            description="List files and subdirectories in a directory.",
            parameters={
                "directory_path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                }
            },
            required=["directory_path"]
        )
        
        # =====================================================================
        # 8. CONVERT METADATA
        # =====================================================================
        def convert_metadata(description: str) -> str:
            """
            Convert natural language experiment description to structured metadata.
            
            Args:
                description: Natural language description of the experiment
            """
            print(f"  ⚡ Tool: Converting metadata from description...")
            
            try:
                from .metadata_converter import generate_metadata_json_from_text
                
                # Save description to temp file
                temp_path = self.orch.output_dir / "temp_metadata_input.txt"
                with open(temp_path, 'w') as f:
                    f.write(description)
                
                result = generate_metadata_json_from_text(
                    input_text_filepath=str(temp_path),
                    api_key=self.orch.api_key,
                    model_name=self.orch.model_name,
                    base_url=self.orch.base_url
                )
                
                temp_path.unlink(missing_ok=True)
                
                if result:
                    self.orch.current_metadata = result
                    return json.dumps({
                        "status": "success",
                        "metadata": result,
                        "hint": "Metadata stored. You can now call analysis tools with this metadata."
                    })
                else:
                    # Fallback: create basic metadata structure
                    basic_metadata = self._parse_basic_metadata(description)
                    self.orch.current_metadata = basic_metadata
                    return json.dumps({
                        "status": "success",
                        "metadata": basic_metadata,
                        "note": "Created basic metadata structure from description"
                    })
                    
            except Exception as e:
                self.logger.error(f"Metadata conversion error: {e}")
                # Fallback
                basic_metadata = self._parse_basic_metadata(description)
                self.orch.current_metadata = basic_metadata
                return json.dumps({
                    "status": "success",
                    "metadata": basic_metadata,
                    "note": f"Used fallback parsing due to: {str(e)[:100]}"
                })
        
        self._register_tool(
            func=convert_metadata,
            name="convert_metadata",
            description=(
                "Convert a natural language experiment description to structured metadata JSON. "
                "Use when user describes their experiment in plain text instead of providing a metadata file."
            ),
            parameters={
                "description": {
                    "type": "string",
                    "description": "Natural language description of the experiment (technique, material, conditions)"
                }
            },
            required=["description"]
        )
        
        # =====================================================================
        # 9. LIST AVAILABLE AGENTS
        # =====================================================================
        def list_available_agents() -> str:
            """List all available analysis agents and their capabilities."""
            print(f"  ⚡ Tool: Listing available agents...")
            
            from .experimental_orchestrator import AGENT_REGISTRY, AgentType
            
            agents = {}
            for agent_type in AgentType:
                info = AGENT_REGISTRY[agent_type]
                agents[agent_type.value] = {
                    "description": info["description"],
                    "data_types": info["data_types"],
                    "file_extensions": info["file_extensions"]
                }
            
            return json.dumps({
                "status": "success",
                "agents": agents
            })
        
        self._register_tool(
            func=list_available_agents,
            name="list_available_agents",
            description="List all available analysis agents and their capabilities.",
            parameters={},
            required=[]
        )
    
    def _run_analysis(self, agent_type: AgentType, data_path: str, metadata_json: str) -> str:
        """
        Internal method to run analysis with a specific agent.
        
        Args:
            agent_type: Which agent to use
            data_path: Path to data file
            metadata_json: JSON string with metadata
        """
        print(f"  ⚡ Tool: Running {agent_type.value} analysis on {data_path}...")
        
        # Parse metadata
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "status": "error",
                "message": f"Invalid metadata JSON: {e}"
            })
        
        # Validate file exists
        if not Path(data_path).exists():
            return json.dumps({
                "status": "error",
                "message": f"File not found: {data_path}"
            })
        
        try:
            # Get or create agent
            agent = self.orch.get_or_create_agent(agent_type)
            
            # Run analysis
            result = agent.analyze(data=data_path, system_info=metadata)
            
            # Store result
            analysis_record = {
                "agent": agent_type.value,
                "data_path": data_path,
                "timestamp": datetime.now().isoformat(),
                "status": result.get("status", "unknown")
            }
            self.orch.last_analysis = result
            self.orch.analyses_run.append(analysis_record)
            
            # Build response summary
            response = {
                "status": result.get("status", "unknown"),
                "agent": agent_type.value,
                "data_path": data_path,
                "output_directory": result.get("output_directory", str(self.orch.output_dir / agent_type.value))
            }
            
            # Add key results
            if "detailed_analysis" in result:
                # Truncate for response
                analysis_text = result["detailed_analysis"]
                response["analysis_summary"] = analysis_text[:1500] + "..." if len(analysis_text) > 1500 else analysis_text
            
            if "scientific_claims" in result:
                claims = result["scientific_claims"]
                response["claims_count"] = len(claims)
                response["top_claims"] = claims[:3] if claims else []
            
            return json.dumps(response)
            
        except Exception as e:
            self.logger.error(f"Analysis error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "agent": agent_type.value
            })
    
    def _parse_basic_metadata(self, description: str) -> Dict[str, Any]:
        """
        Parse basic metadata from description using simple heuristics.
        Fallback when LLM conversion fails.
        """
        desc_lower = description.lower()
        
        # Detect technique
        technique = "Unknown"
        technique_keywords = {
            "stem": "STEM", "tem": "TEM", "sem": "SEM", "afm": "AFM",
            "raman": "Raman", "xrd": "XRD", "pl": "PL", "photoluminescence": "PL",
            "ir": "IR", "infrared": "IR", "ftir": "FTIR",
            "eels": "EELS", "eds": "EDS", "edx": "EDX",
            "uv-vis": "UV-Vis", "absorption": "Absorption"
        }
        for kw, tech in technique_keywords.items():
            if kw in desc_lower:
                technique = tech
                break
        
        # Detect experiment type
        exp_type = "Unknown"
        if technique in ["STEM", "TEM", "SEM", "AFM"]:
            exp_type = "Microscopy"
        elif technique in ["Raman", "XRD", "PL", "IR", "FTIR", "UV-Vis", "Absorption"]:
            exp_type = "Spectroscopy"
        elif technique in ["EELS", "EDS", "EDX"]:
            exp_type = "Spectroscopy"
        
        return {
            "experiment_type": exp_type,
            "experiment": {
                "technique": technique
            },
            "sample": {
                "material": "Unknown",
                "description": description
            }
        }
    
    def _register_tool(self, func: Callable, name: str, description: str,
                       parameters: Dict[str, Any], required: list = None):
        """
        Register a tool in both OpenAI and Gemini formats.
        
        Args:
            func: The Python function to call
            name: Function name
            description: What the function does
            parameters: Dict of parameter definitions
            required: List of required parameter names
        """
        # Add to function map for execution
        self.functions_map[name] = func
        
        # Add to Gemini format (just the function object)
        self.gemini_functions.append(func)
        
        # Build OpenAI schema
        openai_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or []
                }
            }
        }
        self.openai_schemas.append(openai_schema)
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            JSON string with result
        """
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found in registry"
            })
        
        try:
            return self.functions_map[tool_name](**kwargs)
        except Exception as e:
            self.logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name
            })
