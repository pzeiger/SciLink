"""
Tool definitions and schemas for the AnalysisOrchestratorAgent.
Supports both OpenAI (JSON schemas) and LiteLLM formats.
"""

import json
import logging
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List

from .metadata_converter import generate_metadata_json_from_text, METADATA_SCHEMA_DICT
from .orchestrator_agent import OrchestratorAgent, AGENT_MAP


class AnalysisOrchestratorTools:
    """
    Manages tool definitions, schemas, and execution for the AnalysisOrchestratorAgent.
    """
    
    # Agent name mapping for display
    AGENT_NAMES = {
        0: "FFTMicroscopyAnalysisAgent",
        1: "SAMMicroscopyAnalysisAgent", 
        2: "HyperspectralAnalysisAgent",
        3: "CurveFittingAgent"
    }
    
    AGENT_DESCRIPTIONS = {
        0: "Microstructure analysis via FFT/NMF - for grains, phases, domains, and atomic-resolution images",
        1: "Particle/object segmentation - for counting, size distributions",
        2: "Spectroscopic/hyperspectral data analysis",
        3: "1D curve/spectrum fitting - for peaks, band gaps, kinetics"
    }
    
    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: Reference to the parent AnalysisOrchestratorAgent
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build function map and schemas
        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []
        
        self._register_all_tools()

    def _get_human_feedback_enabled(self) -> bool:
        """Get current human feedback setting from orchestrator."""
        return getattr(self.orch, '_enable_human_feedback', True)

    def _register_all_tools(self):
        """Register all tools with OpenAI format."""
        
        # =====================================================================
        # 1. EXAMINE DATA
        # =====================================================================
        def examine_data(data_path: str) -> str:
            """
            Examine a data file to determine its type and characteristics.
            """
            print(f"  ⚡ Tool: Examining data at {data_path}...")
            
            path = Path(data_path)
            if not path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"File not found: {data_path}"
                })
            
            # Get file info
            file_size = path.stat().st_size
            extension = path.suffix.lower()
            
            result = {
                "status": "success",
                "file_path": str(path.absolute()),
                "file_name": path.name,
                "file_size_bytes": file_size,
                "extension": extension
            }
            
            try:
                # Determine data type based on extension and content
                if extension in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']:
                    result["data_type"] = "microscopy"
                    result["suggested_agents"] = [0, 1]  # FFT or SAM
                    
                    # Try to load and get shape
                    try:
                        from ..tools.image_processor import load_image
                        img = load_image(str(path))
                        result["shape"] = list(img.shape)
                        result["dtype"] = str(img.dtype)
                        
                        # Suggest based on image characteristics
                        if len(img.shape) == 2:
                            h, w = img.shape
                        else:
                            h, w = img.shape[:2]
                        
                        result["image_size"] = f"{w}x{h}"
                        result["primary_suggestion"] = 0  # FFT (handles all microscopy including atomic)
                            
                    except Exception as e:
                        result["load_error"] = str(e)
                
                elif extension == '.npy':
                    # Could be spectrum, hyperspectral, or curve data
                    data = np.load(str(path))
                    result["shape"] = list(data.shape)
                    result["dtype"] = str(data.dtype)
                    
                    if data.ndim == 1:
                        result["data_type"] = "curve"
                        result["suggested_agents"] = [3]  # CurveFitting
                        result["primary_suggestion"] = 3
                        result["n_points"] = data.shape[0]
                        
                    elif data.ndim == 2:
                        if data.shape[1] == 2 or data.shape[0] == 2:
                            result["data_type"] = "curve"
                            result["suggested_agents"] = [3]
                            result["primary_suggestion"] = 3
                            result["n_points"] = max(data.shape)
                        else:
                            # Could be image or spectrum
                            result["data_type"] = "microscopy_or_spectrum"
                            result["suggested_agents"] = [0, 2]  # FFT or Hyperspectral
                            result["primary_suggestion"] = 0
                            
                    elif data.ndim == 3:
                        result["data_type"] = "hyperspectral"
                        result["suggested_agents"] = [2]  # Hyperspectral
                        result["primary_suggestion"] = 2
                        result["spatial_shape"] = list(data.shape[:2])
                        result["spectral_channels"] = data.shape[2]
                
                elif extension in ['.csv', '.txt', '.tsv']:
                    result["data_type"] = "tabular"
                    result["suggested_agents"] = [3]  # CurveFitting
                    result["primary_suggestion"] = 3
                    
                    # Try to peek at the file
                    try:
                        with open(path, 'r') as f:
                            first_lines = [f.readline() for _ in range(5)]
                        result["preview"] = first_lines
                    except Exception as e:
                        result["preview_error"] = str(e)
                
                else:
                    result["data_type"] = "unknown"
                    result["message"] = f"Unknown file extension: {extension}"
                    result["suggested_agents"] = []
                
                # Store in orchestrator state
                self.orch.current_data_path = str(path.absolute())
                self.orch.current_data_type = result.get("data_type")
                
                return json.dumps(result)
                
            except Exception as e:
                self.logger.error(f"Error examining data: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=examine_data,
            name="examine_data",
            description=(
                "Examine a data file to determine its type and characteristics. "
                "Returns data type, shape, and suggested analysis agents."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": "Path to the data file to examine"
                }
            },
            required=["data_path"]
        )
        
        # =====================================================================
        # 2. CONVERT METADATA
        # =====================================================================
        def convert_metadata(
            text_input: str = None,
            text_file_path: str = None
        ) -> str:
            """
            Convert natural language description to structured metadata JSON.
            """
            print(f"  ⚡ Tool: Converting metadata...")
            
            if text_file_path:
                path = Path(text_file_path)
                if not path.exists():
                    return json.dumps({
                        "status": "error",
                        "message": f"File not found: {text_file_path}"
                    })
                
                # Use the metadata converter
                try:
                    metadata = generate_metadata_json_from_text(
                        input_text_filepath=str(path),
                        api_key=self.orch.api_key,
                        model_name=self.orch.model_name,
                        base_url=self.orch.base_url
                    )
                    
                    if metadata:
                        self.orch.current_metadata = metadata
                        output_path = self.orch.base_dir / "metadata.json"
                        with open(output_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        return json.dumps({
                            "status": "success",
                            "metadata": metadata,
                            "saved_to": str(output_path)
                        })
                    else:
                        return json.dumps({
                            "status": "error",
                            "message": "Failed to convert metadata"
                        })
                        
                except Exception as e:
                    self.logger.error(f"Metadata conversion error: {e}", exc_info=True)
                    return json.dumps({
                        "status": "error",
                        "message": str(e)
                    })
            
            elif text_input:
                # Create temporary file and convert
                temp_path = self.orch.base_dir / "temp_metadata_input.txt"
                with open(temp_path, 'w') as f:
                    f.write(text_input)
                
                try:
                    metadata = generate_metadata_json_from_text(
                        input_text_filepath=str(temp_path),
                        api_key=self.orch.api_key,
                        model_name=self.orch.model_name,
                        base_url=self.orch.base_url
                    )
                    
                    # Clean up temp file
                    temp_path.unlink()
                    
                    if metadata:
                        self.orch.current_metadata = metadata
                        output_path = self.orch.base_dir / "metadata.json"
                        with open(output_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        return json.dumps({
                            "status": "success",
                            "metadata": metadata,
                            "saved_to": str(output_path)
                        })
                    else:
                        return json.dumps({
                            "status": "error",
                            "message": "Failed to convert metadata"
                        })
                        
                except Exception as e:
                    if temp_path.exists():
                        temp_path.unlink()
                    return json.dumps({
                        "status": "error",
                        "message": str(e)
                    })
            
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Must provide either text_input or text_file_path"
                })
        
        self._register_tool(
            func=convert_metadata,
            name="convert_metadata",
            description=(
                "Convert natural language description to structured metadata JSON. "
                "Accepts either direct text input or a path to a text file. "
                "Use this when user provides experimental description in plain text."
            ),
            parameters={
                "text_input": {
                    "type": "string",
                    "description": "Direct text description of the experiment (alternative to file)"
                },
                "text_file_path": {
                    "type": "string",
                    "description": "Path to a .txt file containing experiment description"
                }
            },
            required=[]
        )
        
        # =====================================================================
        # 3. LOAD METADATA
        # =====================================================================
        def load_metadata(json_path: str) -> str:
            """
            Load existing JSON metadata file.
            """
            print(f"  ⚡ Tool: Loading metadata from {json_path}...")
            
            path = Path(json_path)
            if not path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"File not found: {json_path}"
                })
            
            try:
                with open(path, 'r') as f:
                    metadata = json.load(f)
                
                # Validate basic structure
                required_fields = ["experiment_type", "experiment", "sample"]
                missing = [f for f in required_fields if f not in metadata]
                
                if missing:
                    return json.dumps({
                        "status": "warning",
                        "message": f"Metadata loaded but missing recommended fields: {missing}",
                        "metadata": metadata
                    })
                
                self.orch.current_metadata = metadata
                
                return json.dumps({
                    "status": "success",
                    "metadata": metadata,
                    "experiment_type": metadata.get("experiment_type"),
                    "technique": metadata.get("experiment", {}).get("technique"),
                    "material": metadata.get("sample", {}).get("material")
                })
                
            except json.JSONDecodeError as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid JSON: {e}"
                })
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=load_metadata,
            name="load_metadata",
            description=(
                "Load existing JSON metadata file. "
                "Use this when user provides a .json metadata file."
            ),
            parameters={
                "json_path": {
                    "type": "string",
                    "description": "Path to the JSON metadata file"
                }
            },
            required=["json_path"]
        )
        
        # =====================================================================
        # 4. SELECT AGENT
        # =====================================================================
        def select_agent(
            data_type: str = None,
            analysis_goal: str = None,
            image_path: str = None
        ) -> str:
            """
            Use LLM to select the most appropriate analysis agent.
            """
            print(f"  ⚡ Tool: Selecting analysis agent...")
            
            # Use current state if not provided
            if data_type is None:
                data_type = self.orch.current_data_type or "unknown"
            
            if image_path is None:
                image_path = self.orch.current_data_path
            
            # Build system info for orchestrator
            system_info = {}
            if self.orch.current_metadata:
                system_info = self.orch.current_metadata.copy()
            if analysis_goal:
                system_info["analysis_goal"] = analysis_goal
            
            # Use the existing OrchestratorAgent for selection
            try:
                selector = OrchestratorAgent(
                    api_key=self.orch.api_key,
                    model_name=self.orch.model_name,
                    base_url=self.orch.base_url
                )
                
                agent_id, reasoning = selector.select_agent(
                    data_type=data_type,
                    system_info=system_info,
                    image_path=image_path
                )
                
                if agent_id >= 0:
                    self.orch.selected_agent_id = agent_id
                    
                    return json.dumps({
                        "status": "success",
                        "agent_id": agent_id,
                        "agent_name": self.AGENT_NAMES.get(agent_id, "Unknown"),
                        "description": self.AGENT_DESCRIPTIONS.get(agent_id, ""),
                        "reasoning": reasoning
                    })
                else:
                    return json.dumps({
                        "status": "error",
                        "message": reasoning or "Failed to select agent"
                    })
                    
            except Exception as e:
                self.logger.error(f"Agent selection error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=select_agent,
            name="select_agent",
            description=(
                "Use LLM to select the most appropriate analysis agent based on data type, "
                "metadata, and analysis goal. Returns agent ID and reasoning."
            ),
            parameters={
                "data_type": {
                    "type": "string",
                    "description": "Type of data (microscopy, spectroscopy, curve, hyperspectral)"
                },
                "analysis_goal": {
                    "type": "string",
                    "description": "User's analysis objective (e.g., 'count particles', 'find defects')"
                },
                "image_path": {
                    "type": "string",
                    "description": "Path to image for visual context (optional)"
                }
            },
            required=[]
        )
        
        # =====================================================================
        # 5. RUN ANALYSIS
        # =====================================================================
        def run_analysis(
            data_path: str = None,
            agent_id: int = None,
            analysis_goal: str = None
        ) -> str:
            """
            Execute analysis with the selected or specified agent.
            """
            print(f"  ⚡ Tool: Running analysis...")
            
            # Use current state if not provided
            if data_path is None:
                data_path = self.orch.current_data_path
            
            if agent_id is None:
                agent_id = self.orch.selected_agent_id
            
            # Validate inputs
            if data_path is None:
                return json.dumps({
                    "status": "error",
                    "message": "No data path provided. Use examine_data first."
                })
            
            if agent_id is None:
                return json.dumps({
                    "status": "error",
                    "message": "No agent selected. Use select_agent first."
                })
            
            if self.orch.current_metadata is None:
                return json.dumps({
                    "status": "error",
                    "message": "No metadata available. Use load_metadata or convert_metadata first."
                })
            
            try:
                # Get or create the agent
                agent = self.orch.get_agent(agent_id)
                
                print(f"    Using agent: {type(agent).__name__}")
                print(f"    Data: {data_path}")
                
                # Run analysis
                result = agent.analyze(
                    data=data_path,
                    system_info=self.orch.current_metadata
                )
                
                # Store result
                analysis_record = {
                    "timestamp": datetime.now().isoformat(),
                    "data_path": data_path,
                    "agent_id": agent_id,
                    "agent_name": self.AGENT_NAMES.get(agent_id),
                    "status": result.get("status"),
                    "output_directory": result.get("output_directory")
                }
                self.orch.analysis_results.append(analysis_record)
                
                # Format response
                if result.get("status") == "success":
                    return json.dumps({
                        "status": "success",
                        "agent_used": self.AGENT_NAMES.get(agent_id),
                        "detailed_analysis": result.get("detailed_analysis", "")[:2000],  # Truncate for chat
                        "claims_count": len(result.get("scientific_claims", [])),
                        "output_directory": result.get("output_directory"),
                        "full_result_available": True
                    })
                else:
                    return json.dumps({
                        "status": "error",
                        "error": result.get("error", {}),
                        "agent_used": self.AGENT_NAMES.get(agent_id)
                    })
                    
            except Exception as e:
                self.logger.error(f"Analysis error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=run_analysis,
            name="run_analysis",
            description=(
                "Execute analysis with the selected or specified agent. "
                "Requires data path and metadata to be set. "
                "Returns detailed analysis and scientific claims."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": "Path to data file (uses current if not specified)"
                },
                "agent_id": {
                    "type": "integer",
                    "description": "Agent ID to use (0-4, uses selected if not specified)"
                },
                "analysis_goal": {
                    "type": "string",
                    "description": "Specific analysis objective"
                }
            },
            required=[]
        )
        
        # =====================================================================
        # 6. LIST RESULTS
        # =====================================================================
        def list_results() -> str:
            """
            List analysis results in the session directory.
            """
            print(f"  ⚡ Tool: Listing results...")
            
            results = []
            
            # List files in results directory
            results_dir = self.orch.results_dir
            if results_dir.exists():
                for item in results_dir.iterdir():
                    if item.is_dir():
                        # Agent output directory
                        agent_results = {
                            "name": item.name,
                            "type": "agent_output",
                            "files": []
                        }
                        for f in item.rglob("*"):
                            if f.is_file():
                                agent_results["files"].append(str(f.relative_to(results_dir)))
                        results.append(agent_results)
                    elif item.is_file():
                        results.append({
                            "name": item.name,
                            "type": "file",
                            "size": item.stat().st_size
                        })
            
            return json.dumps({
                "status": "success",
                "session_directory": str(self.orch.base_dir),
                "results_directory": str(results_dir),
                "analysis_history": self.orch.analysis_results,
                "files": results
            })
        
        self._register_tool(
            func=list_results,
            name="list_results",
            description="List analysis results and files in the session directory.",
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 7. SAVE CHECKPOINT
        # =====================================================================
        def save_checkpoint() -> str:
            """
            Save session state for later resumption.
            """
            print(f"  ⚡ Tool: Saving checkpoint...")
            
            try:
                checkpoint_data = {
                    "timestamp": datetime.now().isoformat(),
                    "current_metadata": self.orch.current_metadata,
                    "current_data_path": self.orch.current_data_path,
                    "current_data_type": self.orch.current_data_type,
                    "selected_agent_id": self.orch.selected_agent_id,
                    "analysis_results": self.orch.analysis_results,
                    "message_count": self.orch.message_count,
                    "analysis_mode": self.orch.analysis_mode.value,
                }
                
                with open(self.orch.checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                return json.dumps({
                    "status": "success",
                    "checkpoint_path": str(self.orch.checkpoint_path),
                    "timestamp": checkpoint_data["timestamp"],
                    "analyses_saved": len(self.orch.analysis_results)
                })
                
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=save_checkpoint,
            name="save_checkpoint",
            description=(
                "Save complete session state including metadata, analysis history, "
                "and conversation. Use this to enable session resumption."
            ),
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 8. SHOW AVAILABLE AGENTS
        # =====================================================================
        def show_available_agents() -> str:
            """
            Show list of available analysis agents and their capabilities.
            """
            print(f"  ⚡ Tool: Showing available agents...")
            
            agents = []
            for agent_id in sorted(self.AGENT_NAMES.keys()):
                agents.append({
                    "id": agent_id,
                    "name": self.AGENT_NAMES[agent_id],
                    "description": self.AGENT_DESCRIPTIONS[agent_id]
                })
            
            return json.dumps({
                "status": "success",
                "agents": agents,
                "current_selection": self.orch.selected_agent_id
            })
        
        self._register_tool(
            func=show_available_agents,
            name="show_available_agents",
            description="Show list of available analysis agents and their capabilities.",
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 9. GET METADATA SCHEMA
        # =====================================================================
        def get_metadata_schema() -> str:
            """
            Get the metadata JSON schema for reference.
            """
            print(f"  ⚡ Tool: Getting metadata schema...")
            
            return json.dumps({
                "status": "success",
                "schema": METADATA_SCHEMA_DICT,
                "required_fields": ["experiment_type", "experiment", "sample"],
                "hint": "Use convert_metadata to create metadata from natural language"
            })
        
        self._register_tool(
            func=get_metadata_schema,
            name="get_metadata_schema",
            description=(
                "Get the metadata JSON schema showing required and optional fields. "
                "Use this to understand what metadata is needed."
            ),
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 10. GET MEASUREMENT RECOMMENDATIONS
        # =====================================================================
        def get_recommendations(analysis_index: int = -1) -> str:
            """
            Get measurement recommendations from a completed analysis.
            """
            print(f"  ⚡ Tool: Getting measurement recommendations...")
            
            if not self.orch.analysis_results:
                return json.dumps({
                    "status": "error",
                    "message": "No analyses completed yet. Run an analysis first."
                })
            
            try:
                # Get the analysis record
                if analysis_index == -1:
                    record = self.orch.analysis_results[-1]
                else:
                    record = self.orch.analysis_results[analysis_index]
                
                agent_id = record.get("agent_id")
                if agent_id is None:
                    return json.dumps({
                        "status": "error",
                        "message": "Analysis record missing agent_id"
                    })
                
                # Get the agent
                agent = self.orch.get_agent(agent_id)
                
                # Call recommend_measurements
                result = agent.recommend_measurements(
                    data=record.get("data_path"),
                    system_info=self.orch.current_metadata
                )
                
                return json.dumps({
                    "status": result.get("status", "success"),
                    "recommendations": result.get("measurement_recommendations", []),
                    "analysis_integration": result.get("analysis_integration", "")
                })
                
            except Exception as e:
                self.logger.error(f"Recommendations error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=get_recommendations,
            name="get_recommendations",
            description=(
                "Get measurement recommendations based on a completed analysis. "
                "Returns suggested follow-up experiments and measurements."
            ),
            parameters={
                "analysis_index": {
                    "type": "integer",
                    "description": "Index of analysis to get recommendations for (-1 for most recent)"
                }
            },
            required=[]
        )

    def _register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: list = None
    ):
        """Register a tool in OpenAI format."""
        self.functions_map[name] = func
        
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
        """Execute a tool by name with given arguments."""
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found"
            })
        
        try:
            return self.functions_map[tool_name](**kwargs)
        except Exception as e:
            logging.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name
            })
