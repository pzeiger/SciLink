"""
Tool definitions and schemas for the PlanningOrchestratorAgent.
Supports both Google Gemini (function objects) and OpenAI (JSON schemas).
"""

from datetime import datetime
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Callable
import hashlib

from .parser_utils import write_experiments_to_disk


class OrchestratorTools:
    """
    Manages tool definitions, schemas, and execution for the OrchestratorAgent.
    """
    
    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: Reference to the parent OrchestratorAgent
        """
        self.orch = orchestrator_instance
        
        # Build function map and schemas
        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []
        self.gemini_functions: list = []
        
        self._register_all_tools()

    def _get_human_feedback_enabled(self) -> bool:
        """
        Get current human feedback setting from orchestrator.
        Returns True if not set (backwards compatible default).
        """
        return getattr(self.orch, '_enable_human_feedback', True)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of file content for deduplication."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logging.warning(f"Could not compute hash for {file_path}: {e}")
            return ""


    def _parse_result_input(self, result_data: str):
        """
        Helper to parse result_data into appropriate format.
        
        Returns:
            - String (text input)
            - String (single file path)
            - List of strings (multiple file paths)
        """
        if len(result_data) < 500:  # Reasonable path length
            try:
                # Check if it's a single file path
                path = Path(result_data.strip())
                if path.exists() and path.is_file():
                    print(f"    (Detected file path: {path.name})")
                    return str(path)
                
                # Check if it's comma-separated file paths
                if ',' in result_data:
                    paths = [p.strip() for p in result_data.split(',')]
                    valid_paths = []
                    for p in paths:
                        p_obj = Path(p)
                        if p_obj.exists():
                            valid_paths.append(p)
                    
                    if valid_paths:
                        print(f"    (Detected {len(valid_paths)} file paths)")
                        return valid_paths
                    else:
                        # Treat as text if no valid paths found
                        text_preview = result_data[:100] + "..." if len(result_data) > 100 else result_data
                        print(f"    (Processing as text: '{text_preview}')")
                        return result_data
                else:
                    # Not a valid path - treat as text
                    text_preview = result_data[:100] + "..." if len(result_data) > 100 else result_data
                    print(f"    (Processing text input: '{text_preview}')")
                    return result_data
                    
            except (OSError, ValueError, RuntimeError):
                # Not a valid path - treat as text
                text_preview = result_data[:100] + "..." if len(result_data) > 100 else result_data
                print(f"    (Processing text input: '{text_preview}')")
                return result_data
        else:
            # Too long to be a path - treat as text
            text_preview = result_data[:100] + "..." if len(result_data) > 100 else result_data
            print(f"    (Processing text input: '{text_preview}')")
            return result_data
        
    def _resolve_data_path(self, path_input: str) -> tuple[str, str]:
        """
        Resolves user input to actual file path with fuzzy matching for typos.
        
        Returns:
            (resolved_path, None) on success
            (None, error_json) on failure (with suggestions if available)
        """
        from difflib import get_close_matches
        
        path = Path(path_input.strip())
        
        # Case 1: Path exists as-is
        if path.exists():
            return str(path), None
        
        # Case 2: Try common extensions if no extension provided
        if not path.suffix:
            for ext in ['.csv', '.xlsx', '.xls']:
                candidate = path.with_suffix(ext)
                if candidate.exists():
                    print(f"    🔍 Resolved: {path.name} → {candidate.name}")
                    return str(candidate), None
        
        # Case 3: Try in common data folders
        search_folders = ['./experimental_results', './data', './results', './']
        all_candidates = []  # Track all files we find for fuzzy matching
        
        if not path.is_absolute():
            stem = path.stem if path.suffix else path.name
            
            for folder in search_folders:
                folder_path = Path(folder)
                if not folder_path.exists():
                    continue
                
                # Collect all data files in this folder
                for ext in ['.csv', '.xlsx', '.xls']:
                    all_candidates.extend(folder_path.glob(f"*{ext}"))
                
                # Try exact match with provided extension
                if path.suffix:
                    candidate = folder_path / path.name
                    if candidate.exists():
                        print(f"    🔍 Found: {path.name} in {folder}/")
                        return str(candidate), None
                
                # Try common extensions
                for ext in ['.csv', '.xlsx', '.xls']:
                    candidate = folder_path / f"{stem}{ext}"
                    if candidate.exists():
                        print(f"    🔍 Found: {stem}{ext} in {folder}/")
                        return str(candidate), None
        
        # Case 4: File not found - use fuzzy matching to suggest alternatives
        if all_candidates:
            # Get filenames without path
            candidate_names = [f.name for f in all_candidates]
            
            # Try fuzzy match on the input filename
            input_name = path.name
            matches = get_close_matches(input_name, candidate_names, n=3, cutoff=0.6)
            
            if matches:
                # Find full paths for the matches
                suggested_files = []
                for match in matches:
                    for candidate in all_candidates:
                        if candidate.name == match:
                            suggested_files.append(str(candidate))
                            break
                
                return None, json.dumps({
                    "status": "error",
                    "message": f"File not found: {path_input}",
                    "did_you_mean": matches,
                    "full_paths": suggested_files,
                    "hint": f"Did you mean '{matches[0]}'? Use: primary_data_set='{suggested_files[0]}'"
                })
        
        # No matches found at all
        return None, json.dumps({
            "status": "error",
            "message": f"Could not find file: {path_input}",
            "searched_in": [str(f) for f in search_folders if Path(f).exists()],
            "hint": "Check filename spelling or use /files command to see available files"
        })
    
    def _register_all_tools(self):
        """Register all tools with both OpenAI and Gemini formats."""
        
        # 0. LIST WORKSPACE FILES
        def list_workspace_files():
            """Lists files in the campaign directory including analysis artifacts."""
            print(f"  ⚡ Tool: Listing files in {self.orch.base_dir}...")
            files = [f.name for f in self.orch.base_dir.iterdir() if f.is_file()]
            artifacts_dir = self.orch.base_dir / "analysis_artifacts"
            artifact_names = []
            if artifacts_dir.exists():
                 artifact_names = [f"analysis_artifacts/{f.name}" for f in artifacts_dir.iterdir() if f.is_file()]
            
            all_files = files + artifact_names
            
            # Include data point count for optimization readiness
            data_count = 0
            if self.orch.bo_data_path.exists():
                try:
                    df = pd.read_csv(self.orch.bo_data_path)
                    data_count = len(df)
                except:
                    pass
            
            return json.dumps({
                "status": "success",
                "files": all_files,
                "data_points_collected": data_count,
                "optimization_ready": data_count >= 3,
                "active_analysis_script": Path(self.orch.active_scalarizer_script).name if self.orch.active_scalarizer_script else None
            })
        
        self._register_tool(
            func=list_workspace_files,
            name="list_workspace_files",
            description="Lists files in the session directory (checkpoints, analysis artifacts, etc.). User data files may exist outside the session folder.",
            parameters={}
        )
        
        # 1. GENERATE INITIAL PLAN
        def generate_initial_plan(
            specific_objective: str = None, 
            knowledge_paths: str = None, 
            primary_data_set: str = None,
            additional_context: str = None
        ):
            """
            Generates experimental plan (science strategy only, no code).
            
            Note: code_paths parameter is deprecated. Use generate_implementation_code() 
            as a separate step to add code after plan approval.
            """
            obj = specific_objective if specific_objective else self.orch.objective
            print(f"  ⚡ Tool: Generating Initial Plan for '{obj}'...")
            
            # Parse knowledge paths
            knowledge_list = None
            if knowledge_paths:
                knowledge_list = [p.strip() for p in knowledge_paths.split(',') if p.strip()]
                
                # Validate paths
                invalid_paths = []
                for path in knowledge_list:
                    if not Path(path).exists():
                        invalid_paths.append(path)
                
                if invalid_paths:
                    return json.dumps({
                        "status": "error",
                        "message": f"Knowledge paths not found: {', '.join(invalid_paths)}",
                        "hint": "Check folder names and spelling"
                    })
                
                print(f"    📚 Knowledge sources: {knowledge_list}")
            
            # Parse primary dataset - UPDATED LOGIC
            primary_dataset = None
            if primary_data_set:
                # Try to resolve the path
                resolved_path, error = self._resolve_data_path(primary_data_set)
                
                if error:
                    return error  # Return the error JSON with suggestions
                
                path = Path(resolved_path)
                
                # Now handle resolved path
                if path.is_file():
                    primary_dataset = {"file_path": str(path)}
                    print(f"    📊 Primary data: {path.name}")
                    
                elif path.is_dir():
                    # Directory - check how many data files
                    all_files = []
                    for ext in ['*.csv', '*.xlsx', '*.xls']:
                        all_files.extend(path.glob(ext))
                    
                    if not all_files:
                        return json.dumps({
                            "status": "error",
                            "message": f"No data files (.csv, .xlsx, .xls) found in: {primary_data_set}",
                            "hint": "Add data files to the folder or specify a different path"
                        })
                    
                    elif len(all_files) == 1:
                        # Only one file - use it automatically
                        primary_dataset = {"file_path": str(all_files[0])}
                        print(f"    📊 Primary data (auto-selected): {all_files[0].name}")
                        
                    else:
                        # Multiple files - require user to specify
                        file_list = sorted([f.name for f in all_files])
                        return json.dumps({
                            "status": "error",
                            "message": f"Multiple data files found in '{primary_data_set}'",
                            "available_files": file_list,
                            "file_count": len(file_list),
                            "hint": f"Please specify which file to use. Example: primary_data_set='./experimental_results/{file_list[0]}'"
                        })
            
            # Build context
            context_parts = []
            
            if additional_context:
                context_parts.append(f"User Requirements: {additional_context}")
                print(f"    ℹ️  User context: {additional_context[:60]}...")
            
            # Auto-include TEA results
            if self.orch.latest_tea_results:
                tea_summary = self.orch.latest_tea_results.get('summary', '')
                context_parts.append(f"Economic Analysis Results: {tea_summary}")
                print(f"    💰 Including TEA results in context")
            
            context_dict = None
            if context_parts:
                context_dict = {"user_context": "\n\n".join(context_parts)}
            
            try:
                # Call the new generate_plan method (not propose_experiments!)
                plan = self.orch.planner.generate_plan(
                    objective=obj,
                    knowledge_paths=knowledge_list,
                    primary_data_set=primary_dataset,
                    additional_context=context_dict,
                    enable_human_feedback=self._get_human_feedback_enabled(),
                    reset_state=False
                )
                
                if plan.get("error"):
                    return json.dumps({
                        "status": "error",
                        "message": plan.get("error")
                    })
                
                # Save
                output_path = self.orch.base_dir / "plan.json"
                with open(output_path, 'w') as f:
                    json.dump(plan, f, indent=2)
                
                # Generate HTML
                from .html_generator import HTMLReportGenerator
                html_path = self.orch.base_dir / "plan.html"
                generator = HTMLReportGenerator(self.orch.planner.state)
                generator.generate(str(html_path))
                
                num_experiments = len(plan.get('proposed_experiments', []))
                
                return json.dumps({
                    "status": "success",
                    "iteration": plan.get('iteration'),
                    "num_experiments": num_experiments,
                    "output_path": str(output_path),
                    "html_report": str(html_path),
                    "knowledge_used": knowledge_list is not None,
                    "primary_data_used": primary_dataset is not None,
                    "tea_context_included": self.orch.latest_tea_results is not None,
                    "hint": "Use generate_implementation_code() to add executable code"
                })
                
            except Exception as e:
                logging.error(f"Plan generation error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })

        # Register it
        self._register_tool(
            func=generate_initial_plan,
            name="generate_initial_plan",
            description=(
                "Generates experimental plan (science strategy only, no implementation code). "
                "Automatically includes previous TEA results if available. "
                "Can use: papers/reports, experimental data, lab constraints."
            ),
            parameters={
                "specific_objective": {"type": "string", "description": "Research objective"},
                "knowledge_paths": {"type": "string", "description": "Comma-separated paths to papers/reports/docs folders"},
                "primary_data_set": {"type": "string", "description": "Path to experimental data file or folder"},
                "additional_context": {"type": "string", "description": "Lab constraints, equipment, reagents, budget, etc."}
            },
            required=[]
        )

        # 2. GENERATE IMPLEMENTATION CODE
        def generate_implementation_code(code_paths: str = None):
            """
            Adds implementation code to the most recent experimental plan.
            Use after generate_initial_plan() to map experiments to executable code.
            
            Args:
                code_paths: Comma-separated paths to code folders. 
                        Optional if Code KB already loaded at startup.
            """
            
            if not self.orch.planner.state or not self.orch.planner.state.get("current_plan"):
                return json.dumps({
                    "status": "error",
                    "message": "No active plan. Generate a plan first using generate_initial_plan()"
                })
            
            current_plan = self.orch.planner.state["current_plan"]
            
            # Check if already has code
            if current_plan.get("proposed_experiments"):
                has_code = any(exp.get("implementation_code") for exp in current_plan["proposed_experiments"])
                if has_code:
                    return json.dumps({
                        "status": "warning",
                        "message": "Plan already has implementation code",
                        "hint": "Generate a new plan if you want to change the code source"
                    })
            
            print(f"  ⚡ Tool: Generating implementation code for existing plan...")

            kb_available = (self.orch.planner.kb_code.index and 
                            self.orch.planner.kb_code.index.ntotal > 0)

            if not kb_available and not code_paths:
                return json.dumps({
                    "status": "error",
                    "message": "No Code Knowledge Base available",
                    "hint": "Provide code_paths parameter (e.g., code_paths='./opentrons_api,./automation_lib')",
                    "available_options": [
                        "Option 1: Specify code_paths='./your_code_folder'",
                        "Option 2: If code exists, check folder name and path"
                    ]
                })
            
            # Parse code paths
            code_list = []
            if code_paths:
                code_list = [p.strip() for p in code_paths.split(',') if p.strip()]
                
                # Validate paths (only if code_paths was provided)
                invalid_paths = []
                for path in code_list:
                    if not Path(path).exists():
                        invalid_paths.append(path)
                
                if invalid_paths:
                    # Check for common typos
                    suggestions = []
                    for invalid in invalid_paths:
                        parent = Path(invalid).parent
                        if parent.exists():
                            similar = [f.name for f in parent.iterdir() 
                                    if f.is_dir() and invalid.lower() in f.name.lower()]
                            if similar:
                                suggestions.append(f"Did you mean './{similar[0]}'?")
                    
                    hint = "Check folder names and spelling."
                    if suggestions:
                        hint += " " + " ".join(suggestions)
                    
                    return json.dumps({
                        "status": "error",
                        "message": f"Code paths not found: {', '.join(invalid_paths)}",
                        "hint": hint
                    })
                
                print(f"    💻 Code sources: {code_list}")
            elif kb_available:
                print(f"    💻 Using existing Code KB ({self.orch.planner.kb_code.index.ntotal} vectors)")
            
            try:
                updated_plan = self.orch.planner.generate_implementation_code(
                    plan=current_plan,
                    code_paths=code_list,
                    enable_human_feedback=self._get_human_feedback_enabled()
                )
                
                if updated_plan.get("error"):
                    return json.dumps({
                        "status": "error",
                        "message": updated_plan.get("error")
                    })
                
                # Save
                output_path = self.orch.base_dir / "plan.json"
                with open(output_path, 'w') as f:
                    json.dump(updated_plan, f, indent=2)
                
                # Regenerate HTML
                from .html_generator import HTMLReportGenerator
                html_path = self.orch.base_dir / "plan.html"
                generator = HTMLReportGenerator(self.orch.planner.state)
                generator.generate(str(html_path))
                
                # Save scripts to output folder
                final_out = str(self.orch.base_dir / "output_scripts")
                print(f"\n--- Saving Scripts to: {final_out} ---")
                write_experiments_to_disk(updated_plan, final_out)
                
                return json.dumps({
                    "status": "success",
                    "message": "Implementation code added to plan",
                    "output_path": str(output_path),
                    "html_report": str(html_path),
                    "scripts_saved_to": final_out,
                    "code_sources_used": code_list
                })
                
            except Exception as e:
                logging.error(f"Code generation error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })

        # Register it
        self._register_tool(
            func=generate_implementation_code,
            name="generate_implementation_code",
            description=(
                "Generates executable implementation code for the most recent experimental plan. "
                "Maps experimental steps to code using API documentation and example repositories. "
                "Use after generate_initial_plan() once the scientific strategy is approved. "
                "If Code KB already loaded, code_paths is optional."
            ),
            parameters={
                "code_paths": {
                    "type": "string",
                    "description": (
                        "Comma-separated paths to code/API folders (e.g., './opentrons_api,./automation_lib'). "
                        "OPTIONAL if Code Knowledge Base is already loaded. "
                        "REQUIRED if no Code KB exists."
                    )
                }
            },
            required=[]
        )
        
        # 3. RUN ECONOMIC ANALYSIS
        def run_economic_analysis(
            focus_topic: str = None,
            knowledge_paths: str = None,
            primary_data_set: str = None,
            additional_context: str = None
        ):
            """Performs Technoeconomic Analysis (TEA)."""
            obj = focus_topic if focus_topic else self.orch.objective
            print(f"  ⚡ Tool: Running TEA for '{obj}'...")
            
            # Parse knowledge paths
            knowledge_list = None
            if knowledge_paths:
                knowledge_list = [p.strip() for p in knowledge_paths.split(',') if p.strip()]
                print(f"    📚 Knowledge sources: {knowledge_list}")
            
            # Parse primary dataset
            primary_dataset = None
            if primary_data_set:
                # Try to resolve the path
                resolved_path, error = self._resolve_data_path(primary_data_set)
                
                if error:
                    return error  # Return the error JSON with suggestions
                
                path = Path(resolved_path)
                
                # Now handle resolved path
                if path.is_file():
                    primary_dataset = {"file_path": str(path)}
                    print(f"    📊 Primary data: {path.name}")
                    
                elif path.is_dir():
                    # Directory - check how many data files
                    all_files = []
                    for ext in ['*.csv', '*.xlsx', '*.xls']:
                        all_files.extend(path.glob(ext))
                    
                    if not all_files:
                        return json.dumps({
                            "status": "error",
                            "message": f"No data files (.csv, .xlsx, .xls) found in: {primary_data_set}",
                            "hint": "Add data files to the folder or specify a different path"
                        })
                    
                    elif len(all_files) == 1:
                        # Only one file - use it automatically
                        primary_dataset = {"file_path": str(all_files[0])}
                        print(f"    📊 Primary data (auto-selected): {all_files[0].name}")
                        
                    else:
                        # Multiple files - require user to specify
                        file_list = sorted([f.name for f in all_files])
                        return json.dumps({
                            "status": "error",
                            "message": f"Multiple data files found in '{primary_data_set}'",
                            "available_files": file_list,
                            "file_count": len(file_list),
                            "hint": f"Please specify which file to use. Example: primary_data_set='./experimental_results/{file_list[0]}'"
                        })
            
            try:
                res = self.orch.planner.perform_technoeconomic_analysis(
                    objective=obj,
                    knowledge_paths=knowledge_list,
                    primary_data_set=primary_dataset,
                    output_json_path=str(self.orch.base_dir / "tea_analysis.json")
                )
                
                if res.get("error"):
                    return json.dumps({
                        "status": "error",
                        "message": res.get("error")
                    })
                
                summary = res.get('technoeconomic_assessment', {}).get('summary', 'No summary')
                
                # Store TEA results in orchestrator state
                self.orch.latest_tea_results = {
                    "summary": summary,
                    "full_analysis": res.get('technoeconomic_assessment'),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"    ✅ TEA results stored for future planning")
                
                return json.dumps({
                    "status": "success",
                    "summary": summary,
                    "output_path": str(self.orch.base_dir / "tea_analysis.json"),
                    "html_report": str(self.orch.base_dir / "tea_analysis.html"),
                    "hint": "These results will automatically inform future generate_initial_plan calls"
                })
                
            except Exception as e:
                logging.error(f"TEA error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })

        self._register_tool(
            func=run_economic_analysis,
            name="run_economic_analysis",
            description=(
                "Performs Technoeconomic Analysis (TEA) to assess economic viability, costs, market fit. "
                "Can incorporate papers and experimental data."
            ),
            parameters={
                "focus_topic": {
                    "type": "string",
                    "description": "Specific technology/process to analyze"
                },
                "knowledge_paths": {
                    "type": "string",
                    "description": "Comma-separated folder paths with papers/PDFs"
                },
                "primary_data_set": {
                    "type": "string",
                    "description": "Path to experimental data file or folder"
                },
                "additional_context": {
                    "type": "string",
                    "description": "Any other relevant context (constraints, requirements, etc.)"
                }
            },
            required=[]
        )
        
        # 4. REFINE PLAN (based on results)
        def refine_plan_with_results(result_data: str, use_literature_rag: bool = False):
            """
            Refines the experimental plan (science strategy only) based on results.
            
            Use this for:
            - Strategic pivots or failures
            - Qualitative observations  
            - Visual analysis of plots/images
            - When experiments didn't go as expected
            
            Supports multiple input formats:
            - Text: "Yield was 12%, precipitation observed"
            - File path: "./data.csv" or "./plot.png"
            - Comma-separated files: "./data.csv,./plot.png"
            """
            print(f"  ⚡ Tool: Refining Plan based on Results...")
            
            # Parse input - handle both single paths and comma-separated lists
            payload = self._parse_result_input(result_data)
            
            try:
                plan = self.orch.planner.refine_plan(
                    results=payload,
                    enable_human_feedback=self._get_human_feedback_enabled(),
                    use_literature_rag=use_literature_rag
                )
                
                if plan.get("error"):
                    return json.dumps({
                        "status": "error",
                        "message": plan.get("error")
                    })
                
                # Save
                output_path = self.orch.base_dir / "plan_refined.json"
                with open(output_path, 'w') as f:
                    json.dump(plan, f, indent=2)
                
                # Generate HTML
                from .html_generator import HTMLReportGenerator
                html_path = self.orch.base_dir / "plan_refined.html"
                generator = HTMLReportGenerator(self.orch.planner.state)
                generator.generate(str(html_path))
                
                return json.dumps({
                    "status": "success",
                    "iteration": plan.get('iteration'),
                    "num_experiments": len(plan.get('proposed_experiments', [])),
                    "output_path": str(output_path),
                    "html_report": str(html_path),
                    "hint": "Use refine_implementation_code() to update executable code"
                })
                
            except Exception as e:
                logging.error(f"Plan refinement error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=refine_plan_with_results,
            name="refine_plan_with_results",
            description=(
                "Refines experimental plan (science strategy only) based on results. "
                "Handles text descriptions, single file paths, or comma-separated files. "
                "Use for: failures, pivots, qualitative observations, or visual analysis. "
                "Does NOT update implementation code - use refine_implementation_code() for that."
            ),
            parameters={
                "result_data": {
                    "type": "string",
                    "description": "Experimental results (text, file path, or comma-separated files)"
                },
                "use_literature_rag": {
                    "type": "boolean", 
                    "description": "Search knowledge base for relevant literature context. Default: false."
                }
            },
            required=["result_data"]
        )
        
        # 5. REFINE IMPLEMENTATION CODE (based on refined plan)
        def refine_implementation_code():
            """
            Updates implementation code for the most recently refined plan.
            Use after refine_plan_with_results() to add/update executable code.
            """
            
            if not self.orch.planner.state or not self.orch.planner.state.get("current_plan"):
                return json.dumps({
                    "status": "error",
                    "message": "No active plan. Refine a plan first using refine_plan_with_results()"
                })
            
            current_plan = self.orch.planner.state["current_plan"]
            
            print(f"  ⚡ Tool: Refining implementation code for iteration {current_plan.get('iteration')}...")
            
            try:
                updated_plan = self.orch.planner.refine_implementation_code(
                    plan=current_plan,
                    enable_human_feedback=self._get_human_feedback_enabled()
                )
                
                if updated_plan.get("error"):
                    return json.dumps({
                        "status": "error",
                        "message": updated_plan.get("error")
                    })
                
                # Save
                output_path = self.orch.base_dir / "plan_refined.json"
                with open(output_path, 'w') as f:
                    json.dump(updated_plan, f, indent=2)
                
                # Regenerate HTML
                from .html_generator import HTMLReportGenerator
                html_path = self.orch.base_dir / "plan_refined.html"
                generator = HTMLReportGenerator(self.orch.planner.state)
                generator.generate(str(html_path))
                
                # Save scripts
                final_out = str(self.orch.base_dir / "output_scripts")
                print(f"\n--- Saving Scripts to: {final_out} ---")
                write_experiments_to_disk(updated_plan, final_out)
                
                return json.dumps({
                    "status": "success",
                    "message": "Implementation code updated",
                    "output_path": str(output_path),
                    "html_report": str(html_path),
                    "scripts_saved_to": final_out
                })
                
            except Exception as e:
                logging.error(f"Code refinement error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=refine_implementation_code,
            name="refine_implementation_code",
            description=(
                "Updates implementation code for the most recently refined plan. "
                "Maps refined experimental steps to executable code. "
                "Use after refine_plan_with_results() once the scientific strategy is approved."
            ),
            parameters={},
            required=[]
        )
        
        def analyze_file(
                file_path: str,
                extraction_goal: str = None,
                force_regenerate: bool = False,
                inputs: list[str] = None,
                targets: list[str] = None):
            """
            Analyzes a raw data file (CSV/XLSX) to extract metrics.
            
            Args:
                file_path: Path to data file
                extraction_goal: What to extract
                force_regenerate: If True, regenerates analysis script even if one exists.
                inputs: List of column names to treat as INPUT parameters for optimization
                targets: List of column names to treat as OPTIMIZATION TARGETS
            """
            print(f"  ⚡ Tool: Analyzing {file_path}...")
            
            if not Path(file_path).exists(): 
                return json.dumps({"status": "error", "message": f"File {file_path} not found"})
            
            # Resolve absolute path for tracking
            file_path_abs = str(Path(file_path).resolve())
            
            #  Build schema-aware extraction goal
            enhanced_objective = extraction_goal or ""
            
            if inputs and targets:
                # User explicitly specified schema - incorporate into the objective query
                schema_instruction = f"""
        REQUIRED OUTPUT SCHEMA:
        - INPUT PARAMETERS (for optimization): {inputs}
        - TARGET METRICS (to optimize): {targets}

        Extract EXACTLY these columns from the data. Each row should contain values for all input parameters and all target metrics.
        For multi-objective optimization, we need BOTH targets: {targets}
        """
                enhanced_objective = f"{enhanced_objective}\n\n{schema_instruction}".strip()
                print(f"    📊 User-specified schema:")
                print(f"       Inputs: {inputs}")
                print(f"       Targets: {targets}")
            
            # Determine script to use
            if force_regenerate:
                script_to_use = None
                print(f"    🔄 Force regenerate: Creating new analysis script")
            else:
                script_to_use = self.orch.active_scalarizer_script if (
                    self.orch.active_scalarizer_script and Path(self.orch.active_scalarizer_script).exists()
                ) else None
                
                if script_to_use: 
                    print(f"    (Consistency Mode: Using cached script)")
                else: 
                    print(f"    (Discovery Mode: Generating new script)")
            
            # Pass schema to experiment context
            current_plan = self.orch.planner.state.get("current_plan", {})
            exp_context = current_plan.get("proposed_experiments", [{}])[0] if current_plan else {}
            
            # Inject schema requirements into context
            if inputs and targets:
                exp_context = exp_context.copy() if exp_context else {}
                exp_context["_schema_requirements"] = {
                    "input_columns": inputs,
                    "target_columns": targets,
                    "optimization_type": "multi-objective" if len(targets) > 1 else "single-objective"
                }
            
            try:
                res = self.orch.scalarizer.scalarize(
                    data_path=file_path, 
                    objective_query=enhanced_objective,  
                    reuse_script_path=script_to_use,
                    experiment_context=exp_context, 
                    enable_human_review=self._get_human_feedback_enabled()
                )
                
                if res["status"] != "success":
                    return json.dumps({
                        "status": "error",
                        "message": res.get('error', 'Analysis failed'),
                        "hint": "Try force_regenerate=True if requirements changed"
                    })
                
                if not self.orch.active_scalarizer_script or force_regenerate:
                    self.orch.active_scalarizer_script = res["source_script"]
                    print(f"    ✅ Analysis Logic Locked: {Path(self.orch.active_scalarizer_script).name}")
                
                # Handle both single-row and multi-row results
                metrics = res["metrics"]
                
                if isinstance(metrics, list):
                    df_new = pd.DataFrame(metrics)
                    print(f"    📊 Processing {len(df_new)} data points from multi-well experiment")
                elif isinstance(metrics, dict):
                    df_new = pd.DataFrame([metrics])
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"Unexpected metrics format: {type(metrics)}"
                    })
                
                # DEDUPLICATION - Content-based tracking
                # Compute current file hash
                current_hash = self._compute_file_hash(file_path)
                current_row_count = len(df_new)

                # Get previous tracking for this file (handle both old and new format)
                prev_tracking = self.orch.analyzed_files.get(file_path_abs, {})
                if isinstance(prev_tracking, dict):
                    prev_hash = prev_tracking.get('hash')
                    prev_row_count = prev_tracking.get('row_count', 0)
                else:
                    # Legacy format: just row count as int
                    prev_hash = None
                    prev_row_count = prev_tracking

                # Check for duplicate content across different filenames
                for tracked_path, tracking_info in self.orch.analyzed_files.items():
                    if tracked_path == file_path_abs:
                        continue  # Skip self
                    tracked_hash = tracking_info.get('hash') if isinstance(tracking_info, dict) else None
                    if tracked_hash and tracked_hash == current_hash:
                        print(f"    ⚠️  Duplicate content detected - matches: {Path(tracked_path).name}")
                        df_final = pd.read_csv(self.orch.bo_data_path) if self.orch.bo_data_path.exists() else pd.DataFrame()
                        return json.dumps({
                            "status": "warning",
                            "message": f"This file's content was already analyzed from '{Path(tracked_path).name}'",
                            "data_points_collected": len(df_final),
                            "rows_added": 0,
                            "optimization_ready": len(df_final) >= 3,
                            "hint": "Data already in optimization set. No action needed unless this is different data with identical content."
                        })

                # Determine what to process based on hash and row count
                if prev_hash is None:
                    # FIRST TIME analyzing this file
                    print(f"    ✨ First time analyzing this file")
                    df_to_append = df_new
                    num_new = len(df_new)

                elif prev_hash != current_hash:
                    # FILE CONTENT CHANGED - reprocess entirely
                    print(f"    🔄 File content changed (hash mismatch) - reprocessing entirely")
                    
                    # Remove old data from optimization_data.csv if it exists
                    if self.orch.bo_data_path.exists() and prev_row_count > 0:
                        try:
                            df_existing = pd.read_csv(self.orch.bo_data_path)
                            # Remove the last prev_row_count rows (assumes they're from this file)
                            if len(df_existing) >= prev_row_count:
                                df_existing = df_existing.iloc[:-prev_row_count]
                                df_existing.to_csv(self.orch.bo_data_path, index=False)
                                print(f"    🗑️  Removed {prev_row_count} old rows from optimization data")
                        except Exception as e:
                            logging.warning(f"Could not clean old data: {e}")
                    
                    df_to_append = df_new
                    num_new = len(df_new)

                elif current_row_count > prev_row_count:
                    # ROWS APPENDED - process only new rows
                    df_new_only = df_new.iloc[prev_row_count:]
                    num_skipped = prev_row_count
                    num_new = len(df_new_only)
                    
                    if num_skipped > 0:
                        print(f"    🔍 Skipped {num_skipped} previously analyzed row(s)")
                    print(f"    ✅ Adding {num_new} NEW row(s)")
                    
                    df_to_append = df_new_only

                elif current_row_count == prev_row_count:
                    # prev_hash == current_hash (guaranteed by earlier elif)
                    # TRULY UNCHANGED
                    print(f"    ℹ️  File unchanged (same content hash)")
                    df_final = pd.read_csv(self.orch.bo_data_path) if self.orch.bo_data_path.exists() else pd.DataFrame()
                    
                    return json.dumps({
                        "status": "success",
                        "message": "File already analyzed - no changes detected",
                        "data_points_collected": len(df_final),
                        "rows_added": 0,
                        "optimization_ready": len(df_final) >= 3
                    })

                else:
                    # FEWER ROWS - file was truncated/replaced
                    print(f"    ⚠️  File has fewer rows ({current_row_count} < {prev_row_count}) - reprocessing")
                    
                    # Remove old data
                    if self.orch.bo_data_path.exists() and prev_row_count > 0:
                        try:
                            df_existing = pd.read_csv(self.orch.bo_data_path)
                            if len(df_existing) >= prev_row_count:
                                df_existing = df_existing.iloc[:-prev_row_count]
                                df_existing.to_csv(self.orch.bo_data_path, index=False)
                                print(f"    🗑️  Removed {prev_row_count} old rows from optimization data")
                        except Exception as e:
                            logging.warning(f"Could not clean old data: {e}")
                    
                    df_to_append = df_new
                    num_new = len(df_new)

                # Schema enforcement BEFORE saving
                all_cols = list(df_to_append.columns)

                # Case 1: Agent explicitly provided schema (Enables MOO)
                if inputs and targets:
                    # Validate that requested columns exist in the extracted data
                    missing_inputs = [c for c in inputs if c not in all_cols]
                    missing_targets = [t for t in targets if t not in all_cols]
                    
                    if missing_inputs or missing_targets:
                        # Try fuzzy matching for column names
                        available_cols = all_cols
                        suggestions = {}
                        
                        for missing in missing_inputs + missing_targets:
                            # Simple fuzzy match: find columns containing similar substrings
                            matches = [c for c in available_cols if missing.lower().replace('_', '') in c.lower().replace('_', '') 
                                    or c.lower().replace('_', '') in missing.lower().replace('_', '')]
                            if matches:
                                suggestions[missing] = matches
                        
                        return json.dumps({
                            "status": "error",
                            "message": "Requested columns not found in extracted metrics",
                            "missing_inputs": missing_inputs,
                            "missing_targets": missing_targets,
                            "available_columns": all_cols,
                            "suggestions": suggestions if suggestions else None,
                            "hint": "Column names may differ slightly. Check available_columns and retry with correct names, or use force_regenerate=True with updated extraction_goal."
                        })
                    
                    self.orch.expected_input_columns = inputs
                    self.orch.expected_target_columns = targets
                    print(f"    📊 Schema Enforced (User-Specified):")
                    print(f"       Inputs: {self.orch.expected_input_columns}")
                    print(f"       Targets: {self.orch.expected_target_columns}")
                
                # Case 2: Schema already established from previous analysis
                elif self.orch.expected_input_columns and self.orch.expected_target_columns:
                    print(f"    📊 Schema Enforced (From Previous Analysis):")
                    print(f"       Inputs: {self.orch.expected_input_columns}")
                    print(f"       Targets: {self.orch.expected_target_columns}")
                
                # Case 3: Fallback - Auto-detect (Single-Objective default)
                else:
                    # Heuristic: numeric columns that look like targets go to targets
                    # This is a last resort - prefer explicit schema
                    self.orch.expected_target_columns = [all_cols[-1]]
                    self.orch.expected_input_columns = [c for c in all_cols if c != all_cols[-1]] 
                    print(f"    📊 Schema Auto-Detected (Default Single-Objective):")
                    print(f"       Inputs: {self.orch.expected_input_columns}")
                    print(f"       Targets: {self.orch.expected_target_columns}")
                    print(f"    ⚠️  Warning: Using auto-detected schema. For multi-objective optimization, specify inputs and targets explicitly.")
                
                # SCHEMA ENFORCEMENT ON SAVE
                if self.orch.bo_data_path.exists():
                    df_existing = pd.read_csv(self.orch.bo_data_path)
                    
                    if set(df_to_append.columns) != set(df_existing.columns):
                        return json.dumps({
                            "status": "error",
                            "message": "Schema mismatch detected",
                            "expected_columns": list(df_existing.columns),
                            "received_columns": list(df_to_append.columns),
                            "hint": "All data must have same structure. Use reset_analysis_logic to start fresh."
                        })
                    
                    df_to_append = df_to_append[df_existing.columns]
                    df_to_append.to_csv(self.orch.bo_data_path, mode='a', header=False, index=False)
                else:
                    df_to_append.to_csv(self.orch.bo_data_path, mode='w', header=True, index=False)
                
                # Update tracking
                self.orch.analyzed_files[file_path_abs] = {
                    'row_count': current_row_count,
                    'hash': current_hash,
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.orch.analyzed_files_path, 'w') as f:
                    json.dump(self.orch.analyzed_files, f, indent=2)
                
                df_final = pd.read_csv(self.orch.bo_data_path)
                data_count = len(df_final)
                
                return json.dumps({
                    "status": "success",
                    "metrics": metrics if isinstance(metrics, dict) else f"{len(metrics)} data points",
                    "data_points_collected": data_count,
                    "rows_added": num_new,
                    "optimization_ready": data_count >= 3,
                    "schema": {
                        "inputs": self.orch.expected_input_columns,
                        "targets": self.orch.expected_target_columns
                    }
                })
                
            except Exception as e:
                logging.error(f"Analyze file error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=analyze_file,
            name="analyze_file",
            description=(
                "Analyzes raw data files (CSV/XLSX/TXT) to extract scalar metrics. "
                "Automatically generates analysis code on first use, then reuses it for consistency. "
                "Results are appended to optimization dataset."
            ),
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file to analyze (e.g., 'results/run_001.csv')"
                },
                "extraction_goal": {
                    "type": "string",
                    "description": "Natural language description of what to extract (e.g., 'Calculate peak area and retention time')"
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": (
                        "If true, generates new analysis script even if one exists. "
                        "Use when analysis requirements change (e.g., switching from single-row to multi-row extraction, "
                        "or changing which metrics to extract). Default: false"
                    )
                },
                "inputs": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of column names to treat as INPUT parameters"
                },
                "targets": {
                    "type": "array", 
                    "items": {"type": "string"}, 
                    "description": "List of column names to treat as OPTIMIZATION TARGETS"
                }
            },
            required=["file_path"]
        )
        
        # 7. RESET ANALYSIS LOGIC
        def reset_analysis_logic():
            """Resets the analysis script, optimization data, AND file tracking."""
            self.orch.active_scalarizer_script = None
            self.orch.expected_input_columns = None
            self.orch.expected_target_columns = []
            
            # Clear file tracking completely
            self.orch.analyzed_files = {}
            if self.orch.analyzed_files_path.exists():
                try:
                    self.orch.analyzed_files_path.unlink()
                    print(f"    🗑️  Cleared file tracking history")
                except Exception as e:
                    logging.warning(f"Could not delete analyzed_files.json: {e}")
            
            if self.orch.bo_data_path.exists():
                backup_path = self.orch.bo_data_path.with_suffix('.csv.backup')
                self.orch.bo_data_path.rename(backup_path)
                print(f"    ⚠️  Old data backed up to: {backup_path.name}")
            
            return json.dumps({
                "status": "success",
                "message": "Analysis logic reset. All files will be reprocessed fresh on next analyze_file call.",
                "hint": "Previous optimization data was backed up"
            })
        
        self._register_tool(
            func=reset_analysis_logic,
            name="reset_analysis_logic",
            description=(
                "Resets the locked analysis script and clears optimization data. "
                "Use this when the current analysis approach is fundamentally wrong. "
                "Previous data is backed up before deletion."
            ),
            parameters={},
            required=[]
        )
        
        # 8. RUN OPTIMIZATION
        def run_optimization(parallel_capable: bool = False, batch_size: int = None):
            """
            Runs Bayesian Optimization to suggest next parameters.
            """
            print(f"  ⚡ Tool: Running Bayesian Optimization...")
            
            # --- PRE-FLIGHT CHECKS ---
            if not self.orch.active_scalarizer_script:
                return json.dumps({
                    "status": "error",
                    "message": "No analysis script locked yet",
                    "hint": "Run analyze_file on at least 3 data files first",
                    "workflow": "analyze_file (×3) → run_optimization"
                })
            
            if not self.orch.bo_data_path.exists():
                return json.dumps({
                    "status": "error",
                    "message": "No optimization_data.csv found",
                    "hint": "Run analyze_file to collect data points first"
                })
            
            try:
                df = pd.read_csv(self.orch.bo_data_path)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to read optimization data: {e}",
                    "hint": "CSV may be corrupted. Check optimization_data.csv"
                })
            
            if len(df) < 3:
                return json.dumps({
                    "status": "error", 
                    "message": f"Insufficient data points: {len(df)}/3",
                    "hint": "Collect at least 3 experimental results before optimizing",
                    "current_data_count": len(df)
                })
            
            # Check for plural target columns list
            if not self.orch.expected_target_columns or not self.orch.expected_input_columns:
                return json.dumps({
                    "status": "error", 
                    "message": "Schema not established",
                    "hint": "This shouldn't happen. Try reset_analysis_logic."
                })
            
            # SCHEMA VALIDATION
            missing_targets = [t for t in self.orch.expected_target_columns if t not in df.columns]
            if missing_targets:
                return json.dumps({
                    "status": "error",
                    "message": f"Target columns missing from data: {missing_targets}",
                    "available_columns": list(df.columns)
                })
            
            missing_inputs = [c for c in self.orch.expected_input_columns if c not in df.columns]
            if missing_inputs:
                return json.dumps({
                    "status": "error",
                    "message": f"Input columns missing: {missing_inputs}",
                    "available_columns": list(df.columns)
                })
            
            #  Add list of targets to critical columns
            critical_cols = self.orch.expected_input_columns + self.orch.expected_target_columns
            
            if df[critical_cols].isnull().any().any():
                return json.dumps({
                    "status": "error",
                    "message": "Missing values detected in optimization data",
                    "hint": "Ensure all data files were analyzed successfully",
                    "affected_rows": df[df[critical_cols].isnull().any(axis=1)].index.tolist()
                })
            
            # ============================================
            # BOUNDS & CONSTRAINTS CALCULATION (Unchanged)
            # ============================================

            # 1. Fetch Scientific Constraints from Planner State
            scientific_bounds = {}
            current_plan = self.orch.planner.state.get("current_plan", {})
            
            if current_plan and "proposed_experiments" in current_plan:
                for exp in current_plan["proposed_experiments"]:
                    for param in exp.get("optimization_params", []):
                        name = param.get("parameter_name")
                        min_v = param.get("min_value")
                        max_v = param.get("max_value")
                        
                        if name and min_v is not None and max_v is not None:
                            scientific_bounds[name] = (float(min_v), float(max_v))
                            print(f"  🔬 Scientific Constraint Found: {name} must be between {min_v} and {max_v}")

            # 2. Identify numeric inputs
            numeric_inputs = []
            for col in self.orch.expected_input_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    numeric_inputs.append(col)
                else:
                    print(f"  ⚠️ Skipping non-numeric input column: {col}")

            if not numeric_inputs:
                return json.dumps({
                    "status": "error", 
                    "message": "No numeric input parameters found. Optimization requires at least one numeric parameter (float/int)."
                })

            # Update state to only track numeric inputs for optimization
            self.orch.expected_input_columns = numeric_inputs

            # 3. Calculate Final Bounds
            input_bounds = []
            for col in numeric_inputs:
                if col in scientific_bounds:
                    sci_min, sci_max = scientific_bounds[col]
                    input_bounds.append([sci_min, sci_max])
                    print(f"     -> Bound for '{col}': [{sci_min}, {sci_max}] (Source: PLANNER)")
                else:
                    data_min = float(df[col].min())
                    data_max = float(df[col].max())
                    
                    if data_min == data_max:
                        margin = 1.0 if data_min == 0 else abs(data_min * 0.1)
                    else:
                        margin = (data_max - data_min) * 0.1
                        
                    safe_min = data_min - margin
                    safe_max = data_max + margin
                    
                    input_bounds.append([safe_min, safe_max])
                    print(f"     -> Bound for '{col}': [{safe_min:.2f}, {safe_max:.2f}] (Source: DATA STATISTICS)")
            
            # ============================================
            # BATCH SIZE DETERMINATION (Unchanged)
            # ============================================
            
            if not parallel_capable:
                final_batch_size = 1
                mode_desc = "sequential (single experiment)"
            else:
                if batch_size is None:
                    return json.dumps({
                        "status": "batch_size_required",
                        "message": "Batch size must be specified for parallel optimization.",
                        "instruction": (
                            "Analyze the experimental plan to determine appropriate batch_size "
                            "(e.g., plate format, number of conditions, equipment capacity), "
                            "then call: run_optimization(parallel_capable=True, batch_size=N)"
                        ),
                        "hint": "Common values: 8, 12, 24, 96, 384 for plate-based experiments"
                    })
                
                if batch_size < 1:
                    return json.dumps({
                        "status": "error", 
                        "message": f"Invalid batch_size: {batch_size}. Must be at least 1."
                    })
                
                final_batch_size = batch_size
                mode_desc = f"parallel (batch of {batch_size})"
                print(f"    ℹ️  Using batch_size: {batch_size}")
            
            print(f"    📊 Optimization Setup:")
            print(f"       Mode: {mode_desc}")
            print(f"       Data points: {len(df)}")
            print(f"       Inputs: {self.orch.expected_input_columns}")
            print(f"       Targets: {self.orch.expected_target_columns}") # Log targets list
            print(f"       Bounds: {input_bounds}")
            
            try:
                # Pass list of targets directly
                res = self.orch.bo.run_optimization_loop(
                    data_path=str(self.orch.bo_data_path),
                    objective_text=self.orch.objective,
                    input_cols=self.orch.expected_input_columns,
                    input_bounds=input_bounds,                    
                    target_cols=self.orch.expected_target_columns,
                    output_dir=str(self.orch.base_dir / "bo_artifacts"),
                    batch_size=int(final_batch_size)
                )
                
                if res.get("status") != "success":
                    return json.dumps({
                        "status": "error", 
                        "message": res.get("error", "Optimization failed"),
                        "bo_output": res
                    })
                
                # Format response
                next_params = res.get('next_parameters')
                
                if parallel_capable:
                    hint = f"Run all {final_batch_size} experiments in parallel, then use analyze_file on each result file."
                    params_summary = f"Generated {final_batch_size} parameter sets"
                else:
                    hint = "Run this experiment, then use analyze_file on the result to continue."
                    params_summary = "Generated next experiment parameters"
                
                return json.dumps({
                    "status": "success",
                    "mode": "parallel" if parallel_capable else "sequential",
                    "batch_size": final_batch_size,
                    "recommended_parameters": next_params,
                    "params_summary": params_summary,
                    "strategy_used": res.get('strategy', {}).get('acquisition_strategy', {}).get('type'),
                    "plot_path": res.get('plot_path'),
                    "hint": hint
                })
                
            except Exception as e:
                logging.error(f"Optimization error: {e}")
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=run_optimization,
            name="run_optimization",
            description=(
                "Runs Bayesian Optimization to suggest next experimental parameters. "
                "Requires at least 3 data points from analyze_file. "
                "For parallel mode, batch_size must be specified."
            ),
            parameters={
                "parallel_capable": {
                    "type": "boolean",
                    "description": "True if experiments can run in parallel. False for sequential (default)."
                },
                "batch_size": {
                    "type": "integer",
                    "description": (
                        "Number of parallel experiments (required if parallel_capable=True). "
                        "Infer from experimental plan (e.g., plate format, grid size, equipment capacity)."
                    )
                }
            },
            required=[]
        )

        # 9. SAVE CHECKPOINT
        def save_checkpoint():
            """
            Saves complete orchestrator state including conversation and agent state.
            Use this periodically during long campaigns.
            """
            checkpoint_path = self.orch.base_dir / "checkpoint.json"
            
            # Calculate data points
            data_points = 0
            if self.orch.bo_data_path.exists():
                try:
                    df = pd.read_csv(self.orch.bo_data_path)
                    data_points = len(df)
                except:
                    pass
            
            # Get message count (handle both OpenAI and Gemini)
            if self.orch.use_openai:
                # OpenAI: messages is a list attribute
                message_count = len(self.orch.messages)
            else:
                # Gemini: history is in chat_session
                try:
                    message_count = len(self.orch.chat_session.history) if hasattr(self.orch.chat_session, 'history') else 0
                except:
                    message_count = 0
            
            state = {
                "timestamp": datetime.now().isoformat(),
                "objective": self.orch.objective,
                "active_scalarizer_script": self.orch.active_scalarizer_script,
                "expected_input_columns": self.orch.expected_input_columns,
                "expected_target_columns": self.orch.expected_target_columns,
                "data_points_collected": data_points,
                "message_count": message_count,
                "planner_state": self.orch.planner.state if hasattr(self.orch.planner, 'state') else None,
                "latest_tea_results": self.orch.latest_tea_results,
                "autonomy_level": self.orch.autonomy_level.value if hasattr(self.orch, 'autonomy_level') and self.orch.autonomy_level else None,
                "data_dir": str(self.orch.data_dir) if self.orch.data_dir else None,
                "knowledge_dir": str(self.orch.knowledge_dir) if self.orch.knowledge_dir else None,
                "code_dir": str(self.orch.code_dir) if self.orch.code_dir else None,
            }
            
            try:
                with open(checkpoint_path, 'w') as f:
                    json.dump(state, f, indent=2)
                
                print(f"    💾 Checkpoint saved: {checkpoint_path}")
                
                return json.dumps({
                    "status": "success",
                    "checkpoint_path": str(checkpoint_path),
                    "data_points": data_points,
                    "message_count": message_count,
                    "timestamp": state["timestamp"]
                })
                
            except Exception as e:
                logging.error(f"Checkpoint save failed: {e}")
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to save checkpoint: {e}"
                })
        
        # Register the tool
        self._register_tool(
            func=save_checkpoint,
            name="save_checkpoint",
            description=(
                "Saves complete campaign state including conversation history, "
                "analysis scripts, and optimization data. Use this periodically "
                "during long campaigns (every 3-5 experiments) to enable resumption "
                "after crashes or breaks."
            ),
            parameters={},
            required=[]
        )

        # 10. DISCARD PLAN
        def discard_plan(reason: str = ""):
            """
            Discards the most recent experimental plan (marks it as superseded).
            The plan remains in history for transparency but won't appear in reports.
            
            Args:
                reason: Why the plan is being discarded
            """
            if not self.orch.planner.state:
                return json.dumps({
                    "status": "error",
                    "message": "No active planning session"
                })
            
            history = self.orch.planner.state.get("plan_history", [])
            
            if not history:
                return json.dumps({
                    "status": "error",
                    "message": "No plans in history to discard"
                })
            
            # Find last non-TEA, non-superseded entry
            for i in range(len(history) - 1, -1, -1):
                plan = history[i]
                if (plan.get("type") != "technoeconomic_analysis" and 
                    plan.get("status") != "superseded"):
                    
                    # Mark as superseded instead of deleting
                    plan["status"] = "superseded"
                    plan["superseded_reason"] = reason if reason else "Plan replaced with corrected version"
                    plan["superseded_at"] = datetime.now().isoformat()
                    
                    print(f"    🗑️  Discarded plan: iteration {plan.get('iteration')}")
                    if reason:
                        print(f"       Reason: {reason}")
                    
                    return json.dumps({
                        "status": "success",
                        "message": f"Plan from iteration {plan.get('iteration')} discarded",
                        "reason": plan["superseded_reason"],
                        "hint": "The discarded plan remains in history for transparency"
                    })
            
            return json.dumps({
                "status": "error",
                "message": "No active experimental plans to discard"
            })

        # Register the tool
        self._register_tool(
            func=discard_plan,
            name="discard_plan",
            description=(
                "Discards the most recent experimental plan (marks it as superseded). "
                "The plan remains in full history for transparency but won't appear in final reports. "
                "Use when correcting a wrong plan before generating the corrected version."
            ),
            parameters={
                "reason": {
                    "type": "string",
                    "description": (
                        "Why the plan is being discarded. Be specific about the mismatch. "
                        "Examples: 'Wrong material - data has Mg not Mn', "
                        "'User requested different equipment', 'Incorrect objective interpretation'"
                    )
                }
            },
            required=["reason"]
        )

        def show_directory_guide():
            """
            Shows the recommended directory structure for optimal agent performance.
            """
            guide = """
        ╔══════════════════════════════════════════════════════════════════════════╗
        ║                  RECOMMENDED DIRECTORY STRUCTURE                         ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        📁 my_research_project/          ← Run orchestrator from here
        │
        ├── 📚 papers/                    ← Scientific papers & literature
        │   ├── separation_methods_2024.pdf
        │   ├── lithium_extraction_review.pdf
        │   └── rare_earth_recovery.pdf
        │
        ├── 📊 experimental_results/      ← Raw experimental data files
        │   ├── batch_001.csv
        │   ├── batch_002.csv
        │   ├── batch_003.csv
        │   └── pilot_run_*.xlsx
        │
        ├── 💻 code/                      ← Analysis scripts & API docs (optional)
        │   ├── analysis_pipeline.py
        │   ├── visualization.py
        │   └── api_documentation/
        │
        ├── 📁 campaign_session/          ← Created automatically by orchestrator
        │   ├── optimization_data.csv    (collected metrics)
        │   ├── analysis_artifacts/      (generated analysis scripts)
        │   ├── bo_artifacts/            (optimization plots)
        │   ├── plan.json                (experimental plans)
        │   └── checkpoint.json          (saved state)
        │
        └── 🗂️ kb_storage/                ← Created automatically
            ├── default_kb_docs/         (knowledge base from papers)
            └── default_kb_code/         (knowledge base from code)

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║                           QUICK START GUIDE                              ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        CHAT EXAMPLES:

        📋 Generate plan with papers:
        "Generate a plan for lithium extraction using ./papers/ and ./code/"

        📊 Analyze experimental data:
        "Analyze ./experimental_results/batch_001.csv and extract yield"

        🔬 Run optimization:
        "Run optimization to suggest next experiments"

        💾 Save progress:
        "Save checkpoint"
        """
            
            print(guide)
            
            # Also return as JSON for the LLM
            return json.dumps({
                "status": "success",
                "message": "Directory structure guide displayed",
                "recommended_folders": ["papers/", "experimental_results/", "code/"],
                "auto_created_folders": ["campaign_session/", "kb_storage/"]
            })

        # Register the tool
        self._register_tool(
            func=show_directory_guide,
            name="show_directory_guide",
            description=(
                "Shows recommended directory structure for optimal agent performance. "
                "Use when user asks about setup, organization, or how to structure their project."
            ),
            parameters={},
            required=[]
        )
    
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
            logging.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name
            })


