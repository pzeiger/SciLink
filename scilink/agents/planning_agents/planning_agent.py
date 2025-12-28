import os
from pathlib import Path
import google.generativeai as genai
import json
import logging
import shutil
import uuid
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import PIL.Image as PIL_Image

from .knowledge_base import KnowledgeBase
from .excel_parser import parse_adaptive_excel
from .parser_utils import (
    generate_repo_map, 
    write_experiments_to_disk,
    resolve_primary_data_path
)
from .repo_loader import clone_git_repository

from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS
)

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ..lit_agents.literature_agent import LiteratureSearchAgent
from ..lit_agents.optimize_query import optimize_search_query

from .rag_engine import (
    perform_science_rag, 
    perform_code_rag, 
    refine_plan_with_feedback,
    refine_code_with_feedback,
    verify_plan_relevance
)

from .ingestor import ingest_files, extract_images

from .user_interface import display_plan_summary, get_user_feedback

from .html_generator import HTMLReportGenerator


class PlanningAgent:
    """
    Stateful AI Agent for Autonomous Experimental Planning and Iteration.
    
    The PlanningAgent orchestrates end-to-end research workflows by combining:
    - Dual Knowledge Base system (scientific literature + implementation code)
    - RAG-based hypothesis generation and technoeconomic analysis
    - LLM-driven code generation from experimental procedures
    - Human-in-the-loop feedback at strategic decision points
    - Iterative refinement based on experimental results
    
    Maintains a persistent 'state' dictionary to track:
    - The Research Objective
    - The Evolving Experimental Plan (Science -> Code)
    - Results from executed experiments
    - Feedback history (both Scientific Plan and Code Implementation)

    Args:
        google_api_key (str, optional): API key for Gemini models.
            If not provided, attempts to load from environment.
        futurehouse_api_key (str, optional): FutureHouse API key for literature search.
            If not provided, literature search will be skipped.
        model_name (str, optional): Name of the LLM to use. 
            Defaults to "gemini-3-pro-preview".
        local_model (str, optional): Base URL for OpenAI-compatible local models.
            If provided, uses OpenAI wrapper instead of Gemini.
        embedding_model (str, optional): Embedding model for knowledge bases.
            Defaults to "gemini-embedding-001".
        kb_base_path (str, optional): Base path for knowledge base storage.
            Creates separate `_docs` and `_code` knowledge bases.
            Defaults to "./kb_storage/default_kb".
        code_chunk_size (int, optional): Chunk size for code files in tokens.
            Defaults to 20000 (larger than docs for context preservation).
    """
    def __init__(self, google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 model_name: str = "gemini-3-pro-preview",
                 local_model: str = None,
                 embedding_model: str = "gemini-embedding-001",
                 kb_base_path: str = "./kb_storage/default_kb",
                 code_chunk_size: int = 20000): 
        
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')

        # --- LLM Backend Configuration ---
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            logging.info(f"🏛️  Using OpenAI-compatible model for generation: {model_name}")
            self.model = OpenAIAsGenerativeModel(model_name, api_key=google_api_key, base_url=local_model)
            self.generation_config = None
        else:
            logging.info(f"☁️  Using Google Gemini model for generation: {model_name}")
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        self.lit_agent = None
        if futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY"):
            try:
                self.lit_agent = LiteratureSearchAgent(futurehouse_api_key, max_wait_time=1000)
                logging.info("✅ Literature Search Agent initialized.")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialize Literature Agent: {e}")
        else:
            logging.info("ℹ️ No FutureHouse API key provided. Literature search will be skipped.")
                    
        self.code_chunk_size = code_chunk_size

        # --- Dual KnowledgeBase Initialization ---
        base_path = Path(kb_base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Scientific/Docs KB
        self.kb_docs = KnowledgeBase(google_api_key=google_api_key, embedding_model=embedding_model, local_model=local_model)
        self.kb_docs_prefix = base_path.parent / f"{base_path.name}_docs"
        self.kb_docs_index = str(self.kb_docs_prefix.with_suffix(".faiss"))
        self.kb_docs_chunks = str(self.kb_docs_prefix.with_suffix(".json"))
        self.kb_docs_sources_path = str(self.kb_docs_prefix.with_suffix(".sources.json"))

        # 2. Implementation/Code KB
        self.kb_code = KnowledgeBase(google_api_key=google_api_key, embedding_model=embedding_model, local_model=local_model)
        self.kb_code_prefix = base_path.parent / f"{base_path.name}_code"
        self.kb_code_index = str(self.kb_code_prefix.with_suffix(".faiss"))
        self.kb_code_chunks = str(self.kb_code_prefix.with_suffix(".json"))
        self.kb_code_map_path = str(self.kb_code_prefix.with_suffix(".maps.json"))
        self.kb_code_sources_path = str(self.kb_code_prefix.with_suffix(".sources.json"))

        print("--- Initializing Agent (Dual-KB System) ---")
        self._load_knowledge_bases()

        # --- STATE MANAGEMENT ---
        self.state: Dict[str, Any] = {}

    def restore_state(self, state_file_path: str) -> None:
        """
        Restore agent state from a saved .state.json file.
        
        Args:
            state_file_path: Path to the .state.json file
            
        Example:
            agent = PlanningAgent()
            agent.restore_state("./outputs/session.state.json")
        """        
        path = Path(state_file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {state_file_path}")
        
        if path.suffix != '.json':
            raise ValueError(f"State file must be a .json file, got: {path.suffix}")
        
        print(f"  - 📂 Loading state from: {path.name}")
        
        try:
            with open(path, 'r') as f:
                saved_state = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in state file: {e}")
        
        # Validate structure
        required = ["objective", "current_plan", "iteration_index", "session_id"]
        missing = [f for f in required if f not in saved_state]
        
        if missing:
            raise ValueError(
                f"Invalid state file structure. Missing required fields: {missing}\n"
                f"Expected a complete .state.json file with keys: {required}"
            )
        
        # Restore
        self.state = saved_state
        
        # User feedback
        print(f"  - ✅ Restored session: {saved_state['session_id']}")
        print(f"     • Objective: {saved_state['objective'][:80]}...")
        print(f"     • Current iteration: {saved_state['iteration_index']}")
        print(f"     • History entries: {len(saved_state.get('plan_history', []))}")
        print(f"     • Previous results: {len(saved_state.get('experimental_results', []))}")
        
    def _load_knowledge_bases(self):
        """Attempts to load both KBs from disk."""
        print(f"  - Docs KB: Loading from {self.kb_docs_prefix}...")
        docs_loaded = self.kb_docs.load(
            self.kb_docs_index, self.kb_docs_chunks,
            sources_path=self.kb_docs_sources_path
        )
        
        print(f"  - Code KB: Loading from {self.kb_code_prefix}...")
        code_loaded = self.kb_code.load(
            self.kb_code_index, self.kb_code_chunks, self.kb_code_map_path,
            sources_path=self.kb_code_sources_path
        )

        self._kb_is_built = docs_loaded or code_loaded
        
        if docs_loaded: print("    - ✅ Docs KB loaded.")
        if code_loaded: print("    - ✅ Code KB loaded.")
        if not self._kb_is_built: print("    - ⚠️  No pre-built KBs found.")

    def _initialize_state(self, objective: str, **kwargs) -> Dict[str, Any]:
        """Creates the foundational state dictionary for a new research task."""
        return {
            "session_id": str(uuid.uuid4()),
            "start_time": datetime.now().isoformat(),
            "objective": objective,
            "iteration_index": 0,
            
            # Inputs
            "inputs": {
                "knowledge_paths": kwargs.get("knowledge_paths", []),
                "code_paths": kwargs.get("code_paths", []),
                "additional_context": kwargs.get("additional_context"),
                "primary_data_set": kwargs.get("primary_data_set"),
                "image_paths": kwargs.get("image_paths", []),
                "image_descriptions": kwargs.get("image_descriptions", [])
            },

            # Plan Evolution
            "current_plan": None,   # The active plan dict
            "plan_history": [],     # Snapshots of previous plans
            
            # Feedback Loop
            "experimental_results": [],  # List of result dicts from the lab
            "human_feedback_history": [],
            
            # Status
            "last_error": None,
            "status": "initialized"
        }

    def _save_results_to_json(self, results: Dict[str, Any], file_path: str):
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f: json.dump(results, f, indent=2)
            print(f"    - ✅ Results successfully saved to: {file_path}")
        except Exception as e: logging.error(f"    - ❌ Failed to save results: {e}")

    def _save_state_to_json(self, file_path: str):
        """Saves the full state dictionary (history, results) to a sidecar file."""
        try:
            p = Path(file_path)
            with p.open('w', encoding='utf-8') as f: json.dump(self.state, f, indent=2)
        except Exception as e: logging.error(f"    - ❌ Failed to save state: {e}")

    def _build_and_save_kb(self, knowledge_paths: Optional[List[str]] = None, code_paths: Optional[List[str]] = None) -> bool:
        print("\n--- Rebuilding Knowledge Bases ---")
        
        # 1. Science KB
        doc_chunks = []
        if knowledge_paths:
            print(f"Processing {len(knowledge_paths)} Scientific Paths...")
            doc_chunks.extend(ingest_files(knowledge_paths, is_code_mode=False))

        if doc_chunks:
            print(f"  - Building Scientific KB with {len(doc_chunks)} chunks...")
            self.kb_docs.build(doc_chunks)
            self.kb_docs.save(self.kb_docs_index, self.kb_docs_chunks, sources_path=self.kb_docs_sources_path)
        else:
            print("  - ℹ️  No Scientific docs provided. Docs KB unchanged.")

        # 2. Code KB
        code_chunks = []
        if code_paths:
            print(f"Processing {len(code_paths)} Code Paths...")
            for p in code_paths:
                path_obj = Path(p)
                if path_obj.is_dir():
                    repo_name = path_obj.name
                    print(f"  - 📦 Processing Repo: {repo_name}")
                    self.kb_code.repo_maps[repo_name] = generate_repo_map(str(path_obj))
                    code_chunks.extend(ingest_files([p], is_code_mode=True, code_chunk_size=self.code_chunk_size, repo_name=repo_name))
                else:
                    code_chunks.extend(ingest_files([p], is_code_mode=True, code_chunk_size=self.code_chunk_size))
            
        if code_chunks:
            print(f"  - Building Code KB with {len(code_chunks)} chunks...")
            self.kb_code.build(code_chunks)
            self.kb_code.save(self.kb_code_index, self.kb_code_chunks, self.kb_code_map_path, self.kb_code_sources_path)
        else:
            print("  - ℹ️  No Code docs provided. Code KB unchanged.")

        self._kb_is_built = True
        return True

    def _ensure_kb_is_ready(self, knowledge_paths: Optional[List[str]] = None, code_paths: Optional[List[str]] = None) -> bool:
        new_science = self.kb_docs.source_difference(knowledge_paths)
        new_code = self.kb_code.source_difference(code_paths)
        
        if new_science or new_code:
            return self._build_and_save_kb(new_science, new_code)
        elif not self._kb_is_built:
            logging.error("Knowledge base is not built.")
            return False
        return True
    
    def generate_plan(self,
                    objective: str,
                    knowledge_paths: Optional[List[str]] = None,
                    primary_data_set: Optional[Union[str, Dict[str, str]]] = None,
                    additional_context: Optional[Dict[str, str]] = None,
                    image_paths: Optional[List[str]] = None,
                    image_descriptions: Optional[List[str]] = None,
                    enable_human_feedback: bool = True,
                    reset_state: bool = False) -> Dict[str, Any]:
        """
        Generate experimental plan (science only, no implementation code/protocol).
        
        This method performs:
        1. Knowledge base initialization (docs only)
        2. Literature search (optional)
        3. RAG-based hypothesis generation
        4. Self-correction loop
        5. Human feedback on strategy
        
        Does NOT generate implementation code. Use generate_implementation_code() for that.
        
        Returns:
            Dict with proposed_experiments
        """
        
        # Resolve data and images
        primary_data_set = resolve_primary_data_path(primary_data_set)
        manual_images = image_paths or []
        auto_images = [img for img in extract_images(knowledge_paths) if img not in manual_images]
        all_image_paths = manual_images + auto_images
        
        # Initialize or update state
        if reset_state or not self.state:
            self.state = self._initialize_state(
                objective=objective,
                knowledge_paths=knowledge_paths,
                code_paths=None,  # ← Not used in plan generation
                additional_context=additional_context,
                primary_data_set=primary_data_set,
                image_paths=all_image_paths,
                image_descriptions=image_descriptions
            )
        else:
            print(f"  - 🔄 Appending to existing research session...")
            if objective:
                self.state["objective"] = objective
        
        # Increment iteration
        existing_iter = self.state.get("iteration_index", 0)
        self.state["iteration_index"] = existing_iter + 1
        current_iter = self.state["iteration_index"]
        
        # Build KB (docs only)
        if not self._ensure_kb_is_ready(knowledge_paths, code_paths=None):
            self.state["status"] = "failed"
            self.state["last_error"] = "KB Init Failed"
            return self.state
        
        # Build context string
        ctx_string = ""
        if additional_context:
            for header, content in additional_context.items():
                ctx_string += f"## {header}\n{content}\n\n"
            ctx_string = ctx_string.strip() if ctx_string else None
        
        # Literature search
        lit_context = ""
        if self.lit_agent:
            print(f"  - 🌍 Querying literature...")
            lit_res = self.lit_agent.search_for_hypothesis_context(
                optimize_search_query(objective=objective, model=self.model)
            )
            if lit_res['status'] == 'success':
                lit_context = lit_res['content']
        
        # RAG for science plan
        print(f"\n--- Generating Experimental Strategy ---")
        res = perform_science_rag(
            objective=objective,
            instructions=HYPOTHESIS_GENERATION_INSTRUCTIONS,
            task_name="Experimental Plan",
            kb_docs=self.kb_docs,
            model=self.model,
            generation_config=self.generation_config,
            primary_data_set=primary_data_set,
            image_paths=all_image_paths,
            image_descriptions=image_descriptions,
            additional_context=ctx_string,
            external_context=lit_context
        )
        
        if lit_context:
            res["literature_search"] = lit_context
        
        # Snapshot 1: Science Draft
        res["iteration"] = current_iter
        res["stage"] = "Science Draft"
        self.state["plan_history"].append(res.copy())
        self.state["current_plan"] = res
        
        # Self-correction
        if not res.get("error"):
            is_relevant, critique = verify_plan_relevance(objective, res, self.model, self.generation_config)
            
            if not is_relevant:
                print(f"\n🔄 Self-correction triggered: {critique}")
                res = refine_plan_with_feedback(
                    original_result=res,
                    feedback=f"CRITICAL: {critique}",
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                
                res["iteration"] = current_iter
                res["stage"] = "Auto-Corrected"
                self.state["plan_history"].append(res.copy())
                self.state["current_plan"] = res
        
        # Human feedback on strategy
        if enable_human_feedback and res.get("proposed_experiments") and not res.get("error"):
            display_plan_summary(res)
            user_feedback = get_user_feedback()
            
            if user_feedback:
                print(f"\n📝 Refining plan...")
                self.state["human_feedback_history"].append({"phase": "science", "feedback": user_feedback})
                res = refine_plan_with_feedback(
                    original_result=res,
                    feedback=user_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                
                res["iteration"] = current_iter
                res["stage"] = "Human Refined (Science)"
                self.state["plan_history"].append(res.copy())
                self.state["current_plan"] = res
                
                display_plan_summary(res)
                print("✅ Plan updated.")
            else:
                print("✅ Plan accepted.")
        
        self.state["status"] = "planned"
        
        return res
    
    def generate_implementation_code(self,
                                    plan: Dict[str, Any],
                                    code_paths: List[str],
                                    enable_human_feedback: bool = True) -> Dict[str, Any]:
        """
        Add implementation code to an existing experimental plan.
        
        This method:
        1. Builds code knowledge base
        2. Performs code RAG to map experiments to APIs
        3. Provides human code review
        
        Args:
            plan: Existing plan dict (must have proposed_experiments)
            code_paths: Paths to code/API repositories
            enable_human_feedback: If True, pauses for code review
        
        Returns:
            Updated plan dict with implementation_code added to experiments
        """
        
        # Resolve code paths (handle Git URLs)
        print("\n--- Resolving Code Paths ---")
        effective_code_paths = []
        for path in code_paths:
            if path.strip().startswith(('http://', 'https://', 'git@')):
                print(f"  - 🔗 Cloning: {path}")
                local_path = clone_git_repository(path)
                if local_path:
                    effective_code_paths.append(local_path)
            else:
                effective_code_paths.append(path)
        
        # Build code KB
        if not self._ensure_kb_is_ready(knowledge_paths=None, code_paths=effective_code_paths):
            return {"error": "Code KB build failed"}
        
        # Check if code KB has content
        if not (self.kb_code.index and self.kb_code.index.ntotal > 0):
            print("  - ⚠️  Code KB is empty, skipping code generation")
            return plan
        
        # Generate code
        print(f"\n--- Generating Implementation Code ---")
        current_iter = plan.get("iteration", self.state.get("iteration_index", 1))
        
        res = perform_code_rag(
            result=plan,
            kb_code=self.kb_code,
            model=self.model,
            generation_config=self.generation_config
        )
        
        # Snapshot: Code Generated
        res["iteration"] = current_iter
        res["stage"] = "Code Generated"
        self.state["plan_history"].append(res.copy())
        self.state["current_plan"] = res
        
        # Human code review
        if enable_human_feedback:
            temp_dir = Path("./temp_code_review")
            print(f"\n--- Code Review ---")
            print(f"  - 💾 Saving to: {temp_dir}")
            
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            files = write_experiments_to_disk(res, str(temp_dir))
            
            if not files:
                print("  - ⚠️  No code generated")
            else:
                while True:
                    print("\n" + "="*60)
                    print(f"👀 CODE REVIEW REQUIRED")
                    print("="*60)
                    print(f"1. Review files in: {temp_dir.resolve()}")
                    print(f"2. Press ENTER to approve, or type feedback to refine")
                    print("-"*60)
                    
                    code_feedback = get_user_feedback()
                    
                    if not code_feedback:
                        print("✅ Code accepted")
                        break
                    
                    print(f"\n🛠️  Refining code...")
                    self.state["human_feedback_history"].append({"phase": "code", "feedback": code_feedback})
                    
                    res = refine_code_with_feedback(
                        result=res,
                        feedback=code_feedback,
                        model=self.model,
                        generation_config=self.generation_config
                    )
                    
                    res["iteration"] = current_iter
                    res["stage"] = "Code Refined"
                    self.state["plan_history"].append(res.copy())
                    self.state["current_plan"] = res
                    
                    print(f"  - 💾 Updating files...")
                    files = write_experiments_to_disk(res, str(temp_dir))
        
        return res

    def propose_experiments(self, objective: str, 
                            knowledge_paths: Optional[List[str]] = None, 
                            code_paths: Optional[List[str]] = None,
                            additional_context: Optional[Dict[str, str]] = None,
                            primary_data_set: Optional[Union[str, Dict[str, str]]] = None,
                            image_paths: Optional[List[str]] = None,
                            image_descriptions: Optional[List[str]] = None,
                            output_json_path: Optional[str] = None,
                            enable_human_feedback: bool = True,
                            reset_state: bool = False) -> Dict[str, Any]: # Default False to enable cumulative workflows
        """
        Generate an experimental plan based on scientific literature and implementation knowledge.

        This is the primary entry point for starting a new research workflow. The agent:
        1. Builds/loads dual knowledge bases (scientific docs + implementation code)
        2. Optionally queries external literature databases
        3. Generates experimental hypotheses via RAG
        4. Maps experimental steps to executable code
        5. Provides human-in-the-loop review at both science and code stages

        Args:
            objective (str): High-level research goal. This guides all hypothesis generation
                and plan refinement. Should be specific and measurable.
                Examples:
                    - "Optimize the yield of the Suzuki coupling reaction"
                    - "Screen 96 conditions to selectively precipitate magnesium"
                    - "Develop a high-throughput assay for enzyme activity"
            
            knowledge_paths (Optional[List[str]]): Paths to scientific documents/data.
                Supported formats: PDFs, .txt, .md, .xlsx, .csv, directories.
                You can pass Excel/CSV files directly here. If a .json file 
                with the same name exists next to the data file, it is automatically 
                loaded as metadata.
                These populate the Docs Knowledge Base for hypothesis generation.
                Example: ["./papers/", "./lab_notebooks/protocol.pdf", "./public_data.xlsx", "./public_data.json" ]
            
            code_paths (Optional[List[str]]): Paths to code repositories or API documentation.
                Supported formats: Local directories, Git URLs, Python files
                These populate the Code Knowledge Base for implementation.
                Examples:
                    - ["./opentrons_api/"]  # Local repo
                    - ["https://github.com/org/automation-lib.git"]  # Git URL
            
            additional_context (Optional[Dict[str, str]]): Additional text context
                to inject into the prompt. Keys become section headers.
                Example: {
                    "Safety Constraints": "Maximum temperature is 80°C",
                    "Equipment Available": "Opentrons OT-2, plate reader"
                }
            
            primary_data_set (Optional[Dict[str, str]]): Main dataset to analyze.
                Use for the dataset that drives the research objective.
                Example: {"file_path": "./screening_results.xlsx"}
            
            image_paths (Optional[List[str]]): Paths to images (plots, diagrams, photos).
                Supported formats: .png, .jpg, .jpeg, .tiff, .bmp
                These are passed to the vision model for multimodal analysis.
                Examples: ["./criticality_matrix.png", "./reaction_scheme.jpg"]
            
            image_descriptions (Optional[List[str]]): Text descriptions for each image.
                Should be in same order as image_paths. Helps LLM interpret images.
                Examples: ["Criticality matrix showing material supply risks"]
            
            output_json_path (Optional[str]): Path to save the generated plan.
                Also saves full state to {output_json_path}.state.json
                and generates HTML report at {output_json_path}.html
                Example: "./outputs/experiment_plan.json"
            
            enable_human_feedback (bool): If True, pauses for user input at:
                - Strategy review (after hypothesis generation)
                - Code review (after script generation)
                Set to False for fully autonomous operation.
                Defaults to True.
            
            reset_state (bool): If True, clears any existing state and starts fresh.
                If False, appends to existing research session (cumulative workflow).
                Defaults to False.
        
        Returns:
            Dict[str, Any]: Complete agent state containing:
                - session_id: Unique identifier for this session
                - objective: The research objective
                - iteration_index: Current iteration number (1 for initial plan)
                - current_plan: The active experimental plan, structure
        """
        # Phase 1: Generate experimental plan (science only)
        plan = self.generate_plan(
            objective=objective,
            knowledge_paths=knowledge_paths,
            primary_data_set=primary_data_set,
            additional_context=additional_context,
            image_paths=image_paths,
            image_descriptions=image_descriptions,
            enable_human_feedback=enable_human_feedback,
            reset_state=reset_state
        )
        
        if plan.get("error"):
            if output_json_path:
                self._save_results_to_json(plan, output_json_path)
            return self.state
        
        # Phase 2: Add implementation code (if code_paths provided)
        if code_paths:
            plan = self.generate_implementation_code(
                plan=plan,
                code_paths=code_paths,
                enable_human_feedback=enable_human_feedback
            )
        
        # Save final results
        if output_json_path:
            self._save_results_to_json(plan, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            self._generate_html_report(output_json_path)
        
        # Save scripts
        final_out = "./output_scripts"
        print(f"\n--- Saving Scripts to: {final_out} ---")
        write_experiments_to_disk(plan, final_out)
        
        return self.state

    def update_plan_with_results(self,
                                 results: Any,
                                 output_json_path: Optional[str] = None,
                                 enable_human_feedback: bool = True,
                                 state_file_path: Optional[str] = None 
                                 ) -> Dict[str, Any]:
        """
        Iterates on the current experimental plan based on new experimental results, 
        observations, or data files.

        This method acts as the "feedback loop" of the agent, transforming the system from 
        a linear planner into an iterative scientific partner. It performs Smart Result Parsing, 
        Result-Aware RAG, and Human-in-the-Loop refinement.

        **Capabilities & Workflow:**

        1.  **Smart Result Parsing (Multimodal):**
            -   Detects and parses input types automatically.
            -   **Text/Dicts/Lists:** Converted to JSON strings for the LLM prompt.
            -   **Data Files (.xlsx, .csv):** Automatically summarized using `excel_parser` and injected as text context.
            -   **Images (.png, .jpg):** Loaded and passed to the vision model for visual analysis (e.g., plot trends, failures).
            -   **Logs (.txt, .log):** Read and injected as context.

        2.  **Result-Aware RAG (Retrieval Augmented Generation):**
            -   Uses the content of the results to perform a *new* targeted search in the Docs Knowledge Base (`kb_docs`).
            -   Example: If results mention "precipitation," it retrieves papers discussing solubility limits, even if those papers weren't relevant to the initial plan.

        3.  **Nuanced Scientific Reasoning:**
            -   Prompts the LLM to categorize the outcome into one of five strategic buckets:
                * **CONFIRMED:** Validated hypothesis -> Propose next step.
                * **OPTIMIZATION NEEDED:** Valid sub-optimal result -> Tune parameters (Do not change hypothesis).
                * **INCONCLUSIVE:** Noisy data -> Refine measurement technique.
                * **OPERATIONAL FAILURE:** Code/Equipment error -> Fix implementation (Do not change science).
                * **SCIENTIFIC FAILURE:** Disproven hypothesis -> Pivot to new approach.

        4.  **Human-in-the-Loop (Dual-Phase):**
            -   **Phase A (Strategy):** Pauses after generating the new scientific plan to allow user critique (e.g., "Don't increase temp, safety limit is 50C").
            -   **Phase B (Code):** Pauses after generating the Python scripts. Writes them to a temp folder (`./temp_code_review_iter`) for inspection before finalization.

        Args:
            results (Any): The outcome of the previous experiment. 
                Supported formats:
                -   **String:** Natural language description (e.g., "Yield was 5%").
                -   **Dict/List:** Structured data (e.g., `{"yield": 0.05, "error": None}`).
                -   **File Path (str):** Path to a local file (.xlsx, .csv, .txt, .png, .jpg).
                -   **Structured List:** A list containing a mix of the above, or dictionaries with metadata 
                    (e.g., `[{"path": "./plot.png", "description": "Graph showing thermal runaway"}]`).
            output_json_path (Optional[str]): If provided, saves the updated plan JSON to this path.
                The full state is also saved to `{output_json_path}.state.json`.
            enable_human_feedback (bool): If True, pauses execution for console input at the 
                Strategy and Code review stages. Defaults to True.
            state_file_path: Optional path to .state.json file.
                If provided, restores agent state before processing results.
                Equivalent to calling restore_state() first.

        Returns:
            Dict[str, Any]: Updated state dictionary containing:
                - current_plan: Latest experimental plan
                - plan_history: All historical plans
                - experimental_results: All results received
                - iteration_index: Current iteration number
        """

        # --- 0. STATE RESTORATION ---

        if state_file_path is not None:
            print(f"\n--- 🔄 Restoring State from File ---")
            self.restore_state(state_file_path)

        if not self.state or not self.state.get("current_plan"):
            raise ValueError(
                "No active state found.\n"
                "You must initialize the agent first using one of:\n"
                "  1. agent.propose_experiments(...) - Start new session\n"
                "  2. agent.restore_state('path.state.json') - Restore saved session\n"
                "  3. Pass state_file_path='path.state.json' to this method"
            )
        
        print(f"\n--- 🔄 Iterating Plan based on New Results ---")
        executed_plan_idx = self.state["iteration_index"]
        
        # Extract from state
        objective = self.state["objective"]
        current_plan = self.state["current_plan"]
        
        # --- 1. SMART RESULT PARSING ---
        parsed_text_results = []
        loaded_images = []
        
        # Helper to process a single item (path or text)
        def process_item(item: Any, description: str = "") -> str:
            text_output = ""
            
            # If it's a file path
            if isinstance(item, str) and (Path(item).exists()):
                path = Path(item)
                suffix = path.suffix.lower()
                
                # A. Data Files
                if suffix in ['.xlsx', '.xls', '.csv']:
                    print(f"  - 📄 Parsing data file: {path.name}")
                    try:
                        chunks = parse_adaptive_excel(str(path), context_path="")
                        if chunks:
                            summary = chunks[0]['text']
                            text_output = f"DATA FILE ({path.name}):\n{summary}"
                    except Exception as e:
                        text_output = f"[Error parsing {path.name}: {e}]"

                # B. Images
                elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    print(f"  - 🖼️  Loading result image: {path.name}")
                    try:
                        with PIL_Image.open(path) as img:
                            img.load()  
                            loaded_images.append(img.copy())
                        text_output = f"[Attached Image: {path.name}]"
                    except Exception as e:
                        text_output = f"[Error loading image {path.name}: {e}]"
                
                # C. Logs/Text
                elif suffix in ['.txt', '.log', '.md', '.json']:
                    try:
                        content = path.read_text(encoding='utf-8')
                        text_output = f"LOG FILE ({path.name}):\n{content}"
                    except Exception as e:
                        text_output = f"[Error reading log {path.name}: {e}]"
                
                else:
                    text_output = f"FILE ({path.name})"

            # If not a file, treat as raw text/data
            else:
                if isinstance(item, (dict, list)):
                    text_output = json.dumps(item, indent=2)
                else:
                    text_output = str(item)
            
            # Append description if provided
            if description:
                text_output += f"\n(Context: {description})"
            
            return text_output

        # Recursive Parser to handle Lists and Dictionaries
        items_to_process = results if isinstance(results, list) else [results]
        
        for entry in items_to_process:
            if isinstance(entry, dict):
                # Check for common keys indicating a file + desc structure
                path_val = entry.get('path') or entry.get('file') or entry.get('image')
                desc_val = entry.get('description') or entry.get('desc') or entry.get('caption') or entry.get('notes')
                
                if path_val and isinstance(path_val, str):
                    # It's a structured file entry
                    parsed_text_results.append(process_item(path_val, desc_val if desc_val else ""))
                else:
                    # It's just a data dictionary
                    parsed_text_results.append(json.dumps(entry, indent=2))
            else:
                # It's a direct item (string, number, or path string)
                parsed_text_results.append(process_item(entry))

        # Join all text findings
        consolidated_feedback = "\n\n".join(parsed_text_results)

        # Update State History
        self.state["experimental_results"].append({
            "iteration": executed_plan_idx,
            "timestamp": datetime.now().isoformat(),
            "data_summary": str(results) # Keep reference to raw input
        })
        self.state["iteration_index"] += 1 
        next_plan_idx = self.state["iteration_index"]
        
        # --- 2. Construct Feedback Prompt ---
        feedback_prompt = (
            f"We executed the previous plan. Here are the experimental results:\n"
            f"{consolidated_feedback}\n\n"
            f"**TASK:** Analyze these results (including any attached plots) to Refine or Update the plan.\n"
            f"Select the most appropriate strategy:\n"
            f"1. **CONFIRMED:** If hypothesis is validated, propose next step.\n"
            f"2. **OPTIMIZATION NEEDED:** If result is valid but sub-optimal, tune parameters.\n"
            f"3. **INCONCLUSIVE:** If data is noisy, propose refined experiment.\n"
            f"4. **OPERATIONAL FAILURE:** If failure was code/equipment, propose fix.\n"
            f"5. **SCIENTIFIC FAILURE:** If hypothesis is disproven, propose new approach.\n"
        )
        
        # --- 3. RESULT-AWARE RAG ---
        new_literature_context = None
        if self.kb_docs.index and self.kb_docs.index.ntotal > 0:
            search_query = f"Implications and causes of: {consolidated_feedback[:400]}"
            print(f"  - 🔍 Searching literature for context on results...")
            hits = self.kb_docs.retrieve(search_query, top_k=3)
            if hits:
                new_literature_context = "\n---\n".join([c['text'] for c in hits])
                print(f"    -> Found {len(hits)} relevant document chunks.")
        
        # --- 4. Generate Refined Plan ---
        print(f"  - Reasoning over results with literature context...")
        objective = self.state["objective"]
        current_plan = self.state["current_plan"]
        
        new_plan = refine_plan_with_feedback(
            original_result=current_plan,
            feedback=feedback_prompt,
            objective=objective,
            model=self.model,
            generation_config=self.generation_config,
            new_context=new_literature_context,
            result_images=loaded_images
        )
        
        # SNAPSHOT: REASONING DRAFT
        new_plan["iteration"] = next_plan_idx
        new_plan["stage"] = "Reasoning Draft"
        self.state["plan_history"].append(new_plan.copy())
        self.state["current_plan"] = new_plan

        # =====================================================
        # 5. HUMAN STRATEGY FEEDBACK
        # =====================================================
        if enable_human_feedback and not new_plan.get("error"):
            print("\n" + "="*60)
            print("🧠 AGENT'S PROPOSED REVISION BASED ON RESULTS")
            print("="*60)
            display_plan_summary(new_plan)
            
            user_feedback = get_user_feedback()
            
            if user_feedback:
                print(f"\n📝 Feedback received. Adjusting strategy...")
                self.state["human_feedback_history"].append({"phase": "science_iteration", "feedback": user_feedback})
                new_plan = refine_plan_with_feedback(
                    original_result=new_plan,
                    feedback=user_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                # SNAPSHOT: HUMAN REFINED
                new_plan["iteration"] = next_plan_idx
                new_plan["stage"] = "Human Refined (Science)"
                self.state["plan_history"].append(new_plan.copy())
                self.state["current_plan"] = new_plan
                print("✅ Strategic revision updated.")

        # =====================================================
        # 6. Generate Code
        # =====================================================
        if self.kb_code.index and self.kb_code.index.ntotal > 0 and not new_plan.get("error"):
             
            # Extract previous implementations
            previous_implementations = []
            if current_plan and "proposed_experiments" in current_plan:                
                for exp in current_plan["proposed_experiments"]:
                    if "implementation_code" in exp:
                        previous_implementations.append({
                            'experiment_name': exp.get('experiment_name', 'Unnamed'),
                            'code': exp['implementation_code'],
                            'iteration': executed_plan_idx,
                            'source_files': exp.get('code_source_files', []),
                            'previous_steps': exp.get('experimental_steps', [])
                        })
            
            print(f"\n--- Code Implementation Analysis ---")
            if previous_implementations:
                print(f"  - Context: {len(previous_implementations)} existing implementation(s)")
            else:
                print(f"  - Context: Writing from scratch (no previous code)")
            
            new_plan = perform_code_rag(
                 result=new_plan,
                 kb_code=self.kb_code,
                 model=self.model,
                 generation_config=self.generation_config,
                 previous_implementations=previous_implementations
             )
            
             # SNAPSHOT: CODE GENERATED
            new_plan["iteration"] = next_plan_idx
            new_plan["stage"] = "Code Generated"
            self.state["plan_history"].append(new_plan.copy())
            self.state["current_plan"] = new_plan

        # =====================================================
        # 7. HUMAN CODE REVIEW
        # =====================================================
        if enable_human_feedback and not new_plan.get("error"):
            temp_dir = Path("./temp_code_review_iter")
            print(f"\n--- Human Code Review (Iteration {next_plan_idx}) ---")
            
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            files = write_experiments_to_disk(new_plan, str(temp_dir))
            
            if files:
                while True:
                    print("\n" + "="*60)
                    print(f"👀 ACTION REQUIRED: Code Review")
                    print("="*60)
                    print(f"1. Open folder: {temp_dir.resolve()}")
                    print(f"2. Inspect the {len(files)} new Python file(s).")
                    print("3. Return here to Approve or Request Changes.")
                    
                    code_feedback = get_user_feedback()
                    
                    if not code_feedback:
                        print("✅ Code accepted.")
                        break
                    
                    self.state["human_feedback_history"].append({"phase": "code_iteration", "feedback": code_feedback})
                    print(f"\n🛠️  Refining code based on: '{code_feedback}'...")
                    
                    new_plan = refine_code_with_feedback(
                        result=new_plan,
                        feedback=code_feedback,
                        model=self.model,
                        generation_config=self.generation_config
                    )
                    
                    # SNAPSHOT: CODE REFINED
                    new_plan["iteration"] = next_plan_idx
                    new_plan["stage"] = "Code Refined"
                    self.state["plan_history"].append(new_plan.copy())
                    self.state["current_plan"] = new_plan
                    
                    print(f"  - 💾 Overwriting files in {temp_dir} with refined code...")
                    files = write_experiments_to_disk(new_plan, str(temp_dir))

        # 8. Commit to State & Save
        self.state["current_plan"] = new_plan
        # (Already appended snapshots above, so no final append needed unless we want a 'Final' tag)
        self.state["status"] = "iterated"
        
        final_out = "./output_scripts"
        print(f"\n--- Saving Final Scripts to: {final_out} ---")
        write_experiments_to_disk(new_plan, final_out)
        
        if output_json_path:
            self._save_results_to_json(new_plan, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            
            # TRIGGER HTML REPORT GENERATION
            self._generate_html_report(output_json_path)
            
        return self.state
    
    def _generate_html_report(self, json_path: str):
        """Helper to generate HTML report alongside JSON."""
        if not json_path: return
        html_path = str(Path(json_path).with_suffix('.html'))
        try:
            generator = HTMLReportGenerator(self.state)
            generator.generate(html_path)
        except Exception as e:
            print(f"⚠️ Failed to generate HTML report: {e}")

    def perform_technoeconomic_analysis(self, objective: str,
                                        knowledge_paths: Optional[List[str]] = None,
                                        primary_data_set: Optional[Union[str, Dict[str, str]]] = None,
                                        image_paths: Optional[List[str]] = None,
                                        image_descriptions: Optional[List[str]] = None,
                                        output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs TEA using Dual-KB retrieval. 

        **Workflow:**
        
        1. Knowledge Base Construction (if needed)
        2. External Literature Search (optional, via FutureHouse)
        3. RAG-based Economic Analysis
        4. State Initialization (if starting fresh with TEA)
        5. Report Generation (JSON + HTML)

        **Integration with Planning:**
    
        TEA results are stored in the agent's state and can inform subsequent
        experimental planning:
            >>> # Perform TEA first
            >>> tea_results = agent.perform_technoeconomic_analysis(
            ...     objective="Recover lithium from brine",
            ...     knowledge_paths=["./market_data/", "./reports/"],
            ... )
            >>> 
            >>> # Use TEA insights in experimental planning
            >>> plan = agent.propose_experiments(
            ...             objective="Develop lithium extraction process",
            ...             knowledge_paths=["./extraction_methods/"],
            ...             additional_context=tea_results,
            ...             primary_data_set={
            ...                "file_path": "./brine_composition.xlsx",
            ...                "metadata_path": ./metadata.json}
            ... )
        Args:
        objective (str): Research objective to evaluate economically.
            Should describe the material, process, or technology to assess.
            Examples:
                - "Recover rare earth elements from coal ash"
                - "Evaluate magnesium extraction from produced water"
                - "Assess economic viability of direct air capture"
        
        knowledge_paths (Optional[List[str]]): Paths to documents for TEA context.
            Should include market data, pricing reports, criticality assessments,
            existing TEA studies, and process descriptions. Supports both PDF/TXT and Excel/CSV.
            Examples: ["./market_reports/", "./critical_materials_report.pdf", "./public_data.xlsx", "./public_data.json"]
        
        primary_data_set (Optional[Dict[str, str]]): Main dataset for analysis.
            Can contain composition, concentration, or yield data.
            Example: {"file_path": "./feedstock_composition.xlsx"}
        
        image_paths (Optional[List[str]]): Images to support TEA analysis.
            Examples: criticality matrices, supply chain diagrams, cost breakdowns.
        
        image_descriptions (Optional[List[str]]): Descriptions for each image.
            Example: ["Criticality matrix showing supply risk vs. importance"]
        
        output_json_path (Optional[str]): Path to save TEA results.
            Saves to {output_json_path} (results only)
            Saves to {output_json_path}.state.json (full state)
            Generates {output_json_path}.html (formatted report)
    
    Returns:
        Dict[str, Any]: Technoeconomic analysis results  

    Example - Basic Usage:
        >>> agent = PlanningAgent()
        >>> state = agent.propose_experiments(
        ...     objective="Optimize enzyme kinetics",
        ...     knowledge_paths=["./enzyme_papers/"],
        ...     code_paths=["./plate_reader_api/"],
        ...     output_json_path="./plan.json"
        ... )
        >>> # User reviews in console, provides feedback or approves
        >>> # Final scripts saved to ./output_scripts/

    Example - Advanced with Data:
        >>> state = agent.propose_experiments(
        ...     objective="Identify optimal precipitation conditions",
        ...     knowledge_paths=["./papers/", "./protocols.pdf"],
        ...     code_paths=["https://github.com/opentrons/opentrons"],
        ...     primary_data_set={
        ...         "file_path": "./icpms_results.xlsx",
        ...         "metadata_path": "./icpms_metadata.json"
        ...     },
        ...     image_paths=["./criticality_matrix.jpg"],
        ...     image_descriptions=["Material criticality assessment"],
        ...     additional_context={
        ...         "Constraints": "Use only commodity chemicals",
        ...         "Equipment": "Opentrons OT-2, 96-well plates, ICP-MS"
        ...     },
        ...     output_json_path="./precipitation_plan.json",
        ...     enable_human_feedback=True
        ... )
    """
        
        # 0a. Resolve Primary Data
        primary_data_set = resolve_primary_data_path(primary_data_set)
        # 0b. Resolve image paths
        # Images explicitly specified by user undr image_paths (will be deprecated in the future)
        manual_images = image_paths or []
        # Find new images under the provided knowledge paths but exclude any that are already in manual_images
        auto_images = [img for img in extract_images(knowledge_paths) if img not in manual_images]
        # Append auto-images to the end so manual descriptions stay aligned with manual images
        all_image_paths = manual_images + auto_images

        # 1. State Initialization (if starting fresh with TEA)
        if not self.state:
            self.state = self._initialize_state(
                objective=objective,
                knowledge_paths=knowledge_paths,
                code_paths=None,
                primary_data_set=primary_data_set,
                image_paths=all_image_paths,
                image_descriptions=image_descriptions
            )

        #  TEA is always step 0 (pre-planning)
        self.state["iteration_index"] = 0

        # 2. Build KB if needed
        if not self._ensure_kb_is_ready(knowledge_paths, code_paths=None):
            return {"error": "KB Init Failed"}
        
        # 3. Literature Search
        lit_context = ""
        if self.lit_agent:
            print(f"  - 🌍 Querying literature for TEA context...")
            lit_res = self.lit_agent.search_for_economic_data(
                optimize_search_query(objective=objective, model=self.model)
            )
            if lit_res['status'] == 'success':
                lit_context = lit_res['content']

        # 4. Perform RAG
        res = perform_science_rag(
            objective=objective, 
            instructions=TEA_INSTRUCTIONS, 
            task_name="Technoeconomic Analysis",
            kb_docs=self.kb_docs,
            model=self.model,
            generation_config=self.generation_config,
            primary_data_set=primary_data_set, 
            image_paths=all_image_paths, 
            image_descriptions=image_descriptions,
            external_context=lit_context
        )

        if lit_context:
            res["literature_search"] = lit_context

        # 5. Commit to State
        if not res.get("error"):
            # Tags for the HTML Generator
            res["type"] = "technoeconomic_analysis"
            res["stage"] = "TEA Initial"
            res["iteration"] = 0 # TEA is step 0 (pre-planning)
            
            # Append copy to history (Full Traceability)
            self.state["plan_history"].append(res.copy())
            
            # Update Active Pointer
            self.state["current_plan"] = res

        # 6. Save & Generate Report
        if output_json_path:
            self._save_results_to_json(res, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            
            # Trigger HTML Generation (will show TEA card)
            self._generate_html_report(output_json_path)

        return res