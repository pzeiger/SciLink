"""
Microscopy Analysis Pipeline Factories.

Updated to include LLM-based ReportGenerationController.
"""

from ..controllers.microscopy_controllers import (
    GetFFTParamsController,
    RunFFTNMFController,
    RunGlobalFFTController,
    BuildFFTNMFPromptController,
    FinalLLMAnalysisController,
    SeriesLoaderController,
    FirstFrameAnalysisController,
    UserFeedbackController,
    SeriesBatchController,
    SummaryScriptController,
    ReportGenerationController  # Now requires model + LLM params
)


def create_fftnmf_pipeline(model, logger, generation_config, safety_settings, settings, parse_fn, store_fn):
    """Create single-image FFT/NMF pipeline."""
    return [
        GetFFTParamsController(model, logger, generation_config, safety_settings),
        RunGlobalFFTController(logger, settings),
        RunFFTNMFController(logger, settings),
        BuildFFTNMFPromptController(logger),
        FinalLLMAnalysisController(model, logger, generation_config, safety_settings, parse_fn, store_fn),
    ]


def create_series_pipeline(model, logger, generation_config, safety_settings, settings, 
                           parse_fn, feedback_callback=None):
    """
    Create series analysis pipeline with feedback, script generation, and LLM-analyzed HTML report.
    
    Updated: ReportGenerationController now takes model and LLM parameters for 
    scientific interpretation of results.
    """
    return [
        SeriesLoaderController(logger),
        FirstFrameAnalysisController(model, logger, generation_config, safety_settings, settings),
        UserFeedbackController(logger, settings, feedback_callback),
        SeriesBatchController(logger, settings),
        SummaryScriptController(model, logger, generation_config, safety_settings, parse_fn, settings),
        # Updated: Now passes model for LLM-based analysis
        ReportGenerationController(model, logger, generation_config, safety_settings, parse_fn, settings),
    ]


def create_batch_only_pipeline(model, logger, generation_config, safety_settings, settings, parse_fn, locked_params):
    """Create batch-only pipeline (skip first-frame analysis)."""
    
    class PresetParamsController:
        """Injects preset parameters without LLM estimation."""
        def __init__(self, params):
            self.params = params
        def execute(self, state):
            state["locked_params"] = self.params
            state["first_frame_results"] = {"llm_params": self.params}
            return state
    
    return [
        SeriesLoaderController(logger),
        PresetParamsController(locked_params),
        SeriesBatchController(logger, settings),
        SummaryScriptController(model, logger, generation_config, safety_settings, parse_fn, settings),
        # Updated: Now passes model for LLM-based analysis
        ReportGenerationController(model, logger, generation_config, safety_settings, parse_fn, settings),
    ]


def create_quick_series_pipeline(model, logger, generation_config, safety_settings, settings, parse_fn):
    """
    Create minimal series pipeline - no feedback loop, no script generation.
    
    Useful for batch processing where you just want results + report.
    """
    return [
        SeriesLoaderController(logger),
        FirstFrameAnalysisController(model, logger, generation_config, safety_settings, settings),
        # Skip UserFeedbackController - auto-accept LLM params
        _AutoAcceptController(logger),
        SeriesBatchController(logger, settings),
        # Skip SummaryScriptController - no custom script
        ReportGenerationController(model, logger, generation_config, safety_settings, parse_fn, settings),
    ]


class _AutoAcceptController:
    """Auto-accepts LLM parameters without user feedback."""
    def __init__(self, logger):
        self.logger = logger
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        llm_params = state.get("first_frame_results", {}).get("llm_params", {})
        state["locked_params"] = llm_params
        self.logger.info(f"✅ Auto-accepted LLM parameters: {llm_params}")
        return state