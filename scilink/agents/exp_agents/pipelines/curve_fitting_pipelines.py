# pipelines/curve_fitting_pipelines.py

"""
Pipeline factory for curve fitting analysis.
"""

import logging
from typing import Callable, List, Any

from ..controllers.curve_fitting_controllers import (
    RunCurvePreprocessingController,
    AnalyzeDataController,
    PlanAnalysisController,
    LiteratureSearchController,
    ExecuteFittingController,
    BuildInterpretationPromptController,
    GenerateCurveFittingReportController
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController,
)
from ..instruct import (
    CURVE_ANALYSIS_INSTRUCTIONS,
    FITTING_SCRIPT_INSTRUCTIONS,
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
    FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
    FITTING_INTERPRETATION_INSTRUCTIONS,
)


def create_curve_fitting_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    parse_fn: Callable,
    store_fn: Callable,
    plot_fn: Callable,
    executor: Any,
    output_dir: str,
    preprocessor: Any | None = None,
    literature_agent: Any | None = None,
    enable_human_feedback=False, 
    settings: dict | None = None,  # Deprecated
) -> List:
    """
    Create curve fitting pipeline.

    Args:
        model: LLM model
        logger: Logger
        generation_config: LLM config
        safety_settings: LLM safety settings
        parse_fn: JSON response parser
        store_fn: Image storage function
        plot_fn: Curve plotting function
        executor: Script executor
        output_dir: Output directory
        preprocessor: Optional preprocessor agent
        literature_agent: Optional literature agent (None = disabled)
        enable_human_feedback: Enable human feedback on the proposed fitting approach

    Returns:
        List of pipeline controllers
    """
    pipeline = []

    # 1. Optional preprocessing
    if preprocessor is not None:
        pipeline.append(RunCurvePreprocessingController(logger, preprocessor, output_dir))

    # 2. Analyze data
    pipeline.append(AnalyzeDataController(logger, plot_fn))

    # 3. Plan approach (LLM)
    pipeline.append(
        PlanAnalysisController(
            model, logger, generation_config, safety_settings, parse_fn,
            CURVE_ANALYSIS_INSTRUCTIONS, enable_human_feedback=enable_human_feedback
        )
    )

    # 4. Literature search (runs if agent provided and LLM suggests a query)
    pipeline.append(LiteratureSearchController(logger, literature_agent, output_dir))

    # 5. Execute fitting
    pipeline.append(
        ExecuteFittingController(
            model,
            logger,
            generation_config,
            safety_settings,
            parse_fn,
            executor,
            FITTING_SCRIPT_INSTRUCTIONS,
            FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
            FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
            output_dir,
        )
    )

    # 6. Build interpretation prompt
    pipeline.append(BuildInterpretationPromptController(logger, FITTING_INTERPRETATION_INSTRUCTIONS))

    # 7. Final interpretation (LLM)
    pipeline.append(RunFinalInterpretationController(model, logger, generation_config, safety_settings, parse_fn))

    # 8. Store results
    pipeline.append(StoreAnalysisResultsController(logger, store_fn))

    # 9. Generate HTML report
    pipeline.append(GenerateCurveFittingReportController(logger, output_dir))

    logger.info(f"Pipeline created: {len(pipeline)} steps")
    return pipeline