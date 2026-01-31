"""
SAM Analysis Pipelines - Unified Architecture

Factory functions for creating SAM analysis pipelines.
All analysis now uses a single unified pipeline that handles both
single images (n=1) and batches (n>1) identically.
"""

import logging
from typing import Callable, List

from ..controllers.sam_microscopy_controllers import (
    # Unified pipeline controllers
    HumanFeedbackRefinementController,
    UnifiedBatchProcessingController,
    ConditionalCustomAnalysisController,
    UnifiedSynthesisController,
    UnifiedReportGenerationController,
)


def create_unified_sam_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    Factory function to create the unified SAM analysis pipeline.
    
    This pipeline handles BOTH single images and batches:
    
    1. Human Feedback Refinement (optional)
       - Refines SAM parameters on the first image
       - Skipped if enable_human_feedback=False
       
    2. Batch Processing
       - Processes ALL images (including single images as n=1)
       - Caches SAM model for efficiency
       
    3. Conditional Custom Analysis
       - For n>=2: Generates and executes trend analysis script
       - For n=1: Skipped (no trends to analyze)
       
    4. Synthesis
       - For n>=2: Cross-image synthesis of findings
       - For n=1: Single-image scientific interpretation
       
    5. Report Generation
       - Generates HTML report and JSON summary
       - Adapts format based on single vs batch
    
    Args:
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation configuration
        safety_settings: LLM safety settings
        settings: Pipeline settings dict
        parse_fn: Function to parse LLM responses
        store_fn: Function to store analysis images
    
    Returns:
        List of controller instances to execute in sequence
    """
    pipeline = [
        # Step 1: Human feedback refinement on first image
        HumanFeedbackRefinementController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        ),
        
        # Step 2: Process all images with refined parameters
        UnifiedBatchProcessingController(
            logger=logger,
            settings=settings
        ),
        
        # Step 3: Custom analysis script (conditional on n>=2)
        ConditionalCustomAnalysisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        ),
        
        # Step 4: Scientific synthesis
        UnifiedSynthesisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings,
            store_fn=store_fn
        ),
        
        # Step 5: Report generation
        UnifiedReportGenerationController(
            logger=logger,
            settings=settings
        )
    ]
    
    return pipeline


# =============================================================================
# LEGACY PIPELINE FACTORIES (for backward compatibility if needed)
# =============================================================================

def create_sam_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    DEPRECATED: Use create_unified_sam_pipeline instead.
    
    This factory is preserved for backward compatibility but now
    returns the unified pipeline.
    """
    logger.warning(
        "create_sam_pipeline() is deprecated. "
        "Use create_unified_sam_pipeline() instead."
    )
    return create_unified_sam_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        settings=settings,
        parse_fn=parse_fn,
        store_fn=store_fn
    )


def create_sam_batch_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable
) -> List:
    """
    DEPRECATED: Use create_unified_sam_pipeline instead.
    
    This factory is preserved for backward compatibility but now
    returns the unified pipeline (without store_fn).
    """
    logger.warning(
        "create_sam_batch_pipeline() is deprecated. "
        "Use create_unified_sam_pipeline() instead."
    )
    return create_unified_sam_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        settings=settings,
        parse_fn=parse_fn,
        store_fn=lambda *args, **kwargs: None  # No-op store function
    )