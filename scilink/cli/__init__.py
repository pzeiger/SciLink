"""SciLink CLI interface."""

# Main CLI entry point
from .main import main

# Individual command modules
from . import plan
from . import simulate
from . import analyze

from .workflows import main as experimental_novelty_main
from .agents import add_agent_args

__all__ = ['main', 'plan', 'simulate', 'analyze', 'experimental_novelty_main', 'add_agent_args']
