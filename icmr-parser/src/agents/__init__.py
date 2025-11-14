"""
Multi-agent reasoning system for prescription validation.

This package contains the Generator and Verifier agents used for
creating high-quality reasoning traces for LLM fine-tuning.
"""

from .generator_agent import GeneratorAgent
from .verifier_agent import VerifierAgent

__all__ = ['GeneratorAgent', 'VerifierAgent']

