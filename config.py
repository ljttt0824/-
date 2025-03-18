"""
Configuration settings for the Computer Network Course Knowledge Graph Construction project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SLIDES_DIR = os.path.join(DATA_DIR, "slides")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SLIDES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# OCR settings
OCR_DPI = 300
OCR_LANG = "eng"

# LLM settings
DEEPSEEK_MODEL = "deepseek-ai/deepseek-r1-chat"
OPENAI_MODEL = "gpt-4"

# Entity types for computer network courses
ENTITY_TYPES = [
    "concept",
    "protocol",
    "physical_media",
    "technology",
    "other"
]

# Relationship types
RELATIONSHIP_TYPES = [
    "is_a",           # Taxonomy relationship
    "part_of",        # Composition relationship
    "depends_on",     # Prerequisite relationship
    "equivalent_to",  # Equivalence relationship
    "other"           # Other relationships
]

# Few-shot examples settings
NUM_FEW_SHOT_EXAMPLES = 3

# Extraction settings
MAX_TOKENS = 4096
TEMPERATURE = 0.1
TOP_P = 0.9

# Canonicalization settings
SIMILARITY_THRESHOLD = 0.85

# Quality assessment metrics
QUALITY_METRICS = [
    "entity_coverage",
    "relationship_coverage",
    "entity_accuracy",
    "relationship_accuracy",
    "graph_connectivity",
    "graph_density"
]