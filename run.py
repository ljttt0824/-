#!/usr/bin/env python
"""
Main script for running the Computer Network Course Knowledge Graph Construction pipeline.
"""

import os
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm

# Import modules
from ocr.ocr_processor import OCRProcessor
from preprocessing.text_preprocessor import TextPreprocessor
from extraction.entity_extractor import EntityExtractor
from definition.entity_definer import EntityDefiner
from canonicalization.entity_canonicalizer import EntityCanonicalizer
from exploration.type_explorer import TypeExplorer
from matching.entity_matcher import EntityMatcher
from construction.kg_builder import KnowledgeGraphBuilder
from quality.quality_assessor import QualityAssessor

# Import configuration
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_construction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Computer Network Course Knowledge Graph Construction")
    
    parser.add_argument("--input_dir", type=str, default=config.SLIDES_DIR,
                        help="Directory containing PDF slides")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="Directory to save output files")
    
    # LLM model settings
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="LLM model to use for extraction and processing")
    parser.add_argument("--openai_api_key", type=str, default=config.OPENAI_API_KEY,
                        help="OpenAI API key")
    parser.add_argument("--openai_api_base", type=str, default=config.OPENAI_API_BASE,
                        help="OpenAI API base URL")
    
    # OCR settings
    parser.add_argument("--ocr_dpi", type=int, default=config.OCR_DPI,
                        help="DPI for OCR processing")
    parser.add_argument("--tesseract_path", type=str, default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        help="Path to Tesseract executable")
    parser.add_argument("--poppler_path", type=str, default=None,
                        help="Path to Poppler bin directory")
    parser.add_argument("--skip_ocr", action="store_true",
                        help="Skip OCR processing if already done")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip preprocessing if already done")
    
    return parser.parse_args()

def main():
    """Main function to run the knowledge graph construction pipeline."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set API key and base URL in environment variables
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    
    # Set Tesseract path
    if args.tesseract_path:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    
    # Initialize modules
    ocr_processor = OCRProcessor(dpi=args.ocr_dpi, poppler_path=args.poppler_path)
    text_preprocessor = TextPreprocessor()
    entity_extractor = EntityExtractor(model_name=args.model, api_key=args.openai_api_key, api_base=args.openai_api_base)
    entity_definer = EntityDefiner(model_name=args.model, api_key=args.openai_api_key, api_base=args.openai_api_base)
    entity_canonicalizer = EntityCanonicalizer(model_name=args.model, api_key=args.openai_api_key, api_base=args.openai_api_base)
    type_explorer = TypeExplorer(model_name=args.model, api_key=args.openai_api_key, api_base=args.openai_api_base)
    entity_matcher = EntityMatcher()
    kg_builder = KnowledgeGraphBuilder()
    quality_assessor = QualityAssessor()
    
    # Create data directories if they don't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.SLIDES_DIR, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = list(Path(args.input_dir).glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {args.input_dir}")
    
    if len(pdf_files) == 0:
        logger.warning(f"No PDF files found in {args.input_dir}. Please add PDF files to this directory.")
        return
    
    all_entities = []
    all_relationships = []
    
    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        pdf_name = pdf_file.stem
        logger.info(f"Processing {pdf_name}")
        
        # Step 1: OCR Processing
        if not args.skip_ocr:
            ocr_output_file = os.path.join(args.output_dir, f"{pdf_name}_ocr.json")
            try:
                ocr_result = ocr_processor.process(pdf_file)
                with open(ocr_output_file, 'w', encoding='utf-8') as f:
                    json.dump(ocr_result, f, indent=2, ensure_ascii=False)
                logger.info(f"OCR processing completed for {pdf_name}")
            except Exception as e:
                logger.error(f"Error during OCR processing for {pdf_name}: {e}")
                continue
        else:
            ocr_output_file = os.path.join(args.output_dir, f"{pdf_name}_ocr.json")
            try:
                with open(ocr_output_file, 'r', encoding='utf-8') as f:
                    ocr_result = json.load(f)
            except Exception as e:
                logger.error(f"Error loading OCR results for {pdf_name}: {e}")
                continue
        
        # Step 2: Preprocessing
        if not args.skip_preprocessing:
            preprocessed_output_file = os.path.join(args.output_dir, f"{pdf_name}_preprocessed.json")
            try:
                preprocessed_result = text_preprocessor.process(ocr_result)
                with open(preprocessed_output_file, 'w', encoding='utf-8') as f:
                    json.dump(preprocessed_result, f, indent=2, ensure_ascii=False)
                logger.info(f"Preprocessing completed for {pdf_name}")
            except Exception as e:
                logger.error(f"Error during preprocessing for {pdf_name}: {e}")
                continue
        else:
            preprocessed_output_file = os.path.join(args.output_dir, f"{pdf_name}_preprocessed.json")
            try:
                with open(preprocessed_output_file, 'r', encoding='utf-8') as f:
                    preprocessed_result = json.load(f)
            except Exception as e:
                logger.error(f"Error loading preprocessed results for {pdf_name}: {e}")
                continue
        
        # Step 3: Entity and Relationship Extraction
        extraction_output_file = os.path.join(args.output_dir, f"{pdf_name}_extraction.json")
        try:
            extraction_result = entity_extractor.extract(preprocessed_result)
            with open(extraction_output_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Entity extraction completed for {pdf_name}")
        except Exception as e:
            logger.error(f"Error during entity extraction for {pdf_name}: {e}")
            continue
        
        # Step 4: Entity and Relationship Definition
        definition_output_file = os.path.join(args.output_dir, f"{pdf_name}_definition.json")
        try:
            definition_result = entity_definer.define(extraction_result)
            with open(definition_output_file, 'w', encoding='utf-8') as f:
                json.dump(definition_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Entity definition completed for {pdf_name}")
        except Exception as e:
            logger.error(f"Error during entity definition for {pdf_name}: {e}")
            continue
        
        # Step 5: Canonicalization
        canonicalization_output_file = os.path.join(args.output_dir, f"{pdf_name}_canonicalization.json")
        try:
            canonicalization_result = entity_canonicalizer.canonicalize(definition_result)
            with open(canonicalization_output_file, 'w', encoding='utf-8') as f:
                json.dump(canonicalization_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Canonicalization completed for {pdf_name}")
        except Exception as e:
            logger.error(f"Error during canonicalization for {pdf_name}: {e}")
            continue
        
        # Step 6: Type Exploration
        exploration_output_file = os.path.join(args.output_dir, f"{pdf_name}_exploration.json")
        try:
            exploration_result = type_explorer.explore(canonicalization_result)
            with open(exploration_output_file, 'w', encoding='utf-8') as f:
                json.dump(exploration_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Type exploration completed for {pdf_name}")
        except Exception as e:
            logger.error(f"Error during type exploration for {pdf_name}: {e}")
            continue
        
        # Step 7: Entity Matching
        matching_output_file = os.path.join(args.output_dir, f"{pdf_name}_matching.json")
        try:
            matching_result = entity_matcher.match(exploration_result, ocr_result)
            with open(matching_output_file, 'w', encoding='utf-8') as f:
                json.dump(matching_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Entity matching completed for {pdf_name}")
        except Exception as e:
            logger.error(f"Error during entity matching for {pdf_name}: {e}")
            continue
        
        # Collect entities and relationships
        all_entities.extend(matching_result['entities'])
        all_relationships.extend(matching_result['relationships'])
    
    if not all_entities:
        logger.warning("No entities were extracted. Cannot build knowledge graph.")
        return
    
    # Step 8: Knowledge Graph Construction
    kg_output_file = os.path.join(args.output_dir, "knowledge_graph.json")
    try:
        knowledge_graph = kg_builder.build(all_entities, all_relationships)
        with open(kg_output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)
        logger.info("Knowledge graph construction completed")
    except Exception as e:
        logger.error(f"Error during knowledge graph construction: {e}")
        return
    
    # Step 9: Quality Assessment
    quality_output_file = os.path.join(args.output_dir, "quality_assessment.json")
    try:
        quality_result = quality_assessor.assess(knowledge_graph)
        with open(quality_output_file, 'w', encoding='utf-8') as f:
            json.dump(quality_result, f, indent=2, ensure_ascii=False)
        logger.info("Quality assessment completed")
    except Exception as e:
        logger.error(f"Error during quality assessment: {e}")
    
    logger.info("Knowledge graph construction pipeline completed successfully")

if __name__ == "__main__":
    main()