"""
Entity extractor for identifying entities and relationships from preprocessed text.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple

import requests
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Class for extracting entities (knowledge points) and relationships from preprocessed text.
    """
    
    def __init__(self, model_name: str = config.DEEPSEEK_MODEL, api_key: str = None):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        logger.info(f"Initialized entity extractor with model {model_name}")
        
        # Load extraction prompt templates
        self.entity_prompt = self._load_prompt_template("entity_extraction")
        self.relationship_prompt = self._load_prompt_template("relationship_extraction")
    
    def extract(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and relationships from preprocessed text.
        
        Args:
            preprocessed_data: Dictionary containing preprocessed text
            
        Returns:
            Dictionary containing extracted entities and relationships
        """
        logger.info(f"Extracting entities and relationships from {preprocessed_data['filename']}")
        
        result = {
            "filename": preprocessed_data["filename"],
            "chapter": preprocessed_data["chapter"],
            "entities": [],
            "relationships": [],
            "pages": []
        }
        
        # Process each page
        for page_data in tqdm(preprocessed_data["pages"], desc="Extracting entities and relationships"):
            page_result = self._process_page(page_data, preprocessed_data["chapter"])
            result["pages"].append(page_result)
            
            # Collect entities and relationships
            result["entities"].extend(page_result["entities"])
            result["relationships"].extend(page_result["relationships"])
        
        # Remove duplicate entities and relationships
        result["entities"] = self._remove_duplicate_entities(result["entities"])
        result["relationships"] = self._remove_duplicate_relationships(result["relationships"])
        
        return result
    
    def _process_page(self, page_data: Dict[str, Any], chapter: str) -> Dict[str, Any]:
        """
        Process a single page to extract entities and relationships.
        
        Args:
            page_data: Dictionary containing page data
            chapter: Chapter information
            
        Returns:
            Dictionary containing extracted entities and relationships for the page
        """
        page_number = page_data["page_number"]
        full_text = page_data["full_text"]
        
        # Extract entities
        entities = self._extract_entities(full_text, page_number, chapter)
        
        # Extract relationships
        relationships = self._extract_relationships(full_text, entities, page_number, chapter)
        
        return {
            "page_number": page_number,
            "entities": entities,
            "relationships": relationships
        }
    
    def _extract_entities(self, text: str, page_number: int, chapter: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            page_number: Page number
            chapter: Chapter information
            
        Returns:
            List of extracted entities
        """
        # Prepare prompt for entity extraction
        prompt = self.entity_prompt.format(
            text=text,
            entity_types=", ".join(config.ENTITY_TYPES)
        )
        
        # Call LLM API
        response = self._call_llm_api(prompt)
        
        # Parse entities from response
        entities = self._parse_entities(response, page_number, chapter)
        
        return entities
    
    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]], 
                              page_number: int, chapter: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        
        Args:
            text: Input text
            entities: List of entities
            page_number: Page number
            chapter: Chapter information
            
        Returns:
            List of extracted relationships
        """
        if len(entities) < 2:
            return []
        
        # Prepare entity list for prompt
        entity_list = "\n".join([f"- {entity['name']}" for entity in entities])
        
        # Prepare prompt for relationship extraction
        prompt = self.relationship_prompt.format(
            text=text,
            entities=entity_list,
            relationship_types=", ".join(config.RELATIONSHIP_TYPES)
        )
        
        # Call LLM API
        response = self._call_llm_api(prompt)
        
        # Parse relationships from response
        relationships = self._parse_relationships(response, entities, page_number, chapter)
        
        return relationships
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response
        """
        try:
            # This is a placeholder for the actual API call
            # In a real implementation, you would use the appropriate API client
            
            # Example for DeepSeek API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_TOKENS,
                "top_p": config.TOP_P
            }
            
            # Simulate API call for now
            # In a real implementation, you would make an actual API request
            # response = requests.post("https://api.deepseek.com/v1/chat/completions", 
            #                         headers=headers, json=data)
            # return response.json()["choices"][0]["message"]["content"]
            
            # For now, return a mock response
            return "This is a mock response. In a real implementation, this would be the LLM's response."
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return ""
    
    def _parse_entities(self, response: str, page_number: int, chapter: str) -> List[Dict[str, Any]]:
        """
        Parse entities from LLM response.
        
        Args:
            response: LLM response
            page_number: Page number
            chapter: Chapter information
            
        Returns:
            List of parsed entities
        """
        # This is a placeholder for actual parsing logic
        # In a real implementation, you would parse the JSON response from the LLM
        
        # Mock entities for demonstration
        entities = [
            {
                "id": f"entity_{page_number}_1",
                "name": "TCP/IP",
                "type": "protocol",
                "page_number": page_number,
                "chapter": chapter,
                "first_occurrence": True
            },
            {
                "id": f"entity_{page_number}_2",
                "name": "OSI Model",
                "type": "concept",
                "page_number": page_number,
                "chapter": chapter,
                "first_occurrence": True
            }
        ]
        
        return entities
    
    def _parse_relationships(self, response: str, entities: List[Dict[str, Any]], 
                            page_number: int, chapter: str) -> List[Dict[str, Any]]:
        """
        Parse relationships from LLM response.
        
        Args:
            response: LLM response
            entities: List of entities
            page_number: Page number
            chapter: Chapter information
            
        Returns:
            List of parsed relationships
        """
        # This is a placeholder for actual parsing logic
        # In a real implementation, you would parse the JSON response from the LLM
        
        # Mock relationships for demonstration
        if len(entities) < 2:
            return []
        
        relationships = [
            {
                "id": f"rel_{page_number}_1",
                "source": entities[0]["id"],
                "target": entities[1]["id"],
                "type": "is_a",
                "page_number": page_number,
                "chapter": chapter
            }
        ]
        
        return relationships
    
    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities based on name.
        
        Args:
            entities: List of entities
            
        Returns:
            List of unique entities
        """
        unique_entities = {}
        
        for entity in entities:
            name = entity["name"].lower()
            
            if name not in unique_entities:
                # Mark as first occurrence
                entity["first_occurrence"] = True
                unique_entities[name] = entity
            else:
                # Update existing entity with additional page occurrences
                if "occurrences" not in unique_entities[name]:
                    unique_entities[name]["occurrences"] = [unique_entities[name]["page_number"]]
                
                unique_entities[name]["occurrences"].append(entity["page_number"])
                
                # Keep the entity as not first occurrence
                entity["first_occurrence"] = False
        
        return list(unique_entities.values())
    
    def _remove_duplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships.
        
        Args:
            relationships: List of relationships
            
        Returns:
            List of unique relationships
        """
        unique_relationships = {}
        
        for relationship in relationships:
            key = f"{relationship['source']}_{relationship['type']}_{relationship['target']}"
            
            if key not in unique_relationships:
                unique_relationships[key] = relationship
            else:
                # Update existing relationship with additional page occurrences
                if "occurrences" not in unique_relationships[key]:
                    unique_relationships[key]["occurrences"] = [unique_relationships[key]["page_number"]]
                
                unique_relationships[key]["occurrences"].append(relationship["page_number"])
        
        return list(unique_relationships.values())
    
    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Prompt template string
        """
        # In a real implementation, you would load templates from files
        
        if template_name == "entity_extraction":
            return """
            Extract all computer network entities (knowledge points) from the following text.
            
            Text:
            {text}
            
            For each entity, identify its type from the following options: {entity_types}
            
            Return the results in JSON format with the following structure:
            [
                {
                    "name": "entity name",
                    "type": "entity type"
                }
            ]
            """
        elif template_name == "relationship_extraction":
            return """
            Identify relationships between the following computer network entities in the given text.
            
            Text:
            {text}
            
            Entities:
            {entities}
            
            For each relationship, identify its type from the following options: {relationship_types}
            
            Return the results in JSON format with the following structure:
            [
                {
                    "source": "source entity name",
                    "target": "target entity name",
                    "type": "relationship type"
                }
            ]
            """
        else:
            return ""