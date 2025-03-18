"""
Entity definer for defining entity types and relationships.
"""

import os
import json
import logging
from typing import Dict, List, Any

from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

class EntityDefiner:
    """
    Class for defining entity types and relationships.
    """
    
    def __init__(self, model_name: str = config.DEEPSEEK_MODEL, api_key: str = None):
        """
        Initialize the entity definer.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        logger.info(f"Initialized entity definer with model {model_name}")
        
        # Load definition prompt templates
        self.entity_definition_prompt = self._load_prompt_template("entity_definition")
        self.relationship_definition_prompt = self._load_prompt_template("relationship_definition")
    
    def define(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define entity types and relationships.
        
        Args:
            extraction_result: Dictionary containing extracted entities and relationships
            
        Returns:
            Dictionary containing defined entities and relationships
        """
        logger.info(f"Defining entities and relationships from {extraction_result['filename']}")
        
        result = extraction_result.copy()
        
        # Define entities
        defined_entities = self._define_entities(extraction_result["entities"])
        result["entities"] = defined_entities
        
        # Define relationships
        defined_relationships = self._define_relationships(extraction_result["relationships"], defined_entities)
        result["relationships"] = defined_relationships
        
        return result
    
    def _define_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Define entities.
        
        Args:
            entities: List of entities
            
        Returns:
            List of defined entities
        """
        defined_entities = []
        
        for entity in tqdm(entities, desc="Defining entities"):
            defined_entity = entity.copy()
            
            # Get entity definition
            definition = self._get_entity_definition(entity["name"], entity["type"])
            defined_entity["definition"] = definition
            
            defined_entities.append(defined_entity)
        
        return defined_entities
    
    def _define_relationships(self, relationships: List[Dict[str, Any]], 
                             entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Define relationships.
        
        Args:
            relationships: List of relationships
            entities: List of defined entities
            
        Returns:
            List of defined relationships
        """
        # Create entity lookup dictionary
        entity_dict = {entity["id"]: entity for entity in entities}
        
        defined_relationships = []
        
        for relationship in tqdm(relationships, desc="Defining relationships"):
            defined_relationship = relationship.copy()
            
            # Get source and target entities
            source_entity = entity_dict.get(relationship["source"])
            target_entity = entity_dict.get(relationship["target"])
            
            if source_entity and target_entity:
                # Get relationship definition
                definition = self._get_relationship_definition(
                    source_entity["name"], 
                    target_entity["name"], 
                    relationship["type"]
                )
                defined_relationship["definition"] = definition
                
                # Add source and target entity names for convenience
                defined_relationship["source_name"] = source_entity["name"]
                defined_relationship["target_name"] = target_entity["name"]
                
                defined_relationships.append(defined_relationship)
        
        return defined_relationships
    
    def _get_entity_definition(self, entity_name: str, entity_type: str) -> str:
        """
        Get definition for an entity.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            
        Returns:
            Entity definition
        """
        # Prepare prompt for entity definition
        prompt = self.entity_definition_prompt.format(
            entity_name=entity_name,
            entity_type=entity_type
        )
        
        # Call LLM API
        response = self._call_llm_api(prompt)
        
        return response
    
    def _get_relationship_definition(self, source_name: str, target_name: str, relationship_type: str) -> str:
        """
        Get definition for a relationship.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relationship_type: Type of the relationship
            
        Returns:
            Relationship definition
        """
        # Prepare prompt for relationship definition
        prompt = self.relationship_definition_prompt.format(
            source_name=source_name,
            target_name=target_name,
            relationship_type=relationship_type
        )
        
        # Call LLM API
        response = self._call_llm_api(prompt)
        
        return response
    
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
            if "entity_name" in prompt:
                return "This is a definition of the entity in computer networks."
            else:
                return "This is a definition of the relationship between the two entities in computer networks."
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return ""
    
    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Prompt template string
        """
        # In a real implementation, you would load templates from files
        
        if template_name == "entity_definition":
            return """
            Provide a clear and concise definition for the following computer network entity:
            
            Entity Name: {entity_name}
            Entity Type: {entity_type}
            
            Your definition should be suitable for a computer network course knowledge graph.
            """
        elif template_name == "relationship_definition":
            return """
            Provide a clear and concise definition for the following relationship between two computer network entities:
            
            Source Entity: {source_name}
            Target Entity: {target_name}
            Relationship Type: {relationship_type}
            
            Your definition should explain how these two entities are related in the context of computer networks.
            """
        else:
            return ""