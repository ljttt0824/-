"""
Type explorer for discovering new entity types and relationship types.
"""

import os
import json
import logging
from typing import Dict, List, Any, Set
from collections import Counter

from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

class TypeExplorer:
    """
    Class for exploring and discovering new entity types and relationship types.
    """
    
    def __init__(self, model_name: str = config.DEEPSEEK_MODEL, api_key: str = None):
        """
        Initialize the type explorer.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        logger.info(f"Initialized type explorer with model {model_name}")
        
        # Load exploration prompt templates
        self.entity_type_exploration_prompt = self._load_prompt_template("entity_type_exploration")
        self.relationship_type_exploration_prompt = self._load_prompt_template("relationship_type_exploration")
    
    def explore(self, canonicalization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explore and discover new entity types and relationship types.
        
        Args:
            canonicalization_result: Dictionary containing canonicalized entities and relationships
            
        Returns:
            Dictionary containing exploration results
        """
        logger.info(f"Exploring entity and relationship types from {canonicalization_result['filename']}")
        
        result = canonicalization_result.copy()
        
        # Analyze entity types
        entity_type_analysis = self._analyze_entity_types(canonicalization_result["entities"])
        result["entity_type_analysis"] = entity_type_analysis
        
        # Analyze relationship types
        relationship_type_analysis = self._analyze_relationship_types(canonicalization_result["relationships"])
        result["relationship_type_analysis"] = relationship_type_analysis
        
        # Discover new entity types
        new_entity_types = self._discover_new_entity_types(
            canonicalization_result["entities"],
            entity_type_analysis
        )
        result["new_entity_types"] = new_entity_types
        
        # Discover new relationship types
        new_relationship_types = self._discover_new_relationship_types(
            canonicalization_result["relationships"],
            relationship_type_analysis
        )
        result["new_relationship_types"] = new_relationship_types
        
        # Update entities with new types
        updated_entities = self._update_entities_with_new_types(
            canonicalization_result["entities"],
            new_entity_types
        )
        result["entities"] = updated_entities
        
        # Update relationships with new types
        updated_relationships = self._update_relationships_with_new_types(
            canonicalization_result["relationships"],
            new_relationship_types
        )
        result["relationships"] = updated_relationships
        
        return result
    
    def _analyze_entity_types(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze entity types.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary containing entity type analysis
        """
        # Count entity types
        type_counter = Counter([entity["canonical_type"] for entity in entities])
        
        # Calculate distribution
        total_entities = len(entities)
        type_distribution = {
            type_name: {"count": count, "percentage": count / total_entities * 100}
            for type_name, count in type_counter.items()
        }
        
        # Find entities with mismatched types
        mismatched_entities = [
            {
                "id": entity["id"],
                "name": entity["name"],
                "original_type": entity["type"],
                "canonical_type": entity["canonical_type"]
            }
            for entity in entities
            if entity["type"].lower() != entity["canonical_type"].lower()
        ]
        
        return {
            "type_distribution": type_distribution,
            "mismatched_entities": mismatched_entities
        }
    
    def _analyze_relationship_types(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze relationship types.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Dictionary containing relationship type analysis
        """
        # Count relationship types
        type_counter = Counter([relationship["canonical_type"] for relationship in relationships])
        
        # Calculate distribution
        total_relationships = len(relationships)
        type_distribution = {
            type_name: {"count": count, "percentage": count / total_relationships * 100 if total_relationships > 0 else 0}
            for type_name, count in type_counter.items()
        }
        
        # Find relationships with mismatched types
        mismatched_relationships = [
            {
                "id": relationship["id"],
                "source_name": relationship["source_name"],
                "target_name": relationship["target_name"],
                "original_type": relationship["type"],
                "canonical_type": relationship["canonical_type"]
            }
            for relationship in relationships
            if relationship["type"].lower() != relationship["canonical_type"].lower()
        ]
        
        return {
            "type_distribution": type_distribution,
            "mismatched_relationships": mismatched_relationships
        }
    
    def _discover_new_entity_types(self, entities: List[Dict[str, Any]], 
                                  entity_type_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover new entity types.
        
        Args:
            entities: List of entities
            entity_type_analysis: Entity type analysis
            
        Returns:
            List of new entity types
        """
        # Get mismatched entities
        mismatched_entities = entity_type_analysis["mismatched_entities"]
        
        if not mismatched_entities:
            return []
        
        # Group mismatched entities by original type
        original_types = {}
        for entity in mismatched_entities:
            original_type = entity["original_type"].lower()
            if original_type not in original_types:
                original_types[original_type] = []
            original_types[original_type].append(entity)
        
        # Filter out types with too few entities
        significant_types = {
            type_name: entities
            for type_name, entities in original_types.items()
            if len(entities) >= 3  # Require at least 3 entities to consider a new type
        }
        
        if not significant_types:
            return []
        
        # Prepare data for LLM
        type_examples = []
        for type_name, type_entities in significant_types.items():
            entity_examples = [entity["name"] for entity in type_entities[:5]]
            type_examples.append({
                "type": type_name,
                "examples": entity_examples
            })
        
        # Call LLM to analyze and suggest new types
        prompt = self.entity_type_exploration_prompt.format(
            type_examples=json.dumps(type_examples, indent=2),
            standard_types=", ".join(config.ENTITY_TYPES)
        )
        
        response = self._call_llm_api(prompt)
        
        # Parse response to get new entity types
        new_entity_types = self._parse_new_types(response)
        
        return new_entity_types
    
    def _discover_new_relationship_types(self, relationships: List[Dict[str, Any]], 
                                        relationship_type_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover new relationship types.
        
        Args:
            relationships: List of relationships
            relationship_type_analysis: Relationship type analysis
            
        Returns:
            List of new relationship types
        """
        # Get mismatched relationships
        mismatched_relationships = relationship_type_analysis["mismatched_relationships"]
        
        if not mismatched_relationships:
            return []
        
        # Group mismatched relationships by original type
        original_types = {}
        for relationship in mismatched_relationships:
            original_type = relationship["original_type"].lower()
            if original_type not in original_types:
                original_types[original_type] = []
            original_types[original_type].append(relationship)
        
        # Filter out types with too few relationships
        significant_types = {
            type_name: rels
            for type_name, rels in original_types.items()
            if len(rels) >= 3  # Require at least 3 relationships to consider a new type
        }
        
        if not significant_types:
            return []
        
        # Prepare data for LLM
        type_examples = []
        for type_name, type_relationships in significant_types.items():
            relationship_examples = [
                {
                    "source": rel["source_name"],
                    "target": rel["target_name"]
                }
                for rel in type_relationships[:5]
            ]
            type_examples.append({
                "type": type_name,
                "examples": relationship_examples
            })
        
        # Call LLM to analyze and suggest new types
        prompt = self.relationship_type_exploration_prompt.format(
            type_examples=json.dumps(type_examples, indent=2),
            standard_types=", ".join(config.RELATIONSHIP_TYPES)
        )
        
        response = self._call_llm_api(prompt)
        
        # Parse response to get new relationship types
        new_relationship_types = self._parse_new_types(response)
        
        return new_relationship_types
    
    def _update_entities_with_new_types(self, entities: List[Dict[str, Any]], 
                                       new_entity_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update entities with new types.
        
        Args:
            entities: List of entities
            new_entity_types: List of new entity types
            
        Returns:
            List of updated entities
        """
        if not new_entity_types:
            return entities
        
        # Create a mapping from original type to new type
        type_mapping = {}
        for new_type in new_entity_types:
            for original_type in new_type.get("original_types", []):
                type_mapping[original_type.lower()] = new_type["name"]
        
        # Update entities
        updated_entities = []
        for entity in entities:
            updated_entity = entity.copy()
            
            # Check if the entity's original type maps to a new type
            original_type = entity["type"].lower()
            if original_type in type_mapping:
                updated_entity["explored_type"] = type_mapping[original_type]
            else:
                updated_entity["explored_type"] = entity["canonical_type"]
            
            updated_entities.append(updated_entity)
        
        return updated_entities
    
    def _update_relationships_with_new_types(self, relationships: List[Dict[str, Any]], 
                                           new_relationship_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update relationships with new types.
        
        Args:
            relationships: List of relationships
            new_relationship_types: List of new relationship types
            
        Returns:
            List of updated relationships
        """
        if not new_relationship_types:
            return relationships
        
        # Create a mapping from original type to new type
        type_mapping = {}
        for new_type in new_relationship_types:
            for original_type in new_type.get("original_types", []):
                type_mapping[original_type.lower()] = new_type["name"]
        
        # Update relationships
        updated_relationships = []
        for relationship in relationships:
            updated_relationship = relationship.copy()
            
            # Check if the relationship's original type maps to a new type
            original_type = relationship["type"].lower()
            if original_type in type_mapping:
                updated_relationship["explored_type"] = type_mapping[original_type]
            else:
                updated_relationship["explored_type"] = relationship["canonical_type"]
            
            updated_relationships.append(updated_relationship)
        
        return updated_relationships
    
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
            return """
            {
                "new_types": [
                    {
                        "name": "network_protocol",
                        "definition": "A set of rules that govern data communication between network entities",
                        "original_types": ["protocol", "communication_protocol"]
                    }
                ]
            }
            """
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return ""
    
    def _parse_new_types(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse new types from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            List of new types
        """
        try:
            # Try to parse JSON response
            data = json.loads(response)
            return data.get("new_types", [])
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract structured information
            new_types = []
            
            # Simple regex-based extraction (this is a fallback and not robust)
            import re
            type_blocks = re.split(r'\n\s*\d+\.', response)
            
            for block in type_blocks:
                if not block.strip():
                    continue
                
                # Try to extract name
                name_match = re.search(r'name:\s*([a-z_]+)', block, re.IGNORECASE)
                if not name_match:
                    continue
                
                name = name_match.group(1).strip()
                
                # Try to extract definition
                definition_match = re.search(r'definition:\s*(.+?)(?:\n|$)', block, re.IGNORECASE | re.DOTALL)
                definition = definition_match.group(1).strip() if definition_match else ""
                
                # Try to extract original types
                original_types_match = re.search(r'original types:\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
                original_types = []
                if original_types_match:
                    types_text = original_types_match.group(1).strip()
                    original_types = [t.strip() for t in re.split(r',|\s+and\s+', types_text) if t.strip()]
                
                new_types.append({
                    "name": name,
                    "definition": definition,
                    "original_types": original_types
                })
            
            return new_types
    
    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Prompt template string
        """
        # In a real implementation, you would load templates from files
        
        if template_name == "entity_type_exploration":
            return """
            Analyze the following entity types and examples from a computer network knowledge graph.
            These types were originally identified but were mapped to standard types during canonicalization.
            
            Type Examples:
            {type_examples}
            
            Standard Entity Types: {standard_types}
            
            Your task is to:
            1. Analyze if any of these original types represent meaningful categories that are not well-captured by the standard types
            2. Suggest new entity types that should be added to the standard types
            
            For each suggested new type, provide:
            - A name (use snake_case)
            - A clear definition
            - Which original types should map to this new type
            
            Return your analysis in the following JSON format:
            {{
                "new_types": [
                    {{
                        "name": "type_name",
                        "definition": "type definition",
                        "original_types": ["original_type1", "original_type2"]
                    }}
                ]
            }}
            
            If you don't recommend any new types, return an empty list for "new_types".
            """
        elif template_name == "relationship_type_exploration":
            return """
            Analyze the following relationship types and examples from a computer network knowledge graph.
            These types were originally identified but were mapped to standard types during canonicalization.
            
            Type Examples:
            {type_examples}
            
            Standard Relationship Types: {standard_types}
            
            Your task is to:
            1. Analyze if any of these original types represent meaningful relationships that are not well-captured by the standard types
            2. Suggest new relationship types that should be added to the standard types
            
            For each suggested new type, provide:
            - A name (use snake_case)
            - A clear definition
            - Which original types should map to this new type
            
            Return your analysis in the following JSON format:
            {{
                "new_types": [
                    {{
                        "name": "type_name",
                        "definition": "type definition",
                        "original_types": ["original_type1", "original_type2"]
                    }}
                ]
            }}
            
            If you don't recommend any new types, return an empty list for "new_types".
            """
        else:
            return ""