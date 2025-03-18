"""
Entity canonicalizer for standardizing entities and relationships.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

class EntityCanonicalizer:
    """
    Class for canonicalizing entities and relationships.
    """
    
    def __init__(self, model_name: str = config.DEEPSEEK_MODEL, api_key: str = None,
                embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the entity canonicalizer.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
            embedder_name: Name of the sentence transformer model for embeddings
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedder = SentenceTransformer(embedder_name)
            logger.info(f"Initialized sentence transformer: {embedder_name}")
        except Exception as e:
            logger.error(f"Error initializing sentence transformer: {e}")
            self.embedder = None
        
        logger.info(f"Initialized entity canonicalizer with model {model_name}")
        
        # Load canonicalization prompt templates
        self.entity_canonicalization_prompt = self._load_prompt_template("entity_canonicalization")
        self.relationship_canonicalization_prompt = self._load_prompt_template("relationship_canonicalization")
        
        # Define standard entity types and relationship types
        self.standard_entity_types = config.ENTITY_TYPES
        self.standard_relationship_types = config.RELATIONSHIP_TYPES
    
    def canonicalize(self, definition_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canonicalize entities and relationships.
        
        Args:
            definition_result: Dictionary containing defined entities and relationships
            
        Returns:
            Dictionary containing canonicalized entities and relationships
        """
        logger.info(f"Canonicalizing entities and relationships from {definition_result['filename']}")
        
        result = definition_result.copy()
        
        # Canonicalize entities
        canonicalized_entities = self._canonicalize_entities(definition_result["entities"])
        result["entities"] = canonicalized_entities
        
        # Canonicalize relationships
        canonicalized_relationships = self._canonicalize_relationships(
            definition_result["relationships"], 
            canonicalized_entities
        )
        result["relationships"] = canonicalized_relationships
        
        return result
    
    def _canonicalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Canonicalize entities.
        
        Args:
            entities: List of entities
            
        Returns:
            List of canonicalized entities
        """
        canonicalized_entities = []
        
        for entity in tqdm(entities, desc="Canonicalizing entities"):
            canonicalized_entity = entity.copy()
            
            # Canonicalize entity type
            canonical_type = self._get_canonical_entity_type(
                entity["name"], 
                entity["type"], 
                entity.get("definition", "")
            )
            canonicalized_entity["canonical_type"] = canonical_type
            
            # Normalize entity name
            canonical_name = self._normalize_entity_name(entity["name"])
            canonicalized_entity["canonical_name"] = canonical_name
            
            canonicalized_entities.append(canonicalized_entity)
        
        return canonicalized_entities
    
    def _canonicalize_relationships(self, relationships: List[Dict[str, Any]], 
                                   entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Canonicalize relationships.
        
        Args:
            relationships: List of relationships
            entities: List of canonicalized entities
            
        Returns:
            List of canonicalized relationships
        """
        # Create entity lookup dictionary
        entity_dict = {entity["id"]: entity for entity in entities}
        
        canonicalized_relationships = []
        
        for relationship in tqdm(relationships, desc="Canonicalizing relationships"):
            canonicalized_relationship = relationship.copy()
            
            # Get source and target entities
            source_entity = entity_dict.get(relationship["source"])
            target_entity = entity_dict.get(relationship["target"])
            
            if source_entity and target_entity:
                # Canonicalize relationship type
                canonical_type = self._get_canonical_relationship_type(
                    source_entity["name"], 
                    target_entity["name"], 
                    relationship["type"], 
                    relationship.get("definition", "")
                )
                canonicalized_relationship["canonical_type"] = canonical_type
                
                canonicalized_relationships.append(canonicalized_relationship)
        
        return canonicalized_relationships
    
    def _get_canonical_entity_type(self, entity_name: str, entity_type: str, entity_definition: str) -> str:
        """
        Get canonical entity type.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            entity_definition: Definition of the entity
            
        Returns:
            Canonical entity type
        """
        # If the entity type is already one of the standard types, return it
        if entity_type.lower() in [t.lower() for t in self.standard_entity_types]:
            return next(t for t in self.standard_entity_types if t.lower() == entity_type.lower())
        
        # Otherwise, use embeddings to find the most similar standard type
        if self.embedder:
            # Create a combined text for embedding
            combined_text = f"{entity_name}: {entity_definition}"
            
            # Get embedding for the entity
            entity_embedding = self.embedder.encode(combined_text)
            
            # Get embeddings for standard entity types
            type_embeddings = {}
            for standard_type in self.standard_entity_types:
                type_embedding = self.embedder.encode(standard_type)
                type_embeddings[standard_type] = type_embedding
            
            # Find the most similar standard type
            max_similarity = -1
            best_type = self.standard_entity_types[0]
            
            for standard_type, type_embedding in type_embeddings.items():
                similarity = np.dot(entity_embedding, type_embedding) / (
                    np.linalg.norm(entity_embedding) * np.linalg.norm(type_embedding)
                )
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_type = standard_type
            
            # If similarity is above threshold, return the best type
            if max_similarity > config.SIMILARITY_THRESHOLD:
                return best_type
        
        # If embeddings fail or similarity is below threshold, use LLM
        prompt = self.entity_canonicalization_prompt.format(
            entity_name=entity_name,
            entity_type=entity_type,
            entity_definition=entity_definition,
            standard_types=", ".join(self.standard_entity_types)
        )
        
        response = self._call_llm_api(prompt)
        
        # Parse response to get canonical type
        canonical_type = self._parse_canonical_type(response, self.standard_entity_types)
        
        return canonical_type
    
    def _get_canonical_relationship_type(self, source_name: str, target_name: str, 
                                        relationship_type: str, relationship_definition: str) -> str:
        """
        Get canonical relationship type.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relationship_type: Type of the relationship
            relationship_definition: Definition of the relationship
            
        Returns:
            Canonical relationship type
        """
        # If the relationship type is already one of the standard types, return it
        if relationship_type.lower() in [t.lower() for t in self.standard_relationship_types]:
            return next(t for t in self.standard_relationship_types if t.lower() == relationship_type.lower())
        
        # Otherwise, use embeddings to find the most similar standard type
        if self.embedder:
            # Create a combined text for embedding
            combined_text = f"{source_name} {relationship_type} {target_name}: {relationship_definition}"
            
            # Get embedding for the relationship
            relationship_embedding = self.embedder.encode(combined_text)
            
            # Get embeddings for standard relationship types
            type_embeddings = {}
            for standard_type in self.standard_relationship_types:
                type_embedding = self.embedder.encode(standard_type)
                type_embeddings[standard_type] = type_embedding
            
            # Find the most similar standard type
            max_similarity = -1
            best_type = self.standard_relationship_types[0]
            
            for standard_type, type_embedding in type_embeddings.items():
                similarity = np.dot(relationship_embedding, type_embedding) / (
                    np.linalg.norm(relationship_embedding) * np.linalg.norm(type_embedding)
                )
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_type = standard_type
            
            # If similarity is above threshold, return the best type
            if max_similarity > config.SIMILARITY_THRESHOLD:
                return best_type
        
        # If embeddings fail or similarity is below threshold, use LLM
        prompt = self.relationship_canonicalization_prompt.format(
            source_name=source_name,
            target_name=target_name,
            relationship_type=relationship_type,
            relationship_definition=relationship_definition,
            standard_types=", ".join(self.standard_relationship_types)
        )
        
        response = self._call_llm_api(prompt)
        
        # Parse response to get canonical type
        canonical_type = self._parse_canonical_type(response, self.standard_relationship_types)
        
        return canonical_type
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """
        Normalize entity name.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Normalized entity name
        """
        # Simple normalization: lowercase, remove extra spaces
        normalized_name = entity_name.strip()
        
        # Handle common abbreviations in computer networks
        abbreviations = {
            "transmission control protocol": "TCP",
            "internet protocol": "IP",
            "user datagram protocol": "UDP",
            "hypertext transfer protocol": "HTTP",
            "domain name system": "DNS",
            "file transfer protocol": "FTP",
            "simple mail transfer protocol": "SMTP",
            "open systems interconnection": "OSI",
            "local area network": "LAN",
            "wide area network": "WAN",
            "virtual local area network": "VLAN",
            "media access control": "MAC",
            "routing information protocol": "RIP",
            "open shortest path first": "OSPF",
            "border gateway protocol": "BGP",
            "dynamic host configuration protocol": "DHCP"
        }
        
        # Check if the normalized name is a known abbreviation
        for full_name, abbr in abbreviations.items():
            if normalized_name.lower() == full_name:
                return abbr
        
        return normalized_name
    
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
                return "concept"
            else:
                return "is_a"
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return ""
    
    def _parse_canonical_type(self, response: str, standard_types: List[str]) -> str:
        """
        Parse canonical type from LLM response.
        
        Args:
            response: LLM response
            standard_types: List of standard types
            
        Returns:
            Canonical type
        """
        # This is a placeholder for actual parsing logic
        # In a real implementation, you would parse the response to extract the canonical type
        
        # For now, just check if any standard type is in the response
        for standard_type in standard_types:
            if standard_type.lower() in response.lower():
                return standard_type
        
        # If no standard type is found, return the first standard type
        return standard_types[0]
    
    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Prompt template string
        """
        # In a real implementation, you would load templates from files
        
        if template_name == "entity_canonicalization":
            return """
            Map the following computer network entity to one of the standard entity types:
            
            Entity Name: {entity_name}
            Current Entity Type: {entity_type}
            Entity Definition: {entity_definition}
            
            Standard Entity Types: {standard_types}
            
            Return only the most appropriate standard entity type from the list above.
            """
        elif template_name == "relationship_canonicalization":
            return """
            Map the following relationship between computer network entities to one of the standard relationship types:
            
            Source Entity: {source_name}
            Target Entity: {target_name}
            Current Relationship Type: {relationship_type}
            Relationship Definition: {relationship_definition}
            
            Standard Relationship Types: {standard_types}
            
            Return only the most appropriate standard relationship type from the list above.
            """
        else:
            return ""