"""
Quality assessor for evaluating the quality of the knowledge graph.
"""

import logging
from typing import Dict, List, Any

import config

logger = logging.getLogger(__name__)

class QualityAssessor:
    """
    Class for assessing the quality of the knowledge graph.
    """
    
    def __init__(self):
        """
        Initialize the quality assessor.
        """
        logger.info("Initialized quality assessor")
    
    def assess(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of the knowledge graph.
        
        Args:
            knowledge_graph: Dictionary containing the knowledge graph
            
        Returns:
            Dictionary containing quality assessment results
        """
        logger.info("Assessing knowledge graph quality")
        
        entities = knowledge_graph["entities"]
        relationships = knowledge_graph["relationships"]
        analysis = knowledge_graph["analysis"]
        
        # Calculate quality metrics
        entity_coverage = self._calculate_entity_coverage(entities)
        relationship_coverage = self._calculate_relationship_coverage(relationships)
        entity_type_balance = self._calculate_entity_type_balance(analysis["entity_type_distribution"])
        relationship_type_balance = self._calculate_relationship_type_balance(analysis["relationship_type_distribution"])
        graph_connectivity = self._calculate_graph_connectivity(analysis)
        chapter_coverage = self._calculate_chapter_coverage(analysis["chapter_distribution"])
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(
            entity_coverage,
            relationship_coverage,
            entity_type_balance,
            relationship_type_balance,
            graph_connectivity,
            chapter_coverage
        )
        
        return {
            "entity_coverage": entity_coverage,
            "relationship_coverage": relationship_coverage,
            "entity_type_balance": entity_type_balance,
            "relationship_type_balance": relationship_type_balance,
            "graph_connectivity": graph_connectivity,
            "chapter_coverage": chapter_coverage,
            "overall_score": overall_score
        }
    
    def _calculate_entity_coverage(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate entity coverage metrics.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary containing entity coverage metrics
        """
        total_entities = len(entities)
        
        # Count entities with definitions
        entities_with_definition = sum(1 for entity in entities if entity.get("definition"))
        
        # Count entities with verified occurrences
        entities_verified = sum(1 for entity in entities if entity.get("verified", True))
        
        # Calculate percentages
        definition_percentage = entities_with_definition / total_entities * 100 if total_entities > 0 else 0
        verified_percentage = entities_verified / total_entities * 100 if total_entities > 0 else 0
        
        # Calculate score (0-100)
        score = (definition_percentage + verified_percentage) / 2
        
        return {
            "total_entities": total_entities,
            "entities_with_definition": entities_with_definition,
            "entities_verified": entities_verified,
            "definition_percentage": definition_percentage,
            "verified_percentage": verified_percentage,
            "score": score
        }
    
    def _calculate_relationship_coverage(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate relationship coverage metrics.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Dictionary containing relationship coverage metrics
        """
        total_relationships = len(relationships)
        
        # Count relationships with definitions
        relationships_with_definition = sum(1 for rel in relationships if rel.get("definition"))
        
        # Count relationships with verified occurrences
        relationships_verified = sum(1 for rel in relationships if rel.get("verified", True))
        
        # Calculate percentages
        definition_percentage = relationships_with_definition / total_relationships * 100 if total_relationships > 0 else 0
        verified_percentage = relationships_verified / total_relationships * 100 if total_relationships > 0 else 0
        
        # Calculate score (0-100)
        score = (definition_percentage + verified_percentage) / 2
        
        return {
            "total_relationships": total_relationships,
            "relationships_with_definition": relationships_with_definition,
            "relationships_verified": relationships_verified,
            "definition_percentage": definition_percentage,
            "verified_percentage": verified_percentage,
            "score": score
        }
    
    def _calculate_entity_type_balance(self, entity_type_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate entity type balance metrics.
        
        Args:
            entity_type_distribution: Distribution of entity types
            
        Returns:
            Dictionary containing entity type balance metrics
        """
        total_entities = sum(entity_type_distribution.values())
        
        # Calculate percentages for each type
        type_percentages = {
            entity_type: count / total_entities * 100 if total_entities > 0 else 0
            for entity_type, count in entity_type_distribution.items()
        }
        
        # Calculate balance score
        # A perfectly balanced distribution would have equal percentages for all types
        # We calculate how far the actual distribution is from the ideal
        num_types = len(entity_type_distribution)
        ideal_percentage = 100 / num_types if num_types > 0 else 0
        
        # Calculate the sum of absolute differences from the ideal percentage
        sum_diff = sum(abs(percentage - ideal_percentage) for percentage in type_percentages.values())
        
        # Normalize to a 0-100 score (0 = completely unbalanced, 100 = perfectly balanced)
        # The maximum possible sum of differences is 2 * (100 - ideal_percentage)
        max_diff = 2 * (100 - ideal_percentage) if num_types > 1 else 0
        balance_score = 100 - (sum_diff / max_diff * 100) if max_diff > 0 else 100
        
        return {
            "type_distribution": entity_type_distribution,
            "type_percentages": type_percentages,
            "balance_score": balance_score
        }
    
    def _calculate_relationship_type_balance(self, relationship_type_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate relationship type balance metrics.
        
        Args:
            relationship_type_distribution: Distribution of relationship types
            
        Returns:
            Dictionary containing relationship type balance metrics
        """
        total_relationships = sum(relationship_type_distribution.values())
        
        # Calculate percentages for each type
        type_percentages = {
            relationship_type: count / total_relationships * 100 if total_relationships > 0 else 0
            for relationship_type, count in relationship_type_distribution.items()
        }
        
        # Calculate balance score
        # A perfectly balanced distribution would have equal percentages for all types
        # We calculate how far the actual distribution is from the ideal
        num_types = len(relationship_type_distribution)
        ideal_percentage = 100 / num_types if num_types > 0 else 0
        
        # Calculate the sum of absolute differences from the ideal percentage
        sum_diff = sum(abs(percentage - ideal_percentage) for percentage in type_percentages.values())
        
        # Normalize to a 0-100 score (0 = completely unbalanced, 100 = perfectly balanced)
        # The maximum possible sum of differences is 2 * (100 - ideal_percentage)
        max_diff = 2 * (100 - ideal_percentage) if num_types > 1 else 0
        balance_score = 100 - (sum_diff / max_diff * 100) if max_diff > 0 else 100
        
        return {
            "type_distribution": relationship_type_distribution,
            "type_percentages": type_percentages,
            "balance_score": balance_score
        }
    
    def _calculate_graph_connectivity(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate graph connectivity metrics.
        
        Args:
            analysis: Graph analysis
            
        Returns:
            Dictionary containing graph connectivity metrics
        """
        # Extract relevant metrics from analysis
        num_entities = analysis["num_entities"]
        num_relationships = analysis["num_relationships"]
        density = analysis["density"]
        num_components = analysis["num_components"]
        largest_component_percentage = analysis["largest_component_percentage"]
        
        # Calculate connectivity score
        # A well-connected graph has:
        # - High density
        # - Few components (ideally 1)
        # - Large largest component (ideally 100%)
        
        # Normalize density to 0-100
        # Density is between 0 and 1, where 1 is a complete graph
        # For knowledge graphs, a density of 0.1 is already quite high
        density_score = min(density * 1000, 100)
        
        # Normalize number of components
        # Ideally, we want 1 component (fully connected)
        # We use an exponential decay function: score = 100 * e^(-0.5 * (num_components - 1))
        import math
        component_score = 100 * math.exp(-0.5 * (num_components - 1))
        
        # Largest component percentage is already 0-100
        
        # Calculate overall connectivity score as weighted average
        connectivity_score = (
            0.3 * density_score +
            0.3 * component_score +
            0.4 * largest_component_percentage
        )
        
        return {
            "density": density,
            "num_components": num_components,
            "largest_component_percentage": largest_component_percentage,
            "density_score": density_score,
            "component_score": component_score,
            "connectivity_score": connectivity_score
        }
    
    def _calculate_chapter_coverage(self, chapter_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate chapter coverage metrics.
        
        Args:
            chapter_distribution: Distribution of entities by chapter
            
        Returns:
            Dictionary containing chapter coverage metrics
        """
        total_entities = sum(chapter_distribution.values())
        
        # Calculate percentages for each chapter
        chapter_percentages = {
            chapter: count / total_entities * 100 if total_entities > 0 else 0
            for chapter, count in chapter_distribution.items()
        }
        
        # Calculate coverage score
        # A good coverage has entities from all chapters
        # We calculate how far the actual distribution is from the ideal
        num_chapters = len(chapter_distribution)
        ideal_percentage = 100 / num_chapters if num_chapters > 0 else 0
        
        # Calculate the sum of absolute differences from the ideal percentage
        sum_diff = sum(abs(percentage - ideal_percentage) for percentage in chapter_percentages.values())
        
        # Normalize to a 0-100 score (0 = completely unbalanced, 100 = perfectly balanced)
        # The maximum possible sum of differences is 2 * (100 - ideal_percentage)
        max_diff = 2 * (100 - ideal_percentage) if num_chapters > 1 else 0
        coverage_score = 100 - (sum_diff / max_diff * 100) if max_diff > 0 else 100
        
        return {
            "chapter_distribution": chapter_distribution,
            "chapter_percentages": chapter_percentages,
            "coverage_score": coverage_score
        }
    
    def _calculate_overall_score(self, entity_coverage: Dict[str, Any], 
                               relationship_coverage: Dict[str, Any],
                               entity_type_balance: Dict[str, Any],
                               relationship_type_balance: Dict[str, Any],
                               graph_connectivity: Dict[str, Any],
                               chapter_coverage: Dict[str, Any]) -> float:
        """
        Calculate overall quality score.
        
        Args:
            entity_coverage: Entity coverage metrics
            relationship_coverage: Relationship coverage metrics
            entity_type_balance: Entity type balance metrics
            relationship_type_balance: Relationship type balance metrics
            graph_connectivity: Graph connectivity metrics
            chapter_coverage: Chapter coverage metrics
            
        Returns:
            Overall quality score (0-100)
        """
        # Extract individual scores
        entity_coverage_score = entity_coverage["score"]
        relationship_coverage_score = relationship_coverage["score"]
        entity_type_balance_score = entity_type_balance["balance_score"]
        relationship_type_balance_score = relationship_type_balance["balance_score"]
        graph_connectivity_score = graph_connectivity["connectivity_score"]
        chapter_coverage_score = chapter_coverage["coverage_score"]
        
        # Calculate weighted average
        overall_score = (
            0.2 * entity_coverage_score +
            0.2 * relationship_coverage_score +
            0.15 * entity_type_balance_score +
            0.15 * relationship_type_balance_score +
            0.2 * graph_connectivity_score +
            0.1 * chapter_coverage_score
        )
        
        return overall_score