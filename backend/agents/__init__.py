"""
HPCL hpGPT Multi-Agent System

This package contains specialized AI agents for different domains:
- DocumentAgent: PDF processing, document analysis, Q&A
- AnalyticsAgent: Data analysis, business intelligence, reporting
- ResearchAgent: Market research, industry analysis, competitive intelligence
- CodingAgent: Code generation, debugging, automation scripts

Each agent is designed to work with HPCL's specific requirements and 
can be used independently or as part of the multi-agent workflow.
"""

from .document_agent import DocumentAgent

from .research_agent import ResearchAgent
from .coding_agent import CodingAgent

__all__ = [
    'DocumentAgent',
     
    'ResearchAgent',
    'CodingAgent'
]

# Version info
__version__ = '1.0.0'
__author__ = 'HPCL AI Team'

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    'document': DocumentAgent,
    
    'research': ResearchAgent,
    'coding': CodingAgent
}

def get_agent_class(agent_type: str):
    """
    Get agent class by type name
    
    Args:
        agent_type (str): Type of agent ('document', 'analytics', 'research', 'coding')
    
    Returns:
        Agent class or None if not found
    """
    return AGENT_REGISTRY.get(agent_type.lower())

def list_available_agents():
    """
    List all available agent types
    
    Returns:
        List of available agent type names
    """
    return list(AGENT_REGISTRY.keys())
