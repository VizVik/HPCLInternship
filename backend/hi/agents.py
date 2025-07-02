# hi/agents.py
from .document_agent import DocumentAgent
from autogen import AssistantAgent
from .coding_agents import CodingAgent
from .analytics_agent import AnalyticsAgent
from .research_agents import  ResearchAgent
import os

# Initialize document agent first
document_agent = DocumentAgent()

research_agents = ResearchAgent()


coding_agents = CodingAgent() 
analytics_agent = AnalyticsAgent()

# Other agents remain the same
research_agent = AssistantAgent(
    name="ResearchAgent",
    system_message="You are a researcher. Search and explain complex topics simply."
)

analytics_agent = AssistantAgent(
    name="AnalyticsAgent",
    system_message="You are a data analyst. Perform data analysis and give results."
)

general_agent = AssistantAgent(
    name="GeneralAssistant",
    system_message="You are a general assistant for all-purpose questions."
)

AGENT_MAP = {
    "Document Agent": document_agent,
    "Coding Agent": coding_agents,
    "Research Agent": research_agent,
    "Analytics Agent": analytics_agent,
    "General Assistant": general_agent
}
