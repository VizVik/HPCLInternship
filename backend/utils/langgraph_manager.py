from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Literal, TypedDict, List, Dict, Optional
import os
import logging
from datetime import datetime
import sqlite3
import asyncio
import json
import uuid
from backend.hi.coding_agents import CodingAgent
from backend.hi.analytics_agent import AnalyticsAgent
from backend.hi.research_agents import ResearchAgent




logger = logging.getLogger(__name__)

from backend.hi.document_agents import DocumentAgent
from backend.utils.groq_client import groq_client

class GraphConfig(TypedDict):
    agent_type: Literal["general", "document", "analytics", "research", "coding"]

class HPGPTGraph:
    def __init__(self):
        self.document_agent = DocumentAgent()
        self.analytics_agent = AnalyticsAgent()
        self.research_agents = ResearchAgent()
        self.coding_agents = CodingAgent()
        self.db_path = "hpgpt_memory.db"
        self.sessions_file = "sessions.json"
        self.conversations_file = "conversations.json"
        self.feedback_file = "feedback.json"
        self.load_data()

    def load_data(self):
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    self.sessions = json.load(f)
            else:
                self.sessions = {}
            if os.path.exists(self.conversations_file):
                with open(self.conversations_file, 'r') as f:
                    self.conversations = json.load(f)
            else:
                self.conversations = {}
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
            else:
                self.feedback_data = {}
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.sessions = {}
            self.conversations = {}
            self.feedback_data = {}

    def save_data(self):
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
            with open(self.conversations_file, 'w') as f:
                json.dump(self.conversations, f, indent=2)
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    async def store_feedback(self, session_id: str, message_content: str, feedback_type: str, 
                           agent_type: str, answer_mode: str, timestamp: str) -> Dict:
        try:
            feedback_id = str(uuid.uuid4())
            feedback_entry = {
                "feedback_id": feedback_id,
                "session_id": session_id,
                "message_content": message_content[:500],
                "feedback_type": feedback_type,
                "agent_type": agent_type,
                "answer_mode": answer_mode,
                "timestamp": timestamp,
                "message_length": len(message_content),
                "created_at": datetime.now().isoformat()
            }
            if session_id not in self.feedback_data:
                self.feedback_data[session_id] = []
            self.feedback_data[session_id].append(feedback_entry)
            self.save_data()
            logger.info(f"Stored {feedback_type} feedback for session {session_id}")
            return {
                "feedback_id": feedback_id,
                "status": "stored",
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            return None

    async def get_feedback_analytics(self) -> Dict:
        try:
            analytics = {
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "feedback_by_agent": {},
                "feedback_by_mode": {},
                "recent_feedback": [],
                "improvement_suggestions": []
            }
            all_feedback = []
            for session_feedback in self.feedback_data.values():
                all_feedback.extend(session_feedback)
            analytics["total_feedback"] = len(all_feedback)
            for feedback in all_feedback:
                if feedback["feedback_type"] == "positive":
                    analytics["positive_feedback"] += 1
                else:
                    analytics["negative_feedback"] += 1
                agent = feedback["agent_type"]
                if agent not in analytics["feedback_by_agent"]:
                    analytics["feedback_by_agent"][agent] = {"positive": 0, "negative": 0}
                analytics["feedback_by_agent"][agent][feedback["feedback_type"]] += 1
                mode = feedback["answer_mode"]
                if mode not in analytics["feedback_by_mode"]:
                    analytics["feedback_by_mode"][mode] = {"positive": 0, "negative": 0}
                analytics["feedback_by_mode"][mode][feedback["feedback_type"]] += 1
            sorted_feedback = sorted(all_feedback, key=lambda x: x["timestamp"], reverse=True)
            analytics["recent_feedback"] = sorted_feedback[:10]
            analytics["improvement_suggestions"] = self._generate_improvement_suggestions(analytics)
            return analytics
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {e}")
            return {
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "error": str(e)
            }

    def _generate_improvement_suggestions(self, analytics: Dict) -> List[str]:
        suggestions = []
        try:
            total = analytics["total_feedback"]
            if total == 0:
                return ["No feedback data available yet."]
            positive_rate = analytics["positive_feedback"] / total
            if positive_rate < 0.7:
                suggestions.append("Consider improving response quality - positive feedback rate is below 70%")
            for agent, feedback in analytics["feedback_by_agent"].items():
                agent_total = feedback["positive"] + feedback["negative"]
                if agent_total > 0:
                    agent_positive_rate = feedback["positive"] / agent_total
                    if agent_positive_rate < 0.6:
                        suggestions.append(f"Focus on improving {agent} agent responses")
            for mode, feedback in analytics["feedback_by_mode"].items():
                mode_total = feedback["positive"] + feedback["negative"]
                if mode_total > 0:
                    mode_positive_rate = feedback["positive"] / mode_total
                    if mode_positive_rate < 0.6:
                        suggestions.append(f"Improve {mode} answer mode responses")
            if not suggestions:
                suggestions.append("Great job! Feedback patterns look positive. Keep up the good work!")
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            suggestions.append("Unable to generate suggestions due to data processing error")
        return suggestions

    async def initialize_database(self):
        try:
            if not os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                conn.close()
                logger.info(f"Created database file: {self.db_path}")
            # Removed the problematic AsyncSqliteSaver code that was causing the error
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def _build_graph(self, checkpointer):
        workflow = StateGraph(MessagesState, config_schema=GraphConfig)
        
        workflow.add_node("general_agent", self.general_agent_node)
        workflow.add_node("document_agents", self.document_agents_node)
        workflow.add_node("analytics_agent", self.analytics_agent_node)
        workflow.add_node("research_agent", self.research_agents_node)
        workflow.add_node("coding_agents", self.coding_agent_node)
        
        workflow.add_conditional_edges(
            START,
            self.route_agent,
            {
                "general": "general_agent",
                "document": "document_agents",
                "analytics": "analytics_agent",
                "research": "research_agents",
                "coding": "coding_agents"
            }
        )
        
        workflow.add_edge("general_agent", END)
        workflow.add_edge("document_agents", END)
        workflow.add_edge("analytics_agent", END)
        workflow.add_edge("research_agents", END)
        workflow.add_edge("coding_agents", END)
        
        return workflow.compile(checkpointer=checkpointer)

    def route_agent(self, state: MessagesState):
        config = state.get("config", {})
        return config.get("agent_type", "general")

    def _extract_chat_title(self, first_message: str) -> str:
        title = first_message.strip()
        prefixes_to_remove = ['hi', 'hello', 'hey', 'can you', 'please', 'i need', 'help me']
        title_lower = title.lower()
        
        for prefix in prefixes_to_remove:
            if title_lower.startswith(prefix):
                title = title[len(prefix):].strip()
                break

        if any(word in title_lower for word in ['hpcl', 'hindustan petroleum']):
            return "HPCL Discussion"
        elif any(word in title_lower for word in ['document', 'pdf', 'file']):
            return "Document Processing"
        elif any(word in title_lower for word in ['data', 'analytics']):
            return "Analytics Discussion"
        elif any(word in title_lower for word in ['code', 'script', 'python']):
            return "Coding Assistance"
        elif title_lower.startswith('what'):
            return "General Question"
        elif title_lower.startswith('how'):
            return "How-to Question"
        
        if len(title) > 50:
            return title[:47] + "..."
        
        return title.title() if title else f"Chat - {datetime.now().strftime('%H:%M')}"

    async def _generate_smart_title(self, first_message: str) -> str:
        try:
            title_prompt = [
                {
                    "role": "system", 
                    "content": "Generate a concise 3-6 word title for a chat conversation based on the user's first message. Only return the title, nothing else."
                },
                {
                    "role": "user", 
                    "content": f"First message: {first_message}"
                }
            ]
            
            title_response = await groq_client.generate_response(title_prompt, stream=False)
            title = title_response.strip().replace('"', '').replace("'", '')
            
            if len(title) > 60:
                title = title[:57] + "..."
            
            return title if title else self._extract_chat_title(first_message)
            
        except Exception as e:
            logger.error(f"Error generating smart title: {e}")
            return self._extract_chat_title(first_message)

    def _get_conversation_context(self, conversation_history, current_message):
        context = {
            'user_name': None,
            'previous_topics': [],
            'relevant_history': []
        }
        
        for msg in conversation_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                
                if 'my name is' in content:
                    name_part = content.split('my name is')[1].strip().split()[0]
                    context['user_name'] = name_part.title()
                elif 'i am' in content and len(content.split()) <= 4:
                    name_part = content.split('i am')[1].strip().split()[0]
                    context['user_name'] = name_part.title()
                
                if any(word in content for word in ['hpcl', 'petroleum', 'oil', 'gas']):
                    context['previous_topics'].append('HPCL/Petroleum')
                if any(word in content for word in ['document', 'pdf', 'file']):
                    context['previous_topics'].append('Document Analysis')
                if any(word in content for word in ['data', 'analytics', 'report']):
                    context['previous_topics'].append('Data Analytics')
        
        return context

    def _get_feedback_context(self, session_id: str) -> str:
        try:
            session_feedback = self.feedback_data.get(session_id, [])
            if not session_feedback:
                return ""
            
            recent_feedback = session_feedback[-5:]
            negative_feedback = [f for f in recent_feedback if f["feedback_type"] == "negative"]
            
            if negative_feedback:
                feedback_context = "\nIMPORTANT: Previous responses received negative feedback. "
                feedback_context += "Focus on being more helpful, accurate, and comprehensive."
                return feedback_context
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting feedback context: {e}")
            return ""

    async def general_agent_node(self, state: MessagesState):
        try:
            system_content = f"""You are hpGPT, an AI assistant for HPCL (Hindustan Petroleum Corporation Limited). 

            Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            RESPONSE FORMATTING REQUIREMENTS:
            - Use **bold** for company names, important terms, and key concepts
            - Use *italics* for emphasis and technical terms
            - Use ## for main section headers
            - Use ### for subsection headers
            - Use bullet points (•) for lists and features
            - Add proper spacing between sections with line breaks
            - Use emojis sparingly but effectively (🚀, 📊, 💡, etc.)
            - Format numbers and data clearly
            - Make responses visually appealing and easy to scan

            CONTENT GUIDELINES:
            - Be helpful, accurate, and professional
            - Provide specific, actionable information about HPCL services
            - Structure complex information with clear headers and sections
            - Use examples and practical applications when possible
            - Always provide complete, comprehensive responses
            - NEVER stop mid-sentence or cut off responses
            - Remember and reference previous conversation context when relevant
            """
            
            messages = [{"role": "system", "content": system_content}]
            
            recent_messages = state["messages"][-10:] if len(state["messages"]) > 10 else state["messages"]
            
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    if msg.__class__.__name__ == 'HumanMessage':
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.__class__.__name__ == 'AIMessage':
                        messages.append({"role": "assistant", "content": msg.content})
            
            response = await groq_client.generate_response(messages, stream=False)
            return {"messages": [AIMessage(content=response)]}
            
        except Exception as e:
            logger.error(f"General agent error: {e}")
            return {"messages": [AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")]}

    async def document_agents_node(self, state: MessagesState):
        return await self.document_agents.process(state)

    async def analytics_agent_node(self, state: MessagesState):
        return await self.analytics_agent.process(state)

    async def research_agents_node(self, state: MessagesState):
        return await self.research_agents.process(state)

    async def coding_agent_node(self, state: MessagesState):
        return await self.coding_agents.process(state)

    # COMPLETELY FIXED CHAT METHOD - USES PROVIDED CONTENT DIRECTLY
    async def chat(self, message: str, session_id: str, agent_type: str = "general", files=None, answer_mode: str = "specific", should_stop=None):
        """Enhanced chat method that properly uses provided document content"""
        
        # Initialize session if new
        if session_id not in self.sessions:
            smart_title = await self._generate_smart_title(message)
            
            self.sessions[session_id] = {
                "title": smart_title,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "last_updated": datetime.now().isoformat()
            }
            self.conversations[session_id] = []
            self.save_data()
            logger.info(f"Created new session with smart title: {smart_title}")

        # Check for stop request during streaming
        def check_should_stop():
            return should_stop() if should_stop else False

        # FIXED: Use provided content directly - NO disk reading!
        original_message = message
        if agent_type == "document" and files:
            try:
                logger.info(f"Processing files: {files}")
                
                # Extract content directly from the provided files parameter
                file_content = None
                filename = "Unknown file"
                
                if isinstance(files, list) and files:
                    file_info = files[0]
                    if isinstance(file_info, dict):
                        # Content is already provided in the files parameter!
                        file_content = file_info.get('content', '')
                        filename = file_info.get('name', 'Unknown file')
                        logger.info(f"✅ Using provided content for file: {filename}")
                        logger.info(f"✅ Content length: {len(file_content)} characters")
                
                if file_content and len(file_content.strip()) > 0:
                    # Use the content that's already extracted
                    message = f"""DOCUMENT ANALYSIS REQUEST

FILENAME: {filename}

DOCUMENT CONTENT:
{file_content}

USER QUESTION: {original_message}

INSTRUCTIONS:
- Analyze the document content provided above
- Answer the user's question based specifically on what you find in the document
- Be detailed and specific in your analysis
- If the document contains structured information, organize your response clearly
- Reference specific details from the document to support your answer
"""
                    
                    logger.info(f"✅ Document processed successfully from provided content. Content length: {len(file_content)}")
                else:
                    message = f"No document content was provided or the content is empty. Original question: {original_message}"
                    logger.error(f"❌ No content found in files parameter: {files}")
                    
            except Exception as e:
                logger.error(f"❌ Document processing error: {e}")
                message = f"Error processing document: {str(e)}. Original question: {original_message}"

        # Map agent_type to display names and system prompts
        agent_configs = {
            "general": {
                "name": "General Assistant",
                "system_prompt": """You are hpGPT's General Assistant for HPCL (Hindustan Petroleum Corporation Limited). 
                
You can help with:
- General company information and services
- Basic queries about HPCL operations  
- Customer service questions
- General business guidance

Provide helpful, accurate, and professional responses about HPCL services and operations. Use proper formatting with **bold**, *italics*, and bullet points."""
            },
            "document": {
                "name": "Document Agent", 
                "system_prompt": """You are hpGPT's Document Agent for HPCL (Hindustan Petroleum Corporation Limited).

You are excellent at:
- Reading and analyzing documents (PDFs, reports, contracts, resumes)
- Extracting key information from documents
- Summarizing document contents comprehensively
- Answering questions based on document analysis
- Processing invoices, reports, and technical documents

When analyzing documents:
- Be specific about what you find in the document
- Base your answer entirely on the document content provided
- Organize information clearly with headers and bullet points
- Extract all relevant details (names, dates, qualifications, experience, etc.)
- If asked to summarize, provide a comprehensive overview of all key information

Use proper formatting with **bold**, *italics*, and bullet points."""
            },
            "analytics": {
                "name": "Analytics Agent",
                "system_prompt": """You are hpGPT's Analytics Agent for HPCL (Hindustan Petroleum Corporation Limited).

You specialize in:
- Data analysis and interpretation
- Creating insights from business data  
- Performance metrics and KPI analysis
- Market trend analysis
- Financial and operational analytics
- Generating reports and dashboards

Provide data-driven insights and analytical responses. Use proper formatting with **bold**, *italics*, and bullet points."""
            },
            "research": {
                "name": "Research Agent", 
                "system_prompt": """You are hpGPT's Research Agent for HPCL (Hindustan Petroleum Corporation Limited).

You excel at:
- Market research and industry analysis
- Competitive intelligence gathering
- Technology and innovation research
- Regulatory and policy research
- Industry trends and forecasting
- Strategic research for business decisions

Search, analyze, and explain complex topics clearly and comprehensively. Use proper formatting with **bold**, *italics*, and bullet points."""
            },
            "coding": {
                "name": "Coding Agent",
                "system_prompt": """You are hpGPT's Coding Agent for HPCL (Hindustan Petroleum Corporation Limited).

You are expert at:
- Writing clean, efficient Python code
- Debugging and troubleshooting scripts
- API development and integration
- Data processing and automation
- System integration solutions
- Technical problem solving

Provide working code solutions with clear explanations. Use proper formatting with **bold**, *italics*, and bullet points."""
            }
        }
        
        # Get agent config
        agent_config = agent_configs.get(agent_type, agent_configs["general"])
        agent_name = agent_config["name"]
        system_prompt = agent_config["system_prompt"]
        
        logger.info(f"Selected agent: {agent_name} for agent_type: {agent_type}")

        # Handle greetings with agent-specific responses
        message_clean = message.lower().strip().rstrip('!').rstrip('?').rstrip('.')
        simple_greetings = ['hi', 'hello', 'hey', 'how are you', 'good morning', 'good afternoon', 'good evening']
        
        if message_clean in simple_greetings:
            logger.info(f"🎯 Quick greeting response from {agent_name}")
            
            greeting_responses = {
                "General Assistant": f"Hello! 👋 I'm your **General Assistant** for **HPCL**. I can help with general queries, company information, and various tasks. How can I assist you today?",
                "Research Agent": f"Hello! 🔬 I'm your **Research Agent** for **HPCL**. I specialize in market research, industry analysis, and gathering insights. What would you like me to research for you?",
                "Coding Agent": f"Hello! 💻 I'm your **Coding Agent** for **HPCL**. I can help you write clean Python code, debug scripts, and solve programming challenges. What coding task can I help you with?",
                "Document Agent": f"Hello! 📄 I'm your **Document Agent** for **HPCL**. I'm excellent at reading, analyzing, and processing documents. Do you have any documents you'd like me to analyze?",
                "Analytics Agent": f"Hello! 📊 I'm your **Analytics Agent** for **HPCL**. I specialize in data analysis, creating insights, and generating reports. What data would you like me to analyze?"
            }
            
            quick_response = greeting_responses.get(agent_name, f"Hello! I'm the {agent_name} for HPCL. How can I help you?")
            
            # Stream the response with stop checking
            words = quick_response.split()
            for word in words:
                if check_should_stop():
                    break
                yield word + " "
                await asyncio.sleep(0.02)
            
            # Save to conversation history
            self.conversations[session_id].append({"role": "user", "content": original_message})
            self.conversations[session_id].append({"role": "assistant", "content": quick_response})
            self.sessions[session_id]["message_count"] += 2
            self.sessions[session_id]["last_updated"] = datetime.now().isoformat()
            self.save_data()
            
            return

        # For non-greetings, use groq_client with agent-specific system prompt
        try:
            # Create messages with agent-specific system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            logger.info(f"Calling {agent_name} with groq_client")
            
            # Use groq_client for streaming response
            complete_response = ""
            
            # Get streaming response from groq_client
            complete_response = ""

            if agent_type == "coding" and hasattr(self.coding_agents, "run"):
                # ✅ Use Gemini-powered CodingAgent
                logger.info(f"🧠 Using Gemini CodingAgent.run() for prompt: {message}")
                gemini_response = await self.coding_agents.run(message)
                complete_response = gemini_response # Wrap code for frontend formatting

                # Stream the code chunk by chunk
                for line in complete_response.splitlines():
                    if check_should_stop():
                        break
                    yield line + "\n"
                    await asyncio.sleep(0.01)

            elif agent_type == "analytics" and hasattr(self.analytics_agent, "run"):
                if not files or not isinstance(files, list) or not files[0].get("name"):
                    yield "❌ No Excel, PDF, or CSV file provided for analytics."
                    return

                try:
                    file_info = files[0]
                    filename = file_info["name"]
                    file_content = file_info["content"]

                    # Save uploaded file to temp dir
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, filename)

                    with open(file_path, "wb") as f:
                        f.write(file_content.encode("utf-8"))

                    logger.info(f"✅ Saved uploaded analytics file to {file_path}")
                    logger.info(f"📊 Using Gemini AnalyticsAgent for: {original_message}")

                    # 🔄 Run Gemini Analytics Agent with the file and prompt
                    gemini_response = await self.analytics_agent.run(file_info, original_message)

                    # ✅ Smart rendering based on content type (HTML graph or Python code)
                    if gemini_response.strip().startswith("<div") or "plotly-graph-div" in gemini_response:
                        complete_response = gemini_response  # ✅ Return raw HTML for frontend to render chart
                    else:
                        complete_response = f"```python\n{gemini_response.strip()}\n```"


                    for line in complete_response.splitlines():
                        if check_should_stop():
                            break
                        yield line + "\n"
                        await asyncio.sleep(0.01)

                except Exception as e:
                    logger.error(f"❌ Error in analytics: {e}")
                    yield f"❌ Error in analytics: {e}"


            elif agent_type == "research" and hasattr(self.research_agents, "run"):
                logger.info(f"🧠 Using Gemini ResearchAgent for: {original_message}")

                try:
                    gemini_response = await self.research_agents.run(original_message)
                    complete_response = gemini_response

                    for line in complete_response.splitlines():
                        if check_should_stop():
                            break
                        yield line + "\n"
                        await asyncio.sleep(0.01)

                except Exception as e:
                    logger.error(f"❌ Error in research agent: {e}")
                    yield f"❌ Error in research: {e}"


            elif agent_type == "document" and hasattr(self.document_agent, "run"):
                if files and isinstance(files, list) and files[0].get("name"):
                    logger.info(f"📄 Uploading and indexing document: {files[0]['name']}")
                    gemini_response = await self.document_agent.run(files[0])
                else:
                    logger.info(f"📘 Asking question to Gemini DocumentAgent: {original_message}")
                    gemini_response = await self.document_agent.run(original_message)

                complete_response = gemini_response
                for line in complete_response.splitlines():
                    if check_should_stop():
                        break
                    yield line + "\n"
                    await asyncio.sleep(0.01)





            
            else:
                # Fallback to Groq (general, document, etc.)
                stream_generator = await groq_client.generate_response(messages, stream=True)

                async for chunk in stream_generator:
                    if check_should_stop():
                        break
                    if chunk:
                        complete_response += chunk
                        yield chunk
                
            if not complete_response.strip():
                # Fallback if no response
                fallback_msg = f"I'm the {agent_name} for HPCL. I'm ready to help with your query about '{original_message}'. Please let me know what specific information you need."
                words = fallback_msg.split()
                for word in words:
                    if check_should_stop():
                        break
                    yield word + " "
                    await asyncio.sleep(0.02)
                complete_response = fallback_msg
            
            # Save conversation
            self.conversations[session_id].append({"role": "user", "content": original_message})
            self.conversations[session_id].append({"role": "assistant", "content": complete_response})
            self.sessions[session_id]["message_count"] += 2
            self.sessions[session_id]["last_updated"] = datetime.now().isoformat()
            self.save_data()
            
            logger.info(f"✅ Response from {agent_name} completed and saved for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error with {agent_name}: {e}")
            error_response = f"I'm the {agent_name} but encountered an error: {str(e)}. Please try again."
            
            # Stream the error response
            words = error_response.split()
            for word in words:
                if check_should_stop():
                    break
                yield word + " "
                await asyncio.sleep(0.02)
            
            # Save error response to conversation history
            self.conversations[session_id].append({"role": "user", "content": original_message})
            self.conversations[session_id].append({"role": "assistant", "content": error_response})
            self.sessions[session_id]["message_count"] += 2
            self.sessions[session_id]["last_updated"] = datetime.now().isoformat()
            self.save_data()

    async def get_limited_chat_history(self, session_id: str, limit: int):
        try:
            logger.info(f"Retrieving limited chat history for session: {session_id}, limit: {limit}")
            
            conversation = self.conversations.get(session_id, [])
            
            if conversation and limit > 0:
                limited_conversation = conversation[-limit:]
                formatted_history = [{"messages": limited_conversation}]
                logger.info(f"Retrieved {len(limited_conversation)} limited messages for session {session_id}")
                return formatted_history
            else:
                logger.info(f"No conversation found or invalid limit for session {session_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving limited chat history: {e}")
            return []

    async def get_total_message_count(self, session_id: str) -> int:
        try:
            conversation = self.conversations.get(session_id, [])
            total_count = len(conversation)
            logger.info(f"Total message count for session {session_id}: {total_count}")
            return total_count
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0

    async def get_conversation_stats(self, session_id: str) -> Dict:
        try:
            conversation = self.conversations.get(session_id, [])
            
            user_messages = sum(1 for msg in conversation if msg.get('role') == 'user')
            assistant_messages = sum(1 for msg in conversation if msg.get('role') == 'assistant')
            
            return {
                "total_messages": len(conversation),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "session_id": session_id
            }

    async def get_all_sessions(self) -> List[Dict]:
        try:
            sessions_list = []
            
            for session_id, session_data in self.sessions.items():
                sessions_list.append({
                    "session_id": session_id,
                    "title": session_data.get("title", f"Chat {session_id[:8]}"),
                    "created_at": session_data.get("created_at"),
                    "message_count": session_data.get("message_count", 0),
                    "last_updated": session_data.get("last_updated")
                })
            
            sessions_list.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            
            logger.info(f"Returning {len(sessions_list)} sessions")
            return sessions_list
                
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            return []

    async def get_chat_history(self, session_id: str):
        try:
            logger.info(f"Retrieving chat history for session: {session_id}")
            
            conversation = self.conversations.get(session_id, [])
            
            if conversation:
                formatted_history = [{"messages": conversation}]
                logger.info(f"Retrieved {len(conversation)} messages for session {session_id}")
                return formatted_history
            else:
                logger.info(f"No conversation found for session {session_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []

    async def delete_session(self, session_id: str) -> bool:
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.conversations:
                del self.conversations[session_id]
            if session_id in self.feedback_data:
                del self.feedback_data[session_id]
            
            self.save_data()
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                try:
                    cursor.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
                except sqlite3.OperationalError:
                    pass
                conn.commit()
                conn.close()
            except Exception as lg_error:
                logger.warning(f"LangGraph delete failed: {lg_error}")
            
            logger.info(f"Deleted session {session_id} including feedback data")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

hpgpt_graph = HPGPTGraph()
