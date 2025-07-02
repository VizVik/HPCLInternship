import os
import asyncio
from groq import Groq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        logger.info(f"GroqClient initialized with model: {self.model}")
    
    def _convert_langchain_messages(self, messages):
        converted_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, dict):
                converted_messages.append(msg)
            else:
                converted_messages.append({"role": "user", "content": str(msg)})
        
        return converted_messages
    
    def _is_simple_greeting(self, message):
        """Enhanced greeting detection with better pattern matching"""
        if not message:
            return False
            
        # Clean the message
        message_clean = message.lower().strip().rstrip('!').rstrip('?').rstrip('.')
        logger.info(f"Checking greeting for cleaned message: '{message_clean}'")
        
        # Direct matches
        simple_greetings = [
            'hi', 'hello', 'hey', 'how are you', 'good morning', 
            'good afternoon', 'good evening', 'what is your purpose', 
            'who are you', 'what can you do', 'help', 'what is hpcl'
        ]
        
        # Check direct matches first
        if message_clean in simple_greetings:
            logger.info(f"✅ Direct greeting match: {message_clean}")
            return True
        
        # Pattern-based matching
        simple_patterns = [
            r'^hi$', r'^hello$', r'^hey$', r'^how are you$', r'^good morning$', 
            r'^good afternoon$', r'^good evening$', r'^what is your purpose$', 
            r'^who are you$', r'^what can you do$', r'^help$', r'^what is hpcl$'
        ]
        
        for pattern in simple_patterns:
            if re.match(pattern, message_clean):
                logger.info(f"✅ Pattern greeting match: {message_clean} -> {pattern}")
                return True
        
        logger.info(f"❌ No greeting match for: {message_clean}")
        return False
    
    def _get_quick_response(self, message):
        """Get quick response with enhanced matching"""
        if not message:
            return None
            
        # Clean the message for lookup
        message_clean = message.lower().strip().rstrip('!').rstrip('?').rstrip('.')
        
        quick_responses = {
            'hi': "Hello! 👋 I'm **hpGPT**, your AI assistant for **HPCL**.\n\nHow can I help you today?",
            'hello': "Hi there! 🌟 I'm **hpGPT**, ready to assist you with:\n\n• **HPCL-related queries**\n• **Document analysis**\n• **Research & insights**\n• **Coding & automation**\n\nWhat would you like to explore?",
            'hey': "Hey! 🚀 I'm **hpGPT**, your dedicated HPCL AI assistant.\n\nWhat can I help you with today?",
            'how are you': "I'm doing great, thank you for asking! 😊 I'm **hpGPT**, your AI assistant for **HPCL**, and I'm here and ready to help you with:\n\n• **HPCL operations and services**\n• **Document analysis and processing**\n• **Research and market insights**\n• **Technical assistance and coding**\n• **Data analytics and reporting**\n\nHow are you doing today? What can I assist you with?",
            'good morning': "Good morning! ☀️ I'm **hpGPT**, your AI assistant for **HPCL**.\n\nReady to help you start your day productively! What's on your agenda?",
            'good afternoon': "Good afternoon! 🌅 I'm **hpGPT**, here to assist you with any **HPCL-related** tasks or questions.\n\nHow can I support you today?",
            'good evening': "Good evening! 🌆 I'm **hpGPT**, your HPCL AI assistant.\n\nHow can I help you wind down with some productive work?",
            'what is your purpose': "I'm **hpGPT**, an AI assistant specifically designed for **HPCL (Hindustan Petroleum Corporation Limited)**.\n\n## My Core Capabilities:\n\n### 📄 **Document Analysis**\n• PDF processing and summarization\n• Invoice and report analysis\n• Contract review and insights\n\n### 📊 **Data Analytics**\n• Business intelligence and insights\n• Performance reporting\n• Trend analysis\n\n### 🔬 **Research & Intelligence**\n• Market research and competitor analysis\n• Industry trends and forecasting\n• Strategic insights\n\n### 💻 **Coding & Automation**\n• Script generation and debugging\n• API development\n• Process automation\n\n### ❓ **General Assistance**\n• HPCL-related queries\n• Technical support\n• Strategic guidance\n\n*What would you like me to help you with?*",
            'who are you': "I'm **hpGPT** 🤖, your dedicated AI assistant for **HPCL**.\n\n**My Mission:** To help HPCL professionals with:\n• Document analysis & processing\n• Research & market insights\n• Coding & automation solutions\n• Strategic decision support\n\n*Think of me as your intelligent workplace companion!*",
            'what can you do': "Great question! Here's what I can help you with:\n\n## 🎯 **Core Services**\n\n### 📄 **Document Processing**\n• **PDF Analysis** - Extract insights from reports\n• **Invoice Processing** - Automate data extraction\n• **Contract Review** - Identify key terms and risks\n\n### 📊 **Business Analytics**\n• **Performance Dashboards** - KPI tracking and visualization\n• **Trend Analysis** - Market and operational insights\n• **Predictive Analytics** - Forecasting and planning\n\n### 🔍 **Research & Intelligence**\n• **Market Research** - Competitor and industry analysis\n• **Strategic Planning** - Data-driven recommendations\n• **Regulatory Updates** - Compliance and policy insights\n\n### ⚙️ **Automation & Development**\n• **Script Generation** - Python, SQL, and more\n• **API Development** - Custom integrations\n• **Process Automation** - Workflow optimization\n\n### 💡 **Strategic Support**\n• **Decision Analysis** - Data-backed recommendations\n• **Risk Assessment** - Identify and mitigate risks\n• **Innovation Ideas** - Technology and process improvements\n\n*What specific area interests you most?*",
            'help': "I'm here to help! 🆘 I'm **hpGPT**, your comprehensive HPCL AI assistant.\n\n## 🚀 **Quick Start Guide**\n\n### **Popular Commands:**\n• *\"Analyze this document\"* - Upload PDFs for analysis\n• *\"Research market trends\"* - Get industry insights\n• *\"Generate a Python script\"* - Coding assistance\n• *\"What's new in petroleum industry?\"* - Latest updates\n\n### **Pro Tips:**\n• Be specific with your requests\n• Upload files for detailed analysis\n• Ask follow-up questions for deeper insights\n\n*Just ask me anything - I'm here to make your work easier!*",
            'what is hpcl': "**HPCL (Hindustan Petroleum Corporation Limited)** 🏢\n\n## **Company Overview**\n\n### **Key Facts:**\n• **Founded:** 1974\n• **Headquarters:** Mumbai, India\n• **Industry:** Oil & Gas, Petroleum Refining\n• **Employees:** 10,000+ professionals\n\n### **Core Business Areas:**\n\n#### 🏭 **Refining Operations**\n• **Refineries:** Mumbai, Visakhapatnam, and more\n• **Capacity:** Millions of metric tons annually\n• **Products:** Petrol, diesel, aviation fuel, LPG\n\n#### ⛽ **Marketing & Distribution**\n• **Retail Outlets:** Thousands across India\n• **Brand:** HP (Hindustan Petroleum)\n• **Services:** Fuel, lubricants, convenience stores\n\n#### 🔬 **Innovation & Technology**\n• **R&D Centers:** Advanced research facilities\n• **Green Energy:** Renewable energy initiatives\n• **Digital Transformation:** Modern technology adoption\n\n### **Strategic Focus:**\n• **Sustainability** - Environmental responsibility\n• **Innovation** - Cutting-edge technology\n• **Customer Excellence** - Superior service delivery\n• **Growth** - Expanding market presence\n\n*I'm here to help you with any HPCL-related questions or tasks!*"
        }
        
        response = quick_responses.get(message_clean, None)
        if response:
            logger.info(f"✅ Found quick response for: {message_clean}")
        else:
            logger.warning(f"❌ No quick response found for: {message_clean}")
            logger.info(f"Available keys: {list(quick_responses.keys())}")
        
        return response
    
    async def generate_response_stream(self, messages):
        """Generate streaming response with guaranteed completion"""
        try:
            groq_messages = self._convert_langchain_messages(messages)
            
            # Enhanced greeting detection with debugging
            if len(groq_messages) >= 2:
                last_user_message = groq_messages[-1].get('content', '')
                logger.info(f"🔍 Checking greeting for message: '{last_user_message}'")
                
                if self._is_simple_greeting(last_user_message):
                    logger.info(f"🎯 Detected as simple greeting: {last_user_message}")
                    quick_response = self._get_quick_response(last_user_message)
                    if quick_response:
                        logger.info(f"🚀 Using quick response for: {last_user_message}")
                        # Stream the quick response word by word
                        words = quick_response.split()
                        for word in words:
                            yield word + " "
                            await asyncio.sleep(0.02)
                        return
                    else:
                        logger.warning(f"⚠️ No quick response found for greeting: {last_user_message}")
                else:
                    logger.info(f"📝 Not detected as greeting: {last_user_message}")
            
            logger.info(f"🔄 Starting Groq stream for complex query")
            
            # Use Groq's streaming API with enhanced settings
            stream_response = self.client.chat.completions.create(
                messages=groq_messages,
                model=self.model,
                stream=True,
                temperature=0.7,
                max_tokens=8192,  # Increased for longer responses
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=None
            )
            
            complete_response = ""
            chunk_count = 0
            last_chunk_time = asyncio.get_event_loop().time()
            
            for chunk in stream_response:
                try:
                    current_time = asyncio.get_event_loop().time()
                    
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            complete_response += content
                            chunk_count += 1
                            last_chunk_time = current_time
                            
                            # Very fast streaming for better UX
                            yield content
                            await asyncio.sleep(0.001)  # 1ms delay - very fast!
                        
                        # Check if stream is complete
                        if chunk.choices[0].finish_reason:
                            logger.info(f"✅ Stream completed. Reason: {chunk.choices[0].finish_reason}")
                            logger.info(f"📊 Total chunks: {chunk_count}, Length: {len(complete_response)}")
                            break
                    
                    # Timeout check - 45 seconds max
                    if current_time - last_chunk_time > 45:
                        logger.warning("⏰ Stream timeout - forcing completion")
                        break
                        
                except Exception as chunk_error:
                    logger.error(f"❌ Error processing chunk: {chunk_error}")
                    continue
            
            # Ensure we have a response
            if not complete_response.strip():
                yield "I apologize, but I didn't receive a complete response. Please try asking your question again, perhaps in a different way."
            elif len(complete_response) < 10:
                yield "\n\n*If this response seems incomplete, please let me know and I'll provide more details.*"
                    
        except Exception as e:
            logger.error(f"❌ Groq API streaming error: {e}")
            yield f"I encountered an error while processing your request: {str(e)}. Please try again."
    
    async def generate_response(self, messages, stream=True):
        try:
            groq_messages = self._convert_langchain_messages(messages)
            logger.info(f"📤 Sending {len(groq_messages)} messages to Groq API")
            
            if stream:
                return self.generate_response_stream(messages)
            else:
                # Check for quick responses first
                if len(groq_messages) >= 2:
                    last_user_message = groq_messages[-1].get('content', '')
                    if self._is_simple_greeting(last_user_message):
                        quick_response = self._get_quick_response(last_user_message)
                        if quick_response:
                            return quick_response
                
                response = self.client.chat.completions.create(
                    messages=groq_messages,
                    model=self.model,
                    temperature=0.7,
                    max_tokens=8192,
                    top_p=0.9
                )
                content = response.choices[0].message.content
                logger.info(f"📥 Non-streaming response: {len(content)} characters")
                return content
                
        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            error_msg = f"I encountered an error: {str(e)}. Please try again."
            
            if stream:
                async def error_generator():
                    yield error_msg
                return error_generator()
            else:
                return error_msg

groq_client = GroqClient()
