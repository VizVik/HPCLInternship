from langchain_core.messages import SystemMessage, AIMessage
import re

class CodingAgent:
    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.supported_languages = [
            'python', 'javascript', 'java', 'c++', 'c#', 'sql', 
            'html', 'css', 'bash', 'powershell', 'r', 'matlab',
            'typescript', 'go', 'rust', 'php', 'ruby'
        ]
    
    async def process(self, state):
        messages = [
            {
                "role": "system",
                "content": """You are a specialized coding agent for HPCL (Hindustan Petroleum Corporation Limited). 
                You can write code, debug programs, create scripts for automation, 
                develop APIs, create data analysis scripts, and provide technical solutions.
                Always provide clean, well-commented, production-ready code with explanations.
                Focus on enterprise-level solutions suitable for HPCL's business needs."""
            }
        ]
        
        # Add user messages from state
        for msg in state["messages"]:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == 'HumanMessage':
                    messages.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif msg.__class__.__name__ == 'AIMessage':
                    messages.append({
                        "role": "assistant", 
                        "content": msg.content
                    })
        
        response = await self.groq_client.generate_response(messages, stream=False)
        return {"messages": [AIMessage(content=response)]}
    
    async def generate_code(self, task_description, language="python", complexity="intermediate"):
        """Generate code based on task description"""
        complexity_instructions = {
            "basic": "Write simple, beginner-friendly code with extensive comments and explanations",
            "intermediate": "Write efficient code with good practices, moderate complexity, and clear documentation",
            "advanced": "Write optimized, enterprise-level code with advanced patterns, error handling, and scalability"
        }
        
        prompt = f"""
        Create a {complexity} {language} solution for the following task:
        
        **Task:** {task_description}
        
        **Requirements:**
        - {complexity_instructions[complexity]}
        - Include comprehensive error handling
        - Add detailed docstrings and comments
        - Follow best practices for {language}
        - Make it production-ready for HPCL enterprise systems
        - Include input validation and security considerations
        
        **Please provide:**
        1. **Complete Code Solution**
        2. **Code Explanation** (how it works)
        3. **Usage Examples** (with sample inputs/outputs)
        4. **Dependencies** (required packages/libraries)
        5. **Installation Instructions**
        6. **Testing Recommendations**
        
        Format the code with proper syntax highlighting using markdown code blocks.
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def debug_code(self, code_snippet, error_description="", language="python"):
        """Debug and fix code issues"""
        prompt = f"""
        Debug and fix the following {language} code:
        
        **Original Code:**
        ```
        {code_snippet}
        ```
        
        **Error/Issue Description:** {error_description}
        
        **Please provide:**
        
        ## 1. Issue Analysis
        - Identify all problems in the code
        - Explain why each issue occurs
        - Categorize issues (syntax, logic, performance, security)
        
        ## 2. Fixed Code
        ```
        [Provide the corrected code here]
        ```
        
        ## 3. Explanation of Fixes
        - Detail what was changed and why
        - Explain the reasoning behind each fix
        
        ## 4. Best Practices
        - Suggest improvements for code quality
        - Recommend coding standards to follow
        
        ## 5. Testing Strategy
        - Suggest test cases to verify the fix
        - Recommend unit tests to prevent similar issues
        
        ## 6. Prevention Tips
        - How to avoid similar issues in the future
        - Code review checklist items
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def create_api_endpoint(self, endpoint_description, framework="fastapi"):
        """Create API endpoints for various frameworks"""
        framework_templates = {
            "fastapi": f"""
            Create a FastAPI endpoint for: {endpoint_description}
            
            **Include:**
            - Complete route definition with proper HTTP methods
            - Pydantic models for request/response validation
            - Comprehensive error handling with appropriate HTTP status codes
            - Input validation and sanitization
            - Documentation strings for auto-generated API docs
            - Authentication/authorization if needed
            - Logging for monitoring and debugging
            
            **Provide:**
            1. **Complete FastAPI Code**
            2. **Pydantic Models**
            3. **Error Handling**
            4. **Usage Examples**
            5. **API Documentation**
            """,
            
            "flask": f"""
            Create a Flask endpoint for: {endpoint_description}
            
            **Include:**
            - Route definition with appropriate methods
            - Request handling and validation
            - JSON response formatting
            - Error handling with proper status codes
            - Input validation
            - CORS handling if needed
            
            **Provide:**
            1. **Complete Flask Code**
            2. **Request/Response Examples**
            3. **Error Handling**
            4. **Testing Examples**
            """,
            
            "django": f"""
            Create a Django view for: {endpoint_description}
            
            **Include:**
            - View function or class-based view
            - URL pattern configuration
            - Model integration (if applicable)
            - Form handling and validation
            - Template context (if needed)
            - Serializers for API responses
            
            **Provide:**
            1. **View Code**
            2. **URL Configuration**
            3. **Model Definitions**
            4. **Serializers**
            5. **Usage Examples**
            """
        }
        
        template = framework_templates.get(framework, framework_templates["fastapi"])
        messages = [{"role": "user", "content": template}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def create_automation_script(self, automation_task, platform="python"):
        """Create automation scripts for various tasks"""
        prompt = f"""
        Create an automation script for: {automation_task}
        
        **Platform:** {platform}
        **Target Environment:** HPCL Enterprise Systems
        
        **The script should include:**
        
        ## 1. Core Functionality
        - Robust implementation of the automation task
        - Handle edge cases and error scenarios
        - Configurable parameters for flexibility
        
        ## 2. Enterprise Features
        - Comprehensive logging for audit trails
        - Configuration file support
        - Error recovery and retry mechanisms
        - Performance monitoring
        - Security considerations
        
        ## 3. Monitoring & Alerts
        - Progress tracking
        - Success/failure notifications
        - Performance metrics collection
        
        **Deliverables:**
        
        ### Main Script
        ```
        [Complete automation script]
        ```
        
        ### Configuration File
        ```
        [Configuration template]
        ```
        
        ### Setup Instructions
        - Installation requirements
        - Environment setup
        - Configuration steps
        
        ### Scheduling Guide
        - Cron job examples (Linux)
        - Task Scheduler setup (Windows)
        - Monitoring recommendations
        
        ### Usage Examples
        - Command-line usage
        - Configuration examples
        - Troubleshooting guide
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def create_data_analysis_script(self, data_description, analysis_type="exploratory"):
        """Create data analysis scripts for HPCL business data"""
        analysis_templates = {
            "exploratory": f"""
            Create a comprehensive Python script for exploratory data analysis of: {data_description}
            
            **Analysis Components:**
            
            ## 1. Data Loading & Inspection
            - Multiple data source support (CSV, Excel, Database)
            - Data quality assessment
            - Missing value analysis
            - Data type validation
            
            ## 2. Statistical Analysis
            - Descriptive statistics
            - Distribution analysis
            - Correlation analysis
            - Outlier detection
            
            ## 3. Visualization
            - Interactive plots using Plotly
            - Statistical charts
            - Business dashboards
            - Export capabilities
            
            ## 4. Business Insights
            - KPI calculations relevant to HPCL
            - Trend analysis
            - Performance metrics
            - Actionable recommendations
            """,
            
            "predictive": f"""
            Create a machine learning analysis script for: {data_description}
            
            **ML Pipeline Components:**
            
            ## 1. Data Preprocessing
            - Feature engineering
            - Data cleaning and transformation
            - Train/validation/test splits
            - Scaling and normalization
            
            ## 2. Model Development
            - Multiple algorithm comparison
            - Hyperparameter tuning
            - Cross-validation
            - Model selection
            
            ## 3. Evaluation & Validation
            - Performance metrics
            - Model interpretability
            - Business impact assessment
            - Deployment readiness
            
            ## 4. Production Code
            - Model serialization
            - Prediction pipeline
            - Monitoring hooks
            - API integration ready
            """,
            
            "reporting": f"""
            Create an automated reporting script for: {data_description}
            
            **Reporting Features:**
            
            ## 1. Data Processing
            - Automated data extraction
            - Data validation and cleaning
            - Calculation of business metrics
            
            ## 2. Report Generation
            - Professional PDF reports
            - Interactive HTML dashboards
            - Excel workbooks with charts
            - Email distribution
            
            ## 3. Scheduling & Automation
            - Automated report generation
            - Email notifications
            - Error handling and alerts
            - Performance monitoring
            """
        }
        
        template = analysis_templates.get(analysis_type, analysis_templates["exploratory"])
        messages = [{"role": "user", "content": template}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def optimize_code(self, code_snippet, optimization_focus="performance", language="python"):
        """Optimize existing code for better performance, security, or maintainability"""
        optimization_prompts = {
            "performance": "Optimize this code for better performance, speed, and efficiency",
            "memory": "Optimize this code for better memory usage and resource management",
            "security": "Enhance code security and implement security best practices",
            "maintainability": "Improve code readability, maintainability, and documentation"
        }
        
        prompt = f"""
        {optimization_prompts[optimization_focus]} for the following {language} code:
        
        **Original Code:**
        ```
        {code_snippet}
        ```
        
        **Optimization Focus:** {optimization_focus}
        
        **Please provide:**
        
        ## 1. Optimized Code
        ```
        [Provide the optimized version]
        ```
        
        ## 2. Optimization Analysis
        - **Changes Made:** Detailed list of optimizations
        - **Performance Impact:** Expected improvements
        - **Trade-offs:** Any compromises made
        - **Metrics:** Before/after comparisons where applicable
        
        ## 3. Best Practices Applied
        - Coding standards implemented
        - Design patterns used
        - Security measures added
        
        ## 4. Testing Recommendations
        - Unit tests for optimized code
        - Performance benchmarking
        - Security testing approaches
        
        ## 5. Monitoring & Maintenance
        - Key metrics to monitor
        - Maintenance recommendations
        - Future optimization opportunities
        
        Focus on enterprise-grade solutions suitable for HPCL's production environment.
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def create_database_script(self, database_task, db_type="postgresql"):
        """Create database scripts for various database operations"""
        prompt = f"""
        Create {db_type} database scripts for: {database_task}
        
        **Enterprise Database Requirements:**
        
        ## 1. Schema Design
        - Normalized table structures
        - Proper indexing strategy
        - Foreign key relationships
        - Data integrity constraints
        
        ## 2. Performance Optimization
        - Query optimization
        - Index recommendations
        - Partitioning strategies
        - Caching considerations
        
        ## 3. Security & Compliance
        - Access control and permissions
        - Data encryption at rest
        - Audit trail implementation
        - Backup and recovery procedures
        
        ## 4. Monitoring & Maintenance
        - Performance monitoring queries
        - Health check scripts
        - Maintenance procedures
        - Capacity planning queries
        
        **Deliverables:**
        
        ### DDL Scripts
        ```
        [Table creation and schema scripts]
        ```
        
        ### DML Scripts
        ```
        [Data manipulation scripts]
        ```
        
        ### Stored Procedures/Functions
        ```
        [Business logic implementation]
        ```
        
        ### Performance Scripts
        ```
        [Optimization and monitoring queries]
        ```
        
        ### Documentation
        - Schema documentation
        - Usage guidelines
        - Performance tuning guide
        - Backup/recovery procedures
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
