from langchain_core.messages import SystemMessage, AIMessage

class ResearchAgent:
    def __init__(self, groq_client):
        self.groq_client = groq_client
    
    async def process(self, state):
        messages = [
            {
                "role": "system",
                "content": """You are a specialized research agent for HPCL (Hindustan Petroleum Corporation Limited). 
                You can conduct market research, industry analysis, competitor analysis, 
                petroleum industry trends, regulatory research, and provide comprehensive 
                research reports. Always provide well-structured, factual, and actionable insights 
                relevant to the petroleum and energy sector."""
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
    
    async def conduct_market_research(self, topic):
        """Conduct comprehensive market research analysis"""
        prompt = f"""
        Conduct a comprehensive market research analysis on: {topic}
        
        Please provide a detailed analysis including:
        
        ## Market Overview
        - Market size and current valuation
        - Growth trends and projections
        - Key market segments
        
        ## Market Drivers
        - Primary growth drivers
        - Emerging opportunities
        - Technology trends
        
        ## Competitive Landscape
        - Major market players
        - Market share distribution
        - Competitive positioning
        
        ## HPCL Specific Insights
        - Relevance to HPCL's business
        - Strategic opportunities
        - Potential challenges
        
        ## Recommendations
        - Strategic recommendations for HPCL
        - Market entry strategies
        - Investment opportunities
        
        Focus on petroleum industry context and provide actionable insights.
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def analyze_competitors(self, competitor_list=None):
        """Perform detailed competitor analysis"""
        competitors = competitor_list or [
            "Indian Oil Corporation (IOCL)",
            "Bharat Petroleum Corporation Limited (BPCL)", 
            "Oil and Natural Gas Corporation (ONGC)",
            "Reliance Industries Limited",
            "Nayara Energy"
        ]
        
        prompt = f"""
        Perform a detailed competitor analysis for HPCL against these key competitors:
        {', '.join(competitors)}
        
        ## Competitive Analysis Framework
        
        ### Market Position
        - Market share comparison
        - Revenue and profitability analysis
        - Geographic presence
        
        ### Strengths & Weaknesses
        For each competitor, analyze:
        - Core strengths
        - Key weaknesses
        - Competitive advantages
        
        ### Strategic Initiatives
        - Recent strategic moves
        - Investment in new technologies
        - Expansion plans
        
        ### HPCL Positioning
        - HPCL's competitive position
        - Unique value propositions
        - Areas for improvement
        
        ### Strategic Recommendations
        - How HPCL can strengthen its position
        - Opportunities to exploit competitor weaknesses
        - Defensive strategies
        
        Provide specific, actionable insights for HPCL's strategic planning.
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def research_industry_trends(self, focus_area="petroleum"):
        """Research current industry trends and future outlook"""
        prompt = f"""
        Research and analyze current industry trends in the {focus_area} sector:
        
        ## Current Industry Status
        - Market conditions and dynamics
        - Recent developments
        - Key performance indicators
        
        ## Emerging Trends
        - Technology disruptions
        - Sustainability initiatives
        - Digital transformation
        - Energy transition trends
        
        ## Regulatory Environment
        - Current regulations affecting the industry
        - Upcoming policy changes
        - Compliance requirements
        
        ## Future Outlook
        - 5-year industry projections
        - Potential disruptions
        - Growth opportunities
        
        ## Impact on HPCL
        - How trends affect HPCL specifically
        - Strategic implications
        - Adaptation strategies needed
        
        ## Recommendations
        - Strategic priorities for HPCL
        - Investment areas
        - Risk mitigation strategies
        
        Focus on actionable insights for HPCL's strategic planning.
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
    
    async def generate_research_report(self, research_topics, report_type="executive"):
        """Generate structured research reports"""
        if report_type == "executive":
            prompt = f"""
            Create an executive research summary covering: {', '.join(research_topics)}
            
            # Executive Research Summary
            
            ## Key Findings
            - [Top 3-5 critical insights]
            
            ## Strategic Implications for HPCL
            - [How findings impact HPCL's business]
            
            ## Market Opportunities
            - [Specific opportunities identified]
            
            ## Risk Assessment
            - [Key risks and mitigation strategies]
            
            ## Recommendations
            1. [Immediate actions (0-6 months)]
            2. [Medium-term strategies (6-18 months)]
            3. [Long-term initiatives (18+ months)]
            
            ## Next Steps
            - [Specific action items with timelines]
            
            Keep it concise but comprehensive, focusing on actionable insights.
            """
        
        elif report_type == "detailed":
            prompt = f"""
            Create a comprehensive research report on: {', '.join(research_topics)}
            
            # Comprehensive Research Report
            
            ## Executive Summary
            [2-3 paragraph overview of key findings]
            
            ## Research Methodology
            [Approach and data sources used]
            
            ## Detailed Analysis
            ### Market Analysis
            [In-depth market insights]
            
            ### Competitive Intelligence
            [Detailed competitor analysis]
            
            ### Trend Analysis
            [Industry trends and implications]
            
            ## Strategic Framework
            ### SWOT Analysis for HPCL
            - Strengths
            - Weaknesses  
            - Opportunities
            - Threats
            
            ## Recommendations
            ### Strategic Priorities
            [Detailed strategic recommendations]
            
            ### Implementation Roadmap
            [Step-by-step implementation plan]
            
            ## Appendices
            [Supporting data and references]
            """
        
        else:  # presentation format
            prompt = f"""
            Create a presentation-style research summary for: {', '.join(research_topics)}
            
            # Slide 1: Research Overview
            - **Objective:** [Research goals]
            - **Scope:** [Areas covered]
            - **Key Questions:** [Research questions addressed]
            
            # Slide 2: Market Landscape
            - **Market Size:** [Current market data]
            - **Growth Rate:** [Growth projections]
            - **Key Players:** [Major competitors]
            
            # Slide 3: Key Findings
            - **Finding 1:** [Critical insight]
            - **Finding 2:** [Important trend]
            - **Finding 3:** [Strategic opportunity]
            
            # Slide 4: HPCL Position
            - **Current Position:** [HPCL's market standing]
            - **Competitive Advantage:** [Unique strengths]
            - **Growth Opportunities:** [Potential areas]
            
            # Slide 5: Strategic Recommendations
            - **Short-term (0-6 months):** [Immediate actions]
            - **Medium-term (6-18 months):** [Strategic initiatives]
            - **Long-term (18+ months):** [Future positioning]
            
            # Slide 6: Implementation Plan
            - **Priority Actions:** [Top 3 priorities]
            - **Timeline:** [Implementation schedule]
            - **Success Metrics:** [KPIs to track]
            """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.groq_client.generate_response(messages, stream=False)
