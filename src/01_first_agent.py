"""
==========================================================
DAY 1: Your First AI Agent
==========================================================
AI Agents Bootcamp | https://teodoracoach.substack.com

This script demonstrates the 5 building blocks of AI agents:
1. 🎯 GOAL      - What the agent is trying to accomplish
2. 🧠 REASONING - LLM that plans and decides  
3. 🔧 TOOLS     - Capabilities to take actions
4. 💾 MEMORY    - Store context and results
5. 🔄 FEEDBACK  - Self-evaluation loop

Run: python src/01_first_agent.py
==========================================================
"""

import os
import sys
from datetime import datetime
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ==========================================================
# CONFIGURATION
# ==========================================================

class AgentConfig:
    """Configuration for the research agent."""
    
    # Model settings
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Output settings
    OUTPUT_DIR = "outputs"
    
    # System prompt for research
    RESEARCH_PROMPT = """You are a company research agent helping someone prepare for a job interview.

Your job is to provide comprehensive, accurate, and actionable information.

For each company, provide the following sections:

## 1. Company Overview
- What they do (core business/product)
- Founded (year) and headquarters location
- Size (employees, if known)
- Key leadership

## 2. Products & Services
- Main offerings
- Target customers
- Competitive advantages

## 3. Culture & Values
- Company mission/vision
- Core values
- Work environment reputation
- Notable perks or practices

## 4. Recent Developments (Last 12-18 months)
- Major news, funding, acquisitions
- Product launches
- Leadership changes
- Challenges or controversies

## 5. Interview Preparation Tips
- What they likely look for in candidates
- Common interview themes/questions
- How to demonstrate cultural fit
- Red flags to avoid

Be concise but thorough. Use bullet points for clarity.
Focus on information that would help someone stand out in an interview."""

    VALIDATION_PROMPT = """You are a quality checker for company research reports.

Check if the research covers ALL of these required sections:
1. Company Overview (what they do, when founded, leadership)
2. Products & Services  
3. Culture & Values
4. Recent Developments
5. Interview Preparation Tips

Respond with EXACTLY one of these formats:
- "COMPLETE" if all sections are adequately covered
- "INCOMPLETE: [list what's missing]" if sections are missing or too brief"""


# ==========================================================
# THE AGENT
# ==========================================================

class CompanyResearchAgent:
    """
    An AI agent that researches companies for interview preparation.
    
    This demonstrates the core agent pattern:
    - Receives a goal (research company X)
    - Uses reasoning (LLM) to gather information
    - Stores results in memory
    - Validates output quality (feedback loop)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the agent with OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in .env file "
                "or pass api_key parameter."
            )
        
        # Initialize the LLM client (BUILDING BLOCK 2: REASONING)
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize memory (BUILDING BLOCK 4: MEMORY)
        self.memory = {
            "researched_companies": [],
            "current_research": None,
            "history": []
        }
    
    def research(self, company_name: str) -> dict:
        """
        Research a company and return structured results.
        
        Args:
            company_name: Name of the company to research
            
        Returns:
            Dictionary containing research results and metadata
        """
        print(f"\n{'='*60}")
        print(f"🤖 AGENT ACTIVATED")
        print(f"{'='*60}")
        print(f"📎 Goal: Research '{company_name}' for interview prep")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # BUILDING BLOCK 1: GOAL
        goal = f"Research {company_name} for job interview preparation"
        
        # BUILDING BLOCK 2: REASONING (via LLM)
        print("📊 Step 1/3: Gathering company information...")
        research_output = self._execute_research(company_name)
        
        # BUILDING BLOCK 4: MEMORY - Store results
        self.memory["current_research"] = {
            "company": company_name,
            "goal": goal,
            "research": research_output,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_validation"
        }
        
        # BUILDING BLOCK 5: FEEDBACK LOOP - Validate quality
        print("✅ Step 2/3: Validating research completeness...")
        validation_result = self._validate_research(research_output)
        
        # Update status based on validation
        if "INCOMPLETE" in validation_result:
            self.memory["current_research"]["status"] = "needs_review"
            self.memory["current_research"]["validation_notes"] = validation_result
            print(f"⚠️  Validation: {validation_result}")
        else:
            self.memory["current_research"]["status"] = "complete"
            print("✅ Validation: Research is complete")
        
        # Add to history
        self.memory["researched_companies"].append(company_name)
        self.memory["history"].append(self.memory["current_research"].copy())
        
        print("\n📝 Step 3/3: Preparing output...")
        
        return self.memory["current_research"]
    
    def _execute_research(self, company_name: str) -> str:
        """Execute the research using the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=AgentConfig.MODEL,
                messages=[
                    {"role": "system", "content": AgentConfig.RESEARCH_PROMPT},
                    {"role": "user", "content": f"Research this company: {company_name}"}
                ],
                temperature=AgentConfig.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during research: {str(e)}"
    
    def _validate_research(self, research: str) -> str:
        """Validate the research output for completeness."""
        try:
            response = self.client.chat.completions.create(
                model=AgentConfig.MODEL,
                messages=[
                    {"role": "system", "content": AgentConfig.VALIDATION_PROMPT},
                    {"role": "user", "content": research}
                ],
                temperature=0  # Use 0 for consistent validation
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"INCOMPLETE: Validation error - {str(e)}"
    
    def save_report(self, filepath: Optional[str] = None) -> str:
        """Save the current research to a file."""
        if not self.memory["current_research"]:
            raise ValueError("No research to save. Run research() first.")
        
        # Create output directory if needed
        os.makedirs(AgentConfig.OUTPUT_DIR, exist_ok=True)
        
        # Generate filename if not provided
        if not filepath:
            company_slug = self.memory["current_research"]["company"].lower()
            company_slug = company_slug.replace(" ", "_").replace(".", "")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{AgentConfig.OUTPUT_DIR}/{company_slug}_research_{timestamp}.md"
        
        # Format the report
        research = self.memory["current_research"]
        report = f"""# Company Research Report: {research['company']}

**Generated:** {research['timestamp']}  
**Status:** {research['status']}

---

{research['research']}

---

*Generated by AI Agents Bootcamp | [Standout Systems](https://teodoracoach.substack.com/)*
"""
        
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        
        return filepath


# ==========================================================
# DISPLAY UTILITIES
# ==========================================================

def display_results(research: dict):
    """Pretty print the research results to console."""
    print("\n" + "=" * 60)
    print(f"📋 RESEARCH REPORT: {research['company'].upper()}")
    print("=" * 60)
    print(f"Status: {research['status']}")
    print("-" * 60)
    print(research['research'])
    print("=" * 60)


def display_welcome():
    """Display welcome message."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 AI AGENTS BOOTCAMP - Day 1                          ║
║   ─────────────────────────────────                       ║
║   Your First AI Agent: Company Research                   ║
║                                                           ║
║   This agent demonstrates the 5 building blocks:          ║
║   • Goal      → Research a company                        ║
║   • Reasoning → GPT-4 gathers information                 ║
║   • Tools     → (LLM knowledge as tool)                   ║
║   • Memory    → Stores research results                   ║
║   • Feedback  → Validates completeness                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():
    """Main function to run the research agent."""
    display_welcome()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print("   Please copy .env.example to .env and add your API key.")
        sys.exit(1)
    
    # Get company name from user
    print("Enter a company name to research.")
    print("(Press Enter for default: 'Stripe')\n")
    company = input("🏢 Company: ").strip()
    
    if not company:
        company = "Stripe"
        print(f"   Using default: {company}")
    
    # Initialize and run the agent
    try:
        agent = CompanyResearchAgent()
        results = agent.research(company)
        
        # Display results
        display_results(results)
        
        # Save to file
        filepath = agent.save_report()
        print(f"\n💾 Report saved to: {filepath}")
        
        # Offer to research another company
        print("\n" + "-" * 60)
        another = input("Research another company? (y/n): ").strip().lower()
        if another == 'y':
            main()
        else:
            print("\n👋 Thanks for using AI Agents Bootcamp!")
            print("   Next: Day 2 - Multi-Agent Systems with CrewAI")
            print("   Subscribe: https://teodoracoach.substack.com/\n")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
