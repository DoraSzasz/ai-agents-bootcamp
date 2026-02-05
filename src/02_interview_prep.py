"""
AI Agents Bootcamp - Day 2: Multi-Agent Interview Prep System
=============================================================

A CrewAI-powered system with 3 specialized agents:
1. Research Analyst - Gathers company intelligence
2. Strategy Consultant - Analyzes fit and positioning  
3. Interview Coach - Creates targeted prep materials

This demonstrates:
- Multi-agent orchestration with CrewAI
- Task chaining with shared context
- Tool integration (web search)
- Professional agent design patterns

Run: python src/02_interview_prep.py
Colab: notebooks/02_interview_prep.ipynb

Author: Teodora | https://teodora.coach
Series: https://teodoracoach.substack.com
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

try:
    from crewai import Agent, Crew, Task, Process
    from crewai_tools import SerperDevTool
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print("❌ Missing dependencies. Install with:")
    print("   pip install crewai crewai-tools langchain-openai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class InterviewConfig:
    """Configuration for the interview prep system."""
    
    # LLM Settings
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Interview Details (Customize these!)
    COMPANY = "Anthropic"
    POSITION = "Senior Machine Learning Engineer"
    INTERVIEWER = "Chris Olah"  # From your calendar invite
    
    JOB_DESCRIPTION = """
    We're looking for ML engineers to work on interpretability research.
    You'll develop techniques to understand neural network behavior,
    work on scaling alignment research, and collaborate with researchers
    on safety-focused ML systems. Requirements: 5+ years ML experience,
    strong Python, experience with transformer architectures.
    """
    
    YOUR_BACKGROUND = """
    - 6 years ML engineering at tech companies
    - Built production transformer models for NLP
    - Published paper on attention visualization
    - Led team of 4 ML engineers
    - Strong Python, PyTorch, distributed training
    """
    
    # Output directory
    OUTPUT_DIR = Path("outputs")


def validate_api_keys():
    """Check that required API keys are set."""
    openai_key = os.getenv("OPENAI_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found!")
        print("   Set it in your .env file or environment")
        sys.exit(1)
    
    if not serper_key:
        print("⚠️  SERPER_API_KEY not found - web search disabled")
        print("   Get a free key at https://serper.dev")
        return False
    
    return True


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_research_agent(llm, tools):
    """
    🔍 THE RESEARCHER
    
    Building Blocks:
    - Goal: Comprehensive company intelligence
    - Reasoning: LLM with research-focused backstory
    - Tools: Web search for real-time information
    - Memory: Findings passed to next agents via task context
    """
    return Agent(
        role="Senior Company Research Analyst",
        goal=f"Conduct comprehensive research on {InterviewConfig.COMPANY} "
             f"to prepare candidate for interview success",
        backstory="""You are an elite research analyst who has helped hundreds 
        of candidates land jobs at top companies. You know exactly what information 
        matters: recent news, company culture, technical challenges, and the 
        specific team dynamics. You dig deep and never settle for surface-level 
        information. You cite your sources and focus on actionable insights.""",
        llm=llm,
        tools=tools,
        verbose=True,
        max_iter=5,
        allow_delegation=False
    )


def create_analyzer_agent(llm):
    """
    🎯 THE FIT ANALYZER
    
    Building Blocks:
    - Goal: Strategic positioning for the candidate
    - Reasoning: LLM with career strategy expertise
    - Tools: None (works from research context)
    - Memory: Receives research, outputs to coach
    """
    return Agent(
        role="Career Strategy Consultant",
        goal="Analyze candidate-company fit and develop winning positioning strategy",
        backstory="""You are a career strategist who has coached executives at 
        FAANG companies. You excel at finding the perfect angle to present 
        someone's experience. You see connections others miss—how a candidate's 
        past projects align with a company's current challenges. You think in 
        terms of compelling stories, not bullet points.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_coach_agent(llm):
    """
    💬 THE INTERVIEW COACH
    
    Building Blocks:
    - Goal: Targeted interview preparation
    - Reasoning: LLM with interviewer perspective
    - Tools: None (works from research + analysis)
    - Memory: Synthesizes all prior work into prep guide
    """
    return Agent(
        role="Technical Interview Coach",
        goal=f"Generate highly targeted interview questions and prep strategy for "
             f"{InterviewConfig.POSITION} role",
        backstory="""You've conducted 500+ technical interviews at top tech 
        companies. You know exactly what interviewers look for at each level.
        You create questions that are specific, realistic, and actually 
        prepare people for success—not generic fluff. You think about the 
        interviewer's perspective and what they're really trying to assess.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_research_task(agent):
    """Deep company research task with web search."""
    return Task(
        description=f"""
        Research {InterviewConfig.COMPANY} comprehensively for an upcoming interview.
        
        Cover these areas IN DEPTH:
        
        1. **Recent News** (last 6 months)
           - Major announcements, funding rounds, product launches
           - Any controversies, challenges, or pivots
           - Key wins, milestones, or recognition
        
        2. **Technical Focus**
           - What technical problems are they solving?
           - Their tech stack and approach
           - Recent engineering blog posts, papers, or open source
        
        3. **Team & Culture**
           - Leadership team backgrounds
           - Company values (not PR speak—what they actually DO)
           - Glassdoor/Blind insights on real culture
        
        4. **The Specific Role: {InterviewConfig.POSITION}**
           - What team would this likely be on?
           - Current projects or initiatives
           - What would success look like in 6 months?
        
        5. **Interviewer Research: {InterviewConfig.INTERVIEWER}**
           - Their background and expertise
           - Published work, talks, or public opinions
           - What do they care about professionally?
        
        Be specific and cite sources. Skip generic information.
        """,
        expected_output="""
        A detailed research report with:
        - Executive summary (3-4 key takeaways)
        - Detailed findings organized by category
        - Non-obvious insights that matter for interview
        - Sources cited for key facts
        """,
        agent=agent
    )


def create_analysis_task(agent, research_task):
    """Strategic fit analysis task."""
    return Task(
        description=f"""
        Based on the research findings, analyze candidate-company fit and 
        create a strategic positioning plan.
        
        **Candidate Background:**
        {InterviewConfig.YOUR_BACKGROUND}
        
        **Job Description:**
        {InterviewConfig.JOB_DESCRIPTION}
        
        Analyze and deliver:
        
        1. **Strongest Alignment Points**
           - Which candidate experiences directly match company needs?
           - What's the perfect "lead story" for the interview?
        
        2. **Gap Analysis & Mitigation**
           - What might the interviewer worry about?
           - How should the candidate proactively address concerns?
        
        3. **Unique Value Proposition**
           - What does this candidate bring that others don't?
           - What's the memorable 30-second pitch?
        
        4. **Prepared Stories** (STAR format)
           - 3-5 specific examples to have ready
           - Map each to company values or needs
        
        5. **Things to Avoid**
           - Topics to steer away from
           - Common mistakes for this company/role
        """,
        expected_output="""
        A strategic positioning brief with:
        - Clear positioning statement (2-3 sentences)
        - Experience-to-need mapping table
        - 3-5 prepared stories with STAR outlines
        - Risk mitigation strategies
        """,
        agent=agent,
        context=[research_task]
    )


def create_coaching_task(agent, research_task, analysis_task):
    """Interview prep questions and strategy task."""
    return Task(
        description=f"""
        Based on ALL previous analysis, create a comprehensive interview 
        preparation guide for {InterviewConfig.POSITION} at {InterviewConfig.COMPANY}.
        
        Generate:
        
        1. **Technical Questions** (10 questions)
           - Questions specific to their problems and tech
           - Include expected answer framework for each
           - Mix: coding, system design, ML theory
        
        2. **Behavioral Questions** (8 questions)
           - Questions aligned with company values
           - Note what interviewer is REALLY assessing
           - STAR-format answer prompts
        
        3. **Role-Specific Scenarios** (5 questions)
           - "How would you approach X problem?"
           - Based on their actual technical challenges
        
        4. **Questions to ASK THEM** (5 questions)
           - Smart questions showing you researched
           - Questions that help evaluate if YOU want the job
           - Note the strategic intent of each question
        
        5. **Interview Day Strategy**
           - Strong opening (first 2 minutes script)
           - How to handle "any questions?" at end
           - Energy and presence tips
        
        Every question must be SPECIFIC to this company. Zero generic questions.
        """,
        expected_output="""
        Complete interview prep guide with:
        - All questions categorized and numbered
        - Answer frameworks or key points for each
        - 5 questions to ask (with strategic reasoning)
        - Day-of strategy and talking points
        """,
        agent=agent,
        context=[research_task, analysis_task]
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

class InterviewPrepCrew:
    """
    Multi-agent crew for interview preparation.
    
    Demonstrates the 5 Building Blocks at system level:
    - Goal: Prepare candidate for specific interview
    - Reasoning: 3 specialized LLM agents
    - Tools: Web search for research agent
    - Memory: Task context passing between agents
    - Feedback: Sequential validation through pipeline
    """
    
    def __init__(self, use_web_search=True):
        """Initialize the crew with agents and tasks."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=InterviewConfig.MODEL,
            temperature=InterviewConfig.TEMPERATURE
        )
        
        # Initialize tools
        self.tools = []
        if use_web_search and os.getenv("SERPER_API_KEY"):
            self.tools.append(SerperDevTool())
            print("✅ Web search enabled")
        else:
            print("⚠️  Running without web search (limited research)")
        
        # Create agents
        self.research_agent = create_research_agent(self.llm, self.tools)
        self.analyzer_agent = create_analyzer_agent(self.llm)
        self.coach_agent = create_coach_agent(self.llm)
        
        # Create tasks
        self.research_task = create_research_task(self.research_agent)
        self.analysis_task = create_analysis_task(
            self.analyzer_agent, 
            self.research_task
        )
        self.coaching_task = create_coaching_task(
            self.coach_agent,
            self.research_task,
            self.analysis_task
        )
        
        # Assemble crew
        self.crew = Crew(
            agents=[
                self.research_agent,
                self.analyzer_agent,
                self.coach_agent
            ],
            tasks=[
                self.research_task,
                self.analysis_task,
                self.coaching_task
            ],
            process=Process.sequential,
            verbose=True
        )
    
    def run(self):
        """Execute the interview prep workflow."""
        print("\n" + "=" * 60)
        print("🚀 INTERVIEW PREP SYSTEM ACTIVATED")
        print("=" * 60)
        print(f"📍 Company:    {InterviewConfig.COMPANY}")
        print(f"💼 Position:   {InterviewConfig.POSITION}")
        print(f"👤 Interviewer: {InterviewConfig.INTERVIEWER}")
        print("=" * 60)
        print("\n⏳ Running 3-agent pipeline (2-4 minutes)...\n")
        
        result = self.crew.kickoff()
        
        return result
    
    def save_report(self, result):
        """Save the prep guide to a file."""
        # Ensure output directory exists
        InterviewConfig.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_slug = InterviewConfig.COMPANY.lower().replace(" ", "_")
        filename = f"{company_slug}_interview_prep_{timestamp}.md"
        filepath = InterviewConfig.OUTPUT_DIR / filename
        
        # Build report content
        report = f"""# Interview Prep Guide: {InterviewConfig.COMPANY}
## {InterviewConfig.POSITION}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Interviewer:** {InterviewConfig.INTERVIEWER}

---

{result.raw}

---

*Generated by AI Agents Bootcamp - Day 2*
*https://github.com/DoraSzasz/ai-agents-bootcamp*
"""
        
        # Write file
        with open(filepath, "w") as f:
            f.write(report)
        
        return filepath


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the interview prep system."""
    print("\n" + "=" * 60)
    print("🤖 AI AGENTS BOOTCAMP - DAY 2")
    print("   Multi-Agent Interview Prep System")
    print("=" * 60 + "\n")
    
    # Validate API keys
    has_serper = validate_api_keys()
    
    # Allow user to customize
    print(f"Current settings:")
    print(f"  Company:  {InterviewConfig.COMPANY}")
    print(f"  Position: {InterviewConfig.POSITION}")
    print(f"\n(Edit InterviewConfig in the script to customize)\n")
    
    user_input = input("Press Enter to start or 'q' to quit: ").strip()
    if user_input.lower() == 'q':
        print("Goodbye!")
        return
    
    # Create and run crew
    crew = InterviewPrepCrew(use_web_search=has_serper)
    result = crew.run()
    
    # Display results
    print("\n" + "=" * 60)
    print("📋 YOUR INTERVIEW PREP GUIDE")
    print("=" * 60 + "\n")
    print(result.raw)
    
    # Save report
    filepath = crew.save_report(result)
    print("\n" + "=" * 60)
    print(f"💾 Report saved: {filepath}")
    print("=" * 60)
    
    print("\n✅ Done! Good luck with your interview!\n")


if __name__ == "__main__":
    main()
