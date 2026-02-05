"""
AI Agents Bootcamp - Day 3: Interactive Interview Bot with LangGraph
====================================================================

A stateful, looping interview practice system that demonstrates:
1. LangGraph state machines
2. Conditional routing (loops!)
3. Typed state management
4. Human-in-the-loop patterns
5. Session persistence

This is the production pattern for building real AI systems.

Run: python src/03_langgraph_bot.py
Colab: notebooks/03_langgraph_bot.ipynb

Author: Teodora | https://teodora.coach
Series: https://teodoracoach.substack.com
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Literal, Annotated
import operator

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

try:
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from dotenv import load_dotenv
except ImportError as e:
    print("❌ Missing dependencies. Install with:")
    print("   pip install langgraph langchain-openai langchain-core python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class BotConfig:
    """Configuration for the interview practice bot."""
    
    # LLM Settings
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Interview Settings (Customize these!)
    COMPANY = "Anthropic"
    POSITION = "Senior Machine Learning Engineer"
    DIFFICULTY = "medium"  # easy, medium, hard
    
    # Output
    OUTPUT_DIR = Path("outputs")
    CHECKPOINT_FILE = "interview_checkpoint.json"


def validate_api_key():
    """Check that OpenAI API key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found!")
        print("   Set it in your .env file or environment")
        sys.exit(1)
    return True


# ============================================================================
# STATE DEFINITION
# ============================================================================

class InterviewState(TypedDict):
    """
    The shared state that flows through all nodes.
    
    This is THE critical concept in LangGraph:
    - Every node reads from state
    - Every node writes to state
    - State is typed (catches bugs early)
    - State can be persisted (resume on crash)
    """
    
    # Configuration
    company: str
    position: str
    difficulty: str
    
    # Questions
    questions: List[str]
    current_index: int
    
    # Conversation tracking (Annotated with operator.add = auto-append)
    exchanges: Annotated[List[dict], operator.add]
    
    # Performance metrics
    scores: List[int]
    weak_areas: List[str]
    
    # Flow control
    user_wants_continue: bool
    session_complete: bool


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def get_llm():
    """Initialize the language model."""
    return ChatOpenAI(
        model=BotConfig.MODEL,
        temperature=BotConfig.TEMPERATURE
    )


# ============================================================================
# NODE FUNCTIONS
# Each node: state in → transformed state out
# ============================================================================

def generate_questions(state: InterviewState) -> dict:
    """
    NODE 1: Generate interview questions
    
    Building Blocks:
    - Goal: Create tailored questions
    - Reasoning: LLM with interviewer persona
    - Tools: None (pure generation)
    - Memory: Writes questions to state
    """
    llm = get_llm()
    
    print("\n" + "=" * 60)
    print("🎯 GENERATING QUESTIONS")
    print("=" * 60)
    print(f"\nPreparing {state['difficulty']} questions for {state['position']}...")
    
    response = llm.invoke([
        SystemMessage(content=f"""You are a senior interviewer at {state['company']}.
        Generate exactly 5 interview questions for a {state['position']} role.
        
        Difficulty level: {state['difficulty']}
        
        Create a balanced mix:
        - 2 technical questions (appropriate to the role)
        - 2 behavioral questions (STAR format appropriate)  
        - 1 role-specific scenario question
        
        Return ONLY the questions, numbered 1-5.
        No explanations, no headers, just the questions."""),
        HumanMessage(content="Generate the 5 interview questions now.")
    ])
    
    # Parse questions from response
    lines = response.content.strip().split('\n')
    questions = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove number prefix if present
            if line[0].isdigit():
                line = line.split('.', 1)[-1].strip()
                line = line.split(')', 1)[-1].strip()
            questions.append(line)
    
    # Ensure we have questions
    if not questions:
        questions = [
            "Tell me about a challenging technical project you've worked on.",
            "How do you approach debugging a complex system?",
            "Describe a time you had to learn a new technology quickly.",
            "How would you design a system for [relevant task]?",
            "What interests you about this role at our company?"
        ]
    
    print(f"\n✅ Generated {len(questions)} questions:\n")
    for i, q in enumerate(questions[:5], 1):
        preview = q[:70] + "..." if len(q) > 70 else q
        print(f"   {i}. {preview}")
    
    return {
        "questions": questions[:5],  # Limit to 5
        "current_index": 0,
        "exchanges": [],
        "scores": [],
        "weak_areas": []
    }


def ask_question(state: InterviewState) -> dict:
    """
    NODE 2: Present question and collect answer
    
    Building Blocks:
    - Goal: Get user's response
    - Memory: Store Q&A exchange in state
    - Feedback: User input is the feedback loop
    """
    idx = state["current_index"]
    question = state["questions"][idx]
    total = len(state["questions"])
    
    print("\n" + "=" * 60)
    print(f"📝 QUESTION {idx + 1} of {total}")
    print("=" * 60)
    print(f"\n🎤 {question}\n")
    print("-" * 60)
    print("Type your answer below.")
    print("(Press Enter twice when finished)\n")
    
    # Collect multi-line input
    lines = []
    empty_count = 0
    while empty_count < 1:
        try:
            line = input()
            if line == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break
    
    user_answer = "\n".join(lines).strip()
    
    if not user_answer:
        user_answer = "(No answer provided)"
    
    # Create exchange record
    exchange = {
        "question_num": idx + 1,
        "question": question,
        "answer": user_answer,
        "timestamp": datetime.now().isoformat()
    }
    
    return {"exchanges": [exchange]}


def analyze_answer(state: InterviewState) -> dict:
    """
    NODE 3: Analyze the answer with LLM
    
    Building Blocks:
    - Goal: Score and identify improvements
    - Reasoning: LLM as evaluation engine
    - Memory: Updates scores and weak_areas
    """
    llm = get_llm()
    last_exchange = state["exchanges"][-1]
    
    print("\n⏳ Analyzing your response...")
    
    response = llm.invoke([
        SystemMessage(content="""You are an expert interview coach. 
        Analyze this interview answer and provide structured feedback.
        
        Score the answer from 1-10 based on:
        - Clarity and structure
        - Relevance to the question
        - Specific examples provided
        - Technical accuracy (if applicable)
        - Communication quality
        
        Format your response EXACTLY like this:
        SCORE: [number 1-10]
        
        STRENGTHS:
        • [specific strength 1]
        • [specific strength 2]
        
        IMPROVEMENTS:
        • [specific, actionable improvement 1]
        • [specific, actionable improvement 2]
        
        PRO TIP: [one insider tip for this type of question]
        
        WEAK_AREA: [if score < 7, name ONE skill to work on, e.g., "using specific examples", "structuring responses with STAR", "technical depth". If score >= 7, write "none"]
        """),
        HumanMessage(content=f"""
        Interview Question: {last_exchange['question']}
        
        Candidate's Answer: {last_exchange['answer']}
        
        Provide your analysis now.
        """)
    ])
    
    # Parse the response
    content = response.content
    score = 5  # default
    weak_area = None
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('SCORE:'):
            try:
                score_text = line.split(':')[1].strip()
                # Handle formats like "7/10" or just "7"
                score = int(score_text.split('/')[0].strip())
                score = max(1, min(10, score))  # Clamp to 1-10
            except:
                score = 5
        elif line.startswith('WEAK_AREA:'):
            area = line.split(':', 1)[1].strip()
            if area.lower() not in ['none', 'n/a', '-']:
                weak_area = area
    
    # Store feedback in the exchange
    state["exchanges"][-1]["score"] = score
    state["exchanges"][-1]["feedback"] = content
    
    # Build return updates
    result = {
        "scores": state.get("scores", []) + [score]
    }
    
    if weak_area:
        result["weak_areas"] = state.get("weak_areas", []) + [weak_area]
    
    return result


def give_feedback(state: InterviewState) -> dict:
    """
    NODE 4: Present feedback and ask to continue
    
    Building Blocks:
    - Goal: Help user improve
    - Feedback: Display analysis + get continuation signal
    """
    last_exchange = state["exchanges"][-1]
    score = last_exchange.get("score", 5)
    feedback = last_exchange.get("feedback", "Analysis not available.")
    
    print("\n" + "=" * 60)
    print("📊 FEEDBACK")
    print("=" * 60)
    
    # Visual score bar
    filled = "█" * score
    empty = "░" * (10 - score)
    
    if score >= 8:
        emoji = "🌟"
        rating = "Excellent!"
    elif score >= 6:
        emoji = "✅"
        rating = "Good"
    elif score >= 4:
        emoji = "🔧"
        rating = "Needs Work"
    else:
        emoji = "📚"
        rating = "Keep Practicing"
    
    print(f"\n{emoji} Score: [{filled}{empty}] {score}/10 - {rating}\n")
    print("-" * 60)
    print(feedback)
    print("-" * 60)
    
    # Check if more questions available
    new_index = state["current_index"] + 1
    remaining = len(state["questions"]) - new_index
    
    if remaining > 0:
        print(f"\n📋 {remaining} question(s) remaining")
        continue_input = input("➡️  Continue to next question? (yes/no): ").strip().lower()
        wants_continue = continue_input in ['yes', 'y', '']
    else:
        print("\n📝 All questions completed!")
        wants_continue = False
    
    return {
        "current_index": new_index,
        "user_wants_continue": wants_continue
    }


def wrap_up_session(state: InterviewState) -> dict:
    """
    NODE 5: Summarize session and provide final advice
    
    Building Blocks:
    - Goal: Synthesize performance
    - Memory: Read all exchanges for summary
    - Reasoning: LLM generates personalized advice
    """
    llm = get_llm()
    
    print("\n" + "=" * 60)
    print("🎓 SESSION COMPLETE")
    print("=" * 60)
    
    scores = state.get("scores", [])
    exchanges = state.get("exchanges", [])
    weak_areas = state.get("weak_areas", [])
    
    # Performance statistics
    if scores:
        avg_score = sum(scores) / len(scores)
        best_score = max(scores)
        worst_score = min(scores)
        
        print(f"\n📊 Performance Summary")
        print(f"   ├─ Questions answered: {len(scores)}")
        print(f"   ├─ Average score: {avg_score:.1f}/10")
        print(f"   ├─ Best answer: {best_score}/10")
        print(f"   └─ Lowest answer: {worst_score}/10")
    
    # Score distribution
    if scores:
        print(f"\n📈 Score Distribution")
        for i, (exchange, score) in enumerate(zip(exchanges, scores), 1):
            bar = "█" * score + "░" * (10 - score)
            print(f"   Q{i}: [{bar}] {score}/10")
    
    # Weak areas
    if weak_areas:
        unique_areas = list(set(weak_areas))
        print(f"\n🎯 Areas to Practice")
        for area in unique_areas:
            count = weak_areas.count(area)
            print(f"   • {area}" + (f" (×{count})" if count > 1 else ""))
    
    # Generate personalized advice
    print(f"\n💡 Coach's Final Notes")
    print("-" * 40)
    
    response = llm.invoke([
        SystemMessage(content="""You are an encouraging interview coach.
        Based on this practice session, provide:
        
        1. One thing they did really well (be specific)
        2. The #1 thing to focus on improving
        3. A concrete exercise to practice this week
        
        Keep it under 100 words. Be encouraging but honest."""),
        HumanMessage(content=f"""
        Position: {state['position']} at {state['company']}
        Scores: {scores}
        Weak areas identified: {weak_areas}
        Number of questions: {len(exchanges)}
        """)
    ])
    
    print(response.content)
    print("-" * 40)
    
    return {"session_complete": True}


# ============================================================================
# ROUTER (Conditional Logic)
# ============================================================================

def should_continue(state: InterviewState) -> Literal["ask_question", "wrap_up"]:
    """
    THE ROUTER: Decides where to go next
    
    This is the key LangGraph concept - conditional edges.
    Based on state, we either:
    - Loop back to ask_question
    - Exit to wrap_up
    """
    # User said they want to stop
    if not state.get("user_wants_continue", True):
        return "wrap_up"
    
    # All questions have been asked
    if state["current_index"] >= len(state["questions"]):
        return "wrap_up"
    
    # Continue the loop
    return "ask_question"


# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

def build_interview_graph():
    """
    Assemble the LangGraph state machine.
    
    Flow:
    generate_questions → ask_question → analyze → feedback 
                              ↑                        │
                              └────── (if continue) ───┘
                                             │
                                      (if done) → wrap_up → END
    """
    # Create the graph with our state type
    workflow = StateGraph(InterviewState)
    
    # Add all nodes
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("ask_question", ask_question)
    workflow.add_node("analyze_answer", analyze_answer)
    workflow.add_node("give_feedback", give_feedback)
    workflow.add_node("wrap_up", wrap_up_session)
    
    # Set the entry point
    workflow.set_entry_point("generate_questions")
    
    # Add linear edges
    workflow.add_edge("generate_questions", "ask_question")
    workflow.add_edge("ask_question", "analyze_answer")
    workflow.add_edge("analyze_answer", "give_feedback")
    
    # Add THE LOOP - conditional edge from feedback
    workflow.add_conditional_edges(
        "give_feedback",
        should_continue,
        {
            "ask_question": "ask_question",  # Loop back
            "wrap_up": "wrap_up"              # Exit
        }
    )
    
    # Final edge to END
    workflow.add_edge("wrap_up", END)
    
    # Compile and return
    return workflow.compile()


# ============================================================================
# CHECKPOINT UTILITIES (Production Pattern)
# ============================================================================

def save_checkpoint(state: dict, filename: str = None):
    """Save state to disk for crash recovery."""
    if filename is None:
        filename = BotConfig.OUTPUT_DIR / BotConfig.CHECKPOINT_FILE
    
    BotConfig.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Convert state to serializable format
    save_state = {k: v for k, v in state.items()}
    
    with open(filename, 'w') as f:
        json.dump(save_state, f, indent=2, default=str)
    
    return filename


def load_checkpoint(filename: str = None) -> dict:
    """Load state from disk."""
    if filename is None:
        filename = BotConfig.OUTPUT_DIR / BotConfig.CHECKPOINT_FILE
    
    if not Path(filename).exists():
        return None
    
    with open(filename, 'r') as f:
        return json.load(f)


def save_session_report(state: dict):
    """Save a detailed session report."""
    BotConfig.OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    company_slug = state.get("company", "unknown").lower().replace(" ", "_")
    filename = BotConfig.OUTPUT_DIR / f"interview_session_{company_slug}_{timestamp}.md"
    
    scores = state.get("scores", [])
    exchanges = state.get("exchanges", [])
    
    report = f"""# Interview Practice Session Report

**Company:** {state.get('company', 'N/A')}
**Position:** {state.get('position', 'N/A')}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Difficulty:** {state.get('difficulty', 'N/A')}

---

## Performance Summary

- **Questions Completed:** {len(scores)}
- **Average Score:** {sum(scores)/len(scores):.1f}/10 if scores else "N/A"
- **Score Range:** {min(scores) if scores else 0} - {max(scores) if scores else 0}

---

## Question-by-Question Breakdown

"""
    
    for i, exchange in enumerate(exchanges, 1):
        score = exchange.get('score', 'N/A')
        report += f"""### Question {i}

**Q:** {exchange.get('question', 'N/A')}

**Your Answer:**
{exchange.get('answer', 'N/A')}

**Score:** {score}/10

**Feedback:**
{exchange.get('feedback', 'N/A')}

---

"""
    
    # Weak areas summary
    weak_areas = state.get("weak_areas", [])
    if weak_areas:
        report += """## Areas for Improvement

"""
        for area in set(weak_areas):
            report += f"- {area}\n"
    
    report += f"""
---

*Generated by AI Agents Bootcamp - Day 3*
*https://github.com/DoraSzasz/ai-agents-bootcamp*
"""
    
    with open(filename, 'w') as f:
        f.write(report)
    
    return filename


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the interview practice bot."""
    
    print("\n" + "=" * 60)
    print("🤖 AI AGENTS BOOTCAMP - DAY 3")
    print("   Interactive Interview Practice Bot")
    print("=" * 60)
    
    # Validate setup
    validate_api_key()
    
    # Show configuration
    print(f"\n📍 Company:    {BotConfig.COMPANY}")
    print(f"💼 Position:   {BotConfig.POSITION}")
    print(f"📊 Difficulty: {BotConfig.DIFFICULTY}")
    print(f"\n(Edit BotConfig in the script to customize)")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint and not checkpoint.get("session_complete", False):
        print(f"\n⚠️  Found incomplete session from earlier")
        resume = input("Resume previous session? (yes/no): ").strip().lower()
        if resume in ['yes', 'y']:
            initial_state = checkpoint
            print("✅ Resuming previous session...")
        else:
            checkpoint = None
    
    if not checkpoint:
        # Start fresh
        print(f"\n➡️  Press Enter to start your practice session...")
        input()
        
        initial_state = {
            "company": BotConfig.COMPANY,
            "position": BotConfig.POSITION,
            "difficulty": BotConfig.DIFFICULTY,
            "questions": [],
            "current_index": 0,
            "exchanges": [],
            "scores": [],
            "weak_areas": [],
            "user_wants_continue": True,
            "session_complete": False
        }
    
    # Build and run the graph
    app = build_interview_graph()
    
    try:
        result = app.invoke(initial_state)
        
        # Save session report
        report_path = save_session_report(result)
        print(f"\n💾 Session report saved: {report_path}")
        
        # Clean up checkpoint
        checkpoint_path = BotConfig.OUTPUT_DIR / BotConfig.CHECKPOINT_FILE
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Session interrupted!")
        # Save checkpoint for resume
        save_checkpoint(initial_state)
        print(f"💾 Progress saved. Run again to resume.")
    
    print("\n✅ Good luck with your interview!\n")


if __name__ == "__main__":
    main()
