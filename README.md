# AI Agents Bootcamp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Learn to build AI agents from scratch.** This repository accompanies the 3-day AI Agents series on [Standout Systems](https://teodoracoach.substack.com/).

From "what is an agent?" to production-ready multi-agent systems—with working code at every step.

---

## What You'll Build

| Day | Project | Skills |
|-----|---------|--------|
| **Day 1** | Company Research Agent | Agent fundamentals, 5 building blocks |
| **Day 2** | Interview Prep System | CrewAI, multi-agent orchestration, tools |
| **Day 3** | Interactive Interview Bot | LangGraph, flows, loops, production patterns |

---

## Quick Start

### Option 1: Local Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/ai-agents-bootcamp.git
cd ai-agents-bootcamp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run your first agent!
python src/01_first_agent.py
```

### Option 2: Google Colab (Zero Setup)

Open any notebook directly in Colab:

- [Day 1: First Agent](https://colab.research.google.com/github/your-username/ai-agents-bootcamp/blob/main/notebooks/01_first_agent.ipynb)
- [Day 2: Interview Prep System](https://colab.research.google.com/github/your-username/ai-agents-bootcamp/blob/main/notebooks/02_interview_prep.ipynb)
- [Day 3: LangGraph Interview Bot](https://colab.research.google.com/github/your-username/ai-agents-bootcamp/blob/main/notebooks/03_langgraph_bot.ipynb)

---

## Repository Structure

```
ai-agents-bootcamp/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/                          # Python scripts
│   ├── 01_first_agent.py         # Day 1: Simple research agent
│   ├── 02_interview_prep.py      # Day 2: CrewAI multi-agent system
│   ├── 03_langgraph_bot.py       # Day 3: Interactive interview bot
│   └── utils/
│       └── helpers.py            # Shared utilities
│
├── notebooks/                    # Jupyter notebooks (Colab-ready)
│   ├── 01_first_agent.ipynb
│   ├── 02_interview_prep.ipynb
│   └── 03_langgraph_bot.ipynb
│
└── outputs/                      # Generated reports (gitignored)
    └── .gitkeep
```

---

## API Keys Required

### OpenAI API Key (Required)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up / Log in
3. Navigate to API Keys → Create new key
4. Copy and add to your `.env` file

### Serper API Key (Optional - for web search)
1. Go to [serper.dev](https://serper.dev)
2. Sign up for free (2,500 free searches)
3. Copy API key and add to `.env`

---

## The 5 Building Blocks of AI Agents

Every agent in this repo demonstrates these core concepts:

```
┌─────────────────────────────────────────────────────────────┐
│                     AI AGENT ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. GOAL        → What the agent is trying to achieve   │
│   2. REASONING   → LLM that plans and decides            │
│   3. TOOLS       → Capabilities (search, read, execute)  │
│   4. MEMORY      → Context and history storage           │
│   5. FEEDBACK    → Self-evaluation and improvement       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **OpenAI GPT-4** | LLM backbone for reasoning |
| **CrewAI** | Multi-agent orchestration (Day 2) |
| **LangGraph** | Stateful agent workflows (Day 3) |
| **LangChain** | LLM abstractions and tools |
| **Serper** | Web search capabilities |

---

## Learning Path

### Day 1: Foundations
**Goal:** Understand what agents are and build your first one.

- [x] The 5 building blocks of agents
- [x] Simple research agent
- [x] Agent vs. traditional AI comparison

```bash
python src/01_first_agent.py
```

### Day 2: Multi-Agent Systems
**Goal:** Build agents that work together.

- [x] CrewAI framework
- [x] Agents, Tasks, and Crews
- [x] Adding real tools (web search)
- [x] Interview preparation system

```bash
python src/02_interview_prep.py
```

### Day 3: Production Patterns
**Goal:** Build agents ready for real-world use.

- [x] LangGraph for complex flows
- [x] Conditional routing and loops
- [x] Human-in-the-loop patterns
- [x] State management

```bash
python src/03_langgraph_bot.py
```

---

##  Example Output

Running the first agent:

```
$ python src/01_first_agent.py
Enter a company to research (or press Enter for 'Stripe'): Anthropic

 Agent activated: Researching Anthropic...
--------------------------------------------------
 Gathering company information...
 Validating research completeness...
 Research validated as complete

============================================================
 RESEARCH REPORT: Anthropic
============================================================
## Company Overview
- Founded: 2021 by former OpenAI researchers
- Headquarters: San Francisco, California
- Mission: AI safety research and building reliable AI systems
...

 Report saved to: outputs/anthropic_research.txt
```

---

##  Contributing

Found a bug? Have an improvement? Contributions welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  Questions?

- **Newsletter:** [Standout Systems on Substack](https://teodoracoach.substack.com/)
- **Coaching:** [teodora.coach](https://teodora.coach/)
- **Issues:** Open a GitHub issue

---

##  License

MIT License - feel free to use this code for learning and projects.

---

**Built with ❤️ by [Teodora](https://teodora.coach)**

*If this helped you, consider subscribing to [Standout Systems](https://teodoracoach.substack.com/) for more AI/ML career content.*
