"""
normalobjects_analysis.py
Step 5: Analyze Agent Behavior — tracks tool usage, analyzes chaining patterns,
and compares three routing approaches (agent-based, keyword-based, category-based).
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Dict
from collections import Counter
from datetime import datetime

load_dotenv()

# =============================================================================
# PART 1: TOOL TRACKING — hooked into LangChain callbacks (not manual logging)
# =============================================================================

class ToolTracker(BaseCallbackHandler):
    """
    Tracks tool invocations by hooking into LangChain's callback system.
    FIX: Previously the tracker existed but was never connected to the agent,
    so all counts stayed at zero. Now passed via config={"callbacks": [tracker]}.
    """

    def __init__(self):
        self.calls: List[Dict] = []
        self.tool_usage: Counter = Counter()
        self.call_sequences: List[List[str]] = []
        self._current_sequence: List[str] = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """Fires automatically when the agent calls any tool."""
        tool_name = serialized.get("name", "unknown")
        self.tool_usage[tool_name] += 1
        self._current_sequence.append(tool_name)
        self.calls.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": str(input_str)[:200],
        })

    def on_agent_finish(self, finish, **kwargs):
        """Fires when the agent produces its final answer — ends the sequence."""
        if self._current_sequence:
            self.call_sequences.append(list(self._current_sequence))
            self._current_sequence = []

    def get_stats(self) -> Dict:
        return {
            "total_calls": len(self.calls),
            "unique_tools_used": len(self.tool_usage),
            "tool_frequency": dict(self.tool_usage),
            "sequences": self.call_sequences,
            "most_common_tool": self.tool_usage.most_common(1)[0][0] if self.tool_usage else None,
        }


# Global tracker instance (used as callback)
tracker = ToolTracker()

# =============================================================================
# PART 2: DEFINE TOOLS (same set as main script)
# =============================================================================

@tool
def consult_philosopher(complaint: str) -> str:
    """Consult the Existential Philosopher — questions the nature of the problem."""
    return f"🎭 Philosopher's wisdom on '{complaint}': Is this complaint about the thing itself, or about our expectations of it?"

@tool
def consult_comedian(complaint: str) -> str:
    """Consult the Stand-Up Comic — finds absurd humor in any situation."""
    return f"🎪 Comic's take on '{complaint}': This is peak Normal Objects energy! The real joke is we expected consistency."

@tool
def consult_therapist(complaint: str) -> str:
    """Consult the Quirky Therapist — explores emotional roots of complaints."""
    return f"💭 Therapist's insight on '{complaint}': Your complaint is a mirror of your inner landscape. Let's do interpretive dance."

@tool
def consult_inventor(complaint: str) -> str:
    """Consult the Mad Inventor — proposes absurd technological solutions."""
    return f"🔧 Inventor's solution for '{complaint}': Proposes the Complaint Atomizer 3000 and the Reverse Complaint Accelerator."

@tool
def consult_storyteller(complaint: str) -> str:
    """Consult the Narrative Weaver — transforms complaints into epic hero journeys."""
    return f"📖 Storyteller's tale of '{complaint}': This is Act I of your hero's journey. The complaint is your call to adventure."

@tool
def consult_multiple_sources(complaint: str, sources: List[str] = None) -> str:
    """
    Consult multiple creative sources at once for a comprehensive perspective.
    Use when a complaint deserves input from several angles simultaneously.
    """
    available = {
        "philosopher": consult_philosopher,
        "comedian": consult_comedian,
        "therapist": consult_therapist,
        "inventor": consult_inventor,
        "storyteller": consult_storyteller,
    }
    if sources is None:
        sources = list(available.keys())

    responses = []
    for source in sources:
        if source in available:
            responses.append(available[source].invoke({"complaint": complaint}))
    return "\n---\n".join(responses)


# FIX: renamed loop variable to tool_item to avoid shadowing the @tool decorator
tools = [
    consult_philosopher,
    consult_comedian,
    consult_therapist,
    consult_inventor,
    consult_storyteller,
    consult_multiple_sources,
]

# =============================================================================
# PART 3: STRUCTURED APPROACH #1 — KEYWORD-BASED ROUTING
# =============================================================================

class KeywordRouter:
    """Routes complaints to tools based on keyword matching. Deterministic and fast."""

    KEYWORD_MAPPINGS = {
        "philosopher": ["meaning", "why", "nature", "exist", "purpose"],
        "comedian":    ["funny", "laugh", "absurd", "ridiculous", "joke"],
        "therapist":   ["feel", "emotion", "heal", "trauma", "upset", "angry", "sad"],
        "inventor":    ["fix", "build", "tech", "gadget", "solution", "machine"],
        "storyteller": ["story", "tale", "journey", "adventure", "epic"],
    }

    def __init__(self):
        self.results: List[Dict] = []

    def route(self, complaint: str) -> List[str]:
        lower = complaint.lower()
        selected = {
            name for name, keywords in self.KEYWORD_MAPPINGS.items()
            if any(kw in lower for kw in keywords)
        }
        # Default fallback + always add a second tool for variety
        if not selected:
            selected.add("inventor")
        if len(selected) == 1:
            fallback = next(k for k in self.KEYWORD_MAPPINGS if k not in selected)
            selected.add(fallback)

        result = {"complaint": complaint, "tools": list(selected)}
        self.results.append(result)
        return list(selected)


# =============================================================================
# PART 4: STRUCTURED APPROACH #2 — CATEGORY CLASSIFICATION (LLM-assisted)
# =============================================================================

class CategoryClassifier:
    """Classifies complaints into categories using the LLM, then maps to tools."""

    CATEGORIES = {
        "technical":     ["inventor", "comedian"],
        "emotional":     ["therapist", "storyteller"],
        "philosophical": ["philosopher", "storyteller"],
        "absurd":        ["comedian", "inventor", "storyteller"],
    }

    def __init__(self, llm):
        self.llm = llm
        self.results: List[Dict] = []

    def classify_and_route(self, complaint: str) -> Dict:
        prompt = (
            f'Classify this complaint into ONE category: technical, emotional, philosophical, or absurd.\n\n'
            f'Complaint: "{complaint}"\n\n'
            f'Respond with ONLY the category name (one word, lowercase).'
        )
        try:
            category = self.llm.invoke(prompt).content.strip().lower()
            if category not in self.CATEGORIES:
                category = "absurd"
        except Exception:
            category = "absurd"

        selected_tools = self.CATEGORIES[category]
        result = {"complaint": complaint, "category": category, "tools": selected_tools}
        self.results.append(result)
        return result


# =============================================================================
# PART 5: AGENT FACTORY — builds a fresh agent using modern LangGraph API
# =============================================================================

def build_agent_executor(llm, verbose: bool = False):
    """
    FIX: Updated to use modern LangGraph create_react_agent API.
    Returns a compiled agent graph that can be invoked like the old AgentExecutor.
    """
    system_prompt = (
        "You are the Creative Complaint Handler from the Normal Objects universe. "
        "When handling complaints, actively use the available tools. "
        "For each complaint, call 1–3 relevant tools before giving your final answer."
    )

    return create_react_agent(llm, tools, prompt=system_prompt)


# =============================================================================
# PART 6: MAIN ANALYSIS EXECUTION
# =============================================================================

def run_analysis():
    print("\n" + "="*70)
    print("🔬 STEP 5: ANALYZING AGENT BEHAVIOR")
    print("="*70)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    test_complaints = [
        "My coffee maker only makes cold coffee",
        "My alarm clock runs backwards",
        "The stairs in my house keep rearranging themselves at night",
    ]

    # -------------------------------------------------------------------------
    # APPROACH 1: AGENT-BASED
    # -------------------------------------------------------------------------
    print("\n📊 APPROACH 1: Agent-Based Tool Usage")
    print("-" * 70)

    # FIX: use build_agent_executor instead of the non-existent create_agent
    agent_executor = build_agent_executor(llm, verbose=False)
    agent_results = {"calls": [], "by_complaint": {}}

    for i, complaint in enumerate(test_complaints, 1):
        print(f"\n  Complaint {i}: {complaint[:55]}...")
        try:
            result = agent_executor.invoke(
                {"messages": [("user", complaint)]},
                config={"callbacks": [tracker]},  # FIX: tracker actually connected
            )
            # Extract response from LangGraph result (messages array)
            response_text = ""
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            agent_results["by_complaint"][complaint] = {
                "response_length": len(response_text),
                "success": True,
            }
            print(f"  ✓ Processed ({len(response_text)} chars)")
        except Exception as e:
            agent_results["by_complaint"][complaint] = {"success": False}
            print(f"  ✗ Error: {str(e)[:100]}")

    # FIX: populate agent_results["calls"] from the live tracker
    agent_results["calls"] = tracker.calls

    # -------------------------------------------------------------------------
    # APPROACH 2: KEYWORD-BASED ROUTING
    # -------------------------------------------------------------------------
    print("\n📊 APPROACH 2: Keyword-Based Routing")
    print("-" * 70)

    router = KeywordRouter()
    for i, complaint in enumerate(test_complaints, 1):
        selected = router.route(complaint)
        print(f"  Complaint {i}: {len(selected)} tools → {', '.join(selected)}")

    # -------------------------------------------------------------------------
    # APPROACH 3: CATEGORY CLASSIFICATION
    # -------------------------------------------------------------------------
    print("\n📊 APPROACH 3: Category Classification")
    print("-" * 70)

    classifier = CategoryClassifier(llm)
    for i, complaint in enumerate(test_complaints, 1):
        result = classifier.classify_and_route(complaint)
        print(f"  Complaint {i}: Category = {result['category'].upper()} → {', '.join(result['tools'])}")

    # -------------------------------------------------------------------------
    # COMPARATIVE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n📊 COMPARATIVE ANALYSIS")
    print("-" * 70)

    n = len(test_complaints)

    agent_tool_calls  = len(agent_results["calls"])
    keyword_total     = sum(len(r["tools"]) for r in router.results)
    category_total    = sum(len(r["tools"]) for r in classifier.results)

    agent_unique  = len(set(c["tool"] for c in agent_results["calls"]))
    keyword_unique  = len({t for r in router.results for t in r["tools"]})
    category_unique = len({t for r in classifier.results for t in r["tools"]})

    print(f"\n{'Metric':<30} {'Agent':>10} {'Keyword':>10} {'Category':>10}")
    print("-" * 62)
    print(f"{'Total tool calls':<30} {agent_tool_calls:>10} {keyword_total:>10} {category_total:>10}")
    print(f"{'Avg tools / complaint':<30} {agent_tool_calls/n:>10.1f} {keyword_total/n:>10.1f} {category_total/n:>10.1f}")
    print(f"{'Unique tools used':<30} {agent_unique:>10} {keyword_unique:>10} {category_unique:>10}")

    print("\nAgent tool sequence:")
    for i, seq in enumerate(tracker.call_sequences, 1):
        print(f"  Complaint {i}: {' -> '.join(seq) if seq else '(no tools called)'}")

    # -------------------------------------------------------------------------
    # SAVE MARKDOWN REPORT
    # -------------------------------------------------------------------------
    report = generate_report(
        test_complaints, agent_results, router.results, classifier.results,
        agent_tool_calls, keyword_total, category_total,
        agent_unique, keyword_unique, category_unique, n
    )

    # FIX: use relative path instead of hardcoded desktop path
    report_path = "analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {os.path.abspath(report_path)}")
    print("\n" + "="*70)
    print("✅ STEP 5 ANALYSIS COMPLETE!")
    print("="*70)


# =============================================================================
# PART 7: REPORT GENERATION
# =============================================================================

def generate_report(
    complaints, agent_results, keyword_results, category_results,
    agent_total, keyword_total, category_total,
    agent_unique, keyword_unique, category_unique, n
) -> str:

    lines = [
        "# Step 5: Agent Behavior Analysis Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: gpt-4o-mini | Temperature: 0.7*",
        "",
        "## Test Complaints",
        "",
    ]
    for i, c in enumerate(complaints, 1):
        lines.append(f"{i}. \"{c}\"")

    lines += [
        "",
        "---",
        "",
        "## Approach Comparison",
        "",
        "| Metric | Agent-Based | Keyword-Based | Category-Based |",
        "|--------|-------------|---------------|----------------|",
        f"| Total tool calls | {agent_total} | {keyword_total} | {category_total} |",
        f"| Avg tools / complaint | {agent_total/n:.1f} | {keyword_total/n:.1f} | {category_total/n:.1f} |",
        f"| Unique tools used | {agent_unique} | {keyword_unique} | {category_unique} |",
        f"| Deterministic? | ❌ No | ✅ Yes | ✅ Yes |",
        "",
        "---",
        "",
        "## Keyword-Based Results",
        "",
    ]
    for r in keyword_results:
        lines.append(f"- **\"{r['complaint']}\"** → {', '.join(r['tools'])}")

    lines += [
        "",
        "---",
        "",
        "## Category Classification Results",
        "",
    ]
    for r in category_results:
        lines.append(f"- **\"{r['complaint']}\"** → Category: `{r['category']}` → {', '.join(r['tools'])}")

    lines += [
        "",
        "---",
        "",
        "## Agent Tool Sequences",
        "",
    ]
    for i, seq in enumerate(tracker.call_sequences, 1):
        lines.append(f"- Complaint {i}: `{' -> '.join(seq) if seq else 'no tools'}`")

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "1. **Agent approach** is flexible and context-aware but non-deterministic — tool choice depends on the LLM's reasoning.",
        "2. **Keyword routing** is fast and predictable but brittle — it misses synonyms and context.",
        "3. **Category classification** balances semantic understanding with structured routing.",
        "4. For creative, open-ended tasks: use the agent approach.",
        "5. For production pipelines requiring reliability: use keyword or category routing (or LangGraph).",
    ]

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_analysis()