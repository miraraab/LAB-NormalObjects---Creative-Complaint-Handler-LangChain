import os
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Dict

# ============================================================================
# STEP 1: Initialize LangChain environment
# ============================================================================
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
print("✅ Step 1 complete: LangChain environment initialized")

# ============================================================================
# STEP 2: Create Creative Tools for the Normal Objects Universe
# ============================================================================

@tool
def consult_philosopher(complaint: str) -> str:
    """
    Consult with the Existential Philosopher about a complaint.
    The philosopher questions the fundamental nature of the problem itself
    using Socratic method and absurdist humor.

    Args:
        complaint: The complaint to examine philosophically

    Returns:
        A philosophical perspective on the complaint
    """
    return f"""
    🎭 The Existential Philosopher ponders your complaint: "{complaint}"

    Consider these philosophical angles:
    - Is this complaint about the thing itself, or about our expectations of it?
    - What would Camus say about this absurd situation?
    - Does this complaint reveal something deeper about human nature?
    - Perhaps the complaint IS the solution?
    """


@tool
def consult_comedian(complaint: str) -> str:
    """
    Consult with the Stand-Up Comic who finds humor in any complaint.
    Returns a comedic, absurdist perspective on the problem.

    Args:
        complaint: The complaint to roast comedically

    Returns:
        A comedic take on the complaint
    """
    return f"""
    🎪 The Stand-Up Comic examines your complaint: "{complaint}"

    Comic observations:
    - Have you noticed how this complaint is basically the universe trolling you?
    - The real joke is that we expect things to work smoothly in the first place!
    - I have three words for this situation: "Yeah, that tracks."
    - This is peak Normal Objects energy right here!
    """


@tool
def consult_therapist(complaint: str) -> str:
    """
    Consult with the Quirky Therapist about the emotional roots of a complaint.
    Provides unconventional but empathetic therapeutic wisdom.

    Args:
        complaint: The complaint to explore emotionally

    Returns:
        Therapeutic insights on the complaint
    """
    return f"""
    💭 The Quirky Therapist listens to your complaint: "{complaint}"

    Therapeutic insights:
    - Your complaint is valid, and also a mirror of your inner landscape
    - Have you tried naming the complaint and thanking it for the lesson?
    - What if we reframed this as an opportunity for growth?
    - I sense the complaint is actually your subconscious trying to tell you something
    - Let's do some interpretive dance to process these emotions
    """


@tool
def consult_inventor(complaint: str) -> str:
    """
    Consult with the Mad Inventor who creates absurd technological solutions.
    Returns wildly creative gadget-based fixes for any problem.

    Args:
        complaint: The problem to solve with wild inventions

    Returns:
        Absurd but creative technological solutions
    """
    return f"""
    🔧 The Mad Inventor engineers a solution to: "{complaint}"

    Proposed inventions:
    - The Complaint Atomizer 3000: Breaks problems into manageable atoms
    - Reverse Complaint Accelerator: Applies the problem backwards to cancel itself
    - The Complaint Transcendence Device: Launches problems into space
    - AI-Powered Complaint Converter: Transforms complaints into interpretive art
    - The Rubber Duck Override: Explains the problem to a rubber duck until it solves itself
    """


@tool
def consult_storyteller(complaint: str) -> str:
    """
    Consult with the Narrative Weaver who transforms complaints into epic tales.
    Returns a dramatic narrative reframing of the problem as a hero's journey.

    Args:
        complaint: The complaint to transform into an epic narrative

    Returns:
        A dramatic story reframing of the complaint
    """
    return f"""
    📖 The Narrative Weaver transforms your complaint into an epic: "{complaint}"

    The Tale:
    - In a realm where Normal Objects dwell, a hero faces: {complaint}
    - This is not mere suffering — this is the HERO'S JOURNEY
    - Act I: The Call to Complaint (You encounter the problem)
    - Act II: Trials and Tribulations (You struggle against adversity)
    - Act III: The Triumph (You emerge transformed and wise)
    - The complaint is your monomyth, your personal legend!
    """


@tool
def consult_multiple_sources(complaint: str, sources: List[str] = None) -> str:
    """
    Consult with multiple creative sources simultaneously for a comprehensive response.
    Use this when a complaint deserves input from several different perspectives at once.

    Args:
        complaint: The complaint to address
        sources: List of consultants to use: philosopher, comedian, therapist, inventor, storyteller.
                 If omitted, all sources are consulted.

    Returns:
        Combined wisdom from all requested sources
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
            # FIX: correct .invoke() call signature for LangChain tools
            responses.append(available[source].invoke({"complaint": complaint}))

    return "\n---\n".join(responses)


# FIX: renamed loop variable from `tool` to `tool_item` to avoid shadowing the @tool decorator
tools = [
    consult_philosopher,
    consult_comedian,
    consult_therapist,
    consult_inventor,
    consult_storyteller,
    consult_multiple_sources,
]

print("\n✅ Step 2 complete: Creative tools initialized:")
for tool_item in tools:
    print(f"  - {tool_item.name}")

# ============================================================================
# STEP 3: Create the Agent
# ============================================================================

system_prompt = """You are the Creative Complaint Handler, a wonderfully eccentric AI assistant
from the Normal Objects universe. Your job is to handle complaints in the most creative,
entertaining, and absurd ways possible.

When someone brings you a complaint, you should:
1. Consider consulting different sources — philosopher, comedian, therapist, inventor, storyteller
2. Chain tools together creatively to provide a response that is both helpful and hilarious
3. Embrace the absurdity of the Normal Objects universe
4. Use multiple perspectives to give the complaint a thorough (if bizarre) examination

Always use at least 2 tools per complaint. You can call consult_multiple_sources to query
several perspectives at once, or call individual tools for more targeted insights."""

# Create agent using LangGraph's create_react_agent (modern LangChain 1.x API)
agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

print("\n✅ Step 3 complete: Agent and AgentExecutor created!")

# ============================================================================
# STEP 4: Test the Agent with Creative Complaints
# ============================================================================

# FIX: Added ToolUsageTracker callback so tool usage is actually tracked
class ToolUsageTracker(BaseCallbackHandler):
    """Tracks tool usage by hooking into LangChain's callback system."""

    def __init__(self):
        self.usage_count: Dict[str, int] = {t.name: 0 for t in tools}
        self.tool_sequences: List[str] = []
        self._current_sequence: List[str] = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """Called automatically every time the agent invokes a tool."""
        tool_name = serialized.get("name", "unknown")
        if tool_name in self.usage_count:
            self.usage_count[tool_name] += 1
        else:
            self.usage_count[tool_name] = 1
        self._current_sequence.append(tool_name)

    def on_agent_finish(self, finish, **kwargs):
        """Called when the agent produces its final answer — marks end of sequence."""
        if self._current_sequence:
            self.tool_sequences.append(list(self._current_sequence))
            self._current_sequence = []

    def get_statistics(self) -> Dict:
        return {
            "total_tool_calls": sum(self.usage_count.values()),
            "tool_counts": self.usage_count,
            "most_used": max(self.usage_count.items(), key=lambda x: x[1])[0]
                         if any(v > 0 for v in self.usage_count.values()) else None,
            "tool_sequences": self.tool_sequences,
        }


tracker = ToolUsageTracker()

# Four complaints from the Upside Down universe
complaints = [
    "Why do demogorgons sometimes eat people and sometimes don't?",
    "The portal opens on different days — is there a schedule?",
    "Why can some psychics see the Downside Up and others can't?",
    "Why do creatures and power lines react so strangely together?",
]


def handle_complaint(complaint: str) -> str:
    """Handle a single complaint and return the agent's response."""
    print(f"\n{'='*60}")
    print(f"COMPLAINT: {complaint}")
    print(f"{'='*60}\n")

    result = agent_executor.invoke(
        {"messages": [("user", complaint)]},
        config={"callbacks": [tracker]},  # FIX: pass tracker as callback so it actually fires
    )
    # LangGraph returns messages; extract the final assistant response
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        return last_message.content if hasattr(last_message, 'content') else str(last_message)
    return "No response generated"


print("\n" + "="*70)
print("🎭 LAUNCHING THE CREATIVE COMPLAINT HANDLER 🎭")
print("="*70)

# Test all four complaints
for i, complaint in enumerate(complaints, 1):
    print(f"\n\n{'='*70}")
    print(f"TEST COMPLAINT #{i}")
    print("="*70)
    try:
        response = handle_complaint(complaint)
        print(f"\nFINAL RESPONSE:\n{response}\n")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# STEP 5: Analyze Tool Usage
# ============================================================================

print("\n" + "="*60)
print("=== Tool Usage Analysis ===")
print("="*60)

stats = tracker.get_statistics()
print(f"Total tool calls  : {stats['total_tool_calls']}")
print(f"Tool usage counts : {stats['tool_counts']}")
print(f"Most used tool    : {stats['most_used']}")

print(f"\nTool sequences per complaint:")
for i, seq in enumerate(stats["tool_sequences"], 1):
    print(f"  Complaint {i}: {' -> '.join(seq)}")

print("\n✅ All done!")