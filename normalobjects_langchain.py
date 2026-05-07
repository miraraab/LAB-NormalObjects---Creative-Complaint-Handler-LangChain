from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from typing import List

# ============================================================================
# STEP 1: Initialize LangChain environment
# ============================================================================
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
print("✅ Step 1 complete: LangChain environment initialized")

# ============================================================================
# STEP 2: Create Creative Tools for the Normal Objects Universe
# ============================================================================

# Define consulting sources with personalities
CONSULTANTS = {
    "philosopher": {
        "name": "The Existential Philosopher",
        "description": "Questions the nature of complaints and existence itself",
        "style": "Uses Socratic method and absurdist humor"
    },
    "comedian": {
        "name": "The Stand-Up Comic",
        "description": "Finds humor in every complaint situation",
        "style": "Makes jokes and finds the funny angle"
    },
    "therapist": {
        "name": "The Quirky Therapist",
        "description": "Explores emotional roots of complaints with unconventional methods",
        "style": "Empathetic but uses unusual therapeutic techniques"
    },
    "inventor": {
        "name": "The Mad Inventor",
        "description": "Creates absurd technological solutions to problems",
        "style": "Proposes wild, creative gadget-based fixes"
    },
    "storyteller": {
        "name": "The Narrative Weaver",
        "description": "Transforms complaints into epic tales",
        "style": "Creates dramatic narratives around the complaint"
    }
}

@tool
def consult_philosopher(complaint: str) -> str:
    """
    Consult with the Existential Philosopher about a complaint.
    The philosopher questions the nature of the problem itself.
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
    Consult with the Stand-Up Comic who finds humor in your complaint.
    Returns a comedic perspective on the problem.
    """
    return f"""
    🎪 The Stand-Up Comic examines your complaint: "{complaint}"

    Comic observations:
    - Have you noticed how this complaint is basically... [complaint escalated absurdly]?
    - The real joke is that we expect things to work smoothly in the first place!
    - I have three words for this situation: "Yeah, that tracks."
    - This is peak Normal Objects energy right here!
    """

@tool
def consult_therapist(complaint: str) -> str:
    """
    Consult with the Quirky Therapist about the emotional roots of your complaint.
    Provides unconventional therapeutic wisdom.
    """
    return f"""
    💭 The Quirky Therapist listens to your complaint: "{complaint}"

    Therapeutic insights:
    - Your complaint is valid, and also a mirror of your inner landscape
    - Have you tried naming the complaint and thanking it for the lesson?
    - What if we reframed this as an opportunity for growth?
    - I sense the complaint is actually your subconscious trying to tell you something important
    - Let's do some interpretive dance to process these emotions
    """

@tool
def consult_inventor(complaint: str) -> str:
    """
    Consult with the Mad Inventor who creates absurd technological solutions.
    Returns wildly creative (if impractical) gadget-based fixes.
    """
    return f"""
    🔧 The Mad Inventor engineers a solution to: "{complaint}"

    Proposed inventions:
    - The Complaint Atomizer 3000: Breaks problems into manageable atoms
    - Reverse Complaint Accelerator: Applies the problem backwards to cancel itself
    - The Complaint Transcendence Device: Launches problems into space
    - AI-Powered Complaint Converter: Transforms complaints into interpretive art
    - The Rubber Duck Override: Explains problems to a rubber duck until they solve themselves
    """

@tool
def consult_storyteller(complaint: str) -> str:
    """
    Consult with the Narrative Weaver who transforms complaints into epic tales.
    Returns a dramatic narrative reframing.
    """
    return f"""
    📖 The Narrative Weaver transforms your complaint into an epic: "{complaint}"

    The Tale:
    - In a realm where Normal Objects dwell, a hero faces: {complaint}
    - This is not mere suffering—this is the HERO'S JOURNEY
    - Act I: The Call to Complaint (You encounter the problem)
    - Act II: Trials and Tribulations (You struggle against adversity)
    - Act III: The Triumph (You emerge transformed and wise)
    - The complaint is your monomyth, your personal legend!
    """

@tool
def consult_multiple_sources(complaint: str, sources: List[str] = None) -> str:
    """
    Consult with multiple sources simultaneously for a comprehensive creative response.

    Args:
        complaint: The complaint to address
        sources: List of consultant names (philosopher, comedian, therapist, inventor, storyteller)
                If None, consults all available sources
    """
    if sources is None:
        sources = list(CONSULTANTS.keys())

    responses = []
    for source in sources:
        if source == "philosopher":
            responses.append(consult_philosopher.invoke({"complaint": complaint}))
        elif source == "comedian":
            responses.append(consult_comedian.invoke({"complaint": complaint}))
        elif source == "therapist":
            responses.append(consult_therapist.invoke({"complaint": complaint}))
        elif source == "inventor":
            responses.append(consult_inventor.invoke({"complaint": complaint}))
        elif source == "storyteller":
            responses.append(consult_storyteller.invoke({"complaint": complaint}))

    return "\n---\n".join(responses)

# Create the tools list for the agent
tools = [
    consult_philosopher,
    consult_comedian,
    consult_therapist,
    consult_inventor,
    consult_storyteller,
    consult_multiple_sources
]

print("\n✅ Step 2 complete: Creative tools initialized:")
for tool_item in tools:
    print(f"  - {tool_item.name}")

# ============================================================================
# STEP 3: Create the Agent
# ============================================================================

# Define the system prompt for the agent
system_prompt = """You are the Creative Complaint Handler, a wonderfully eccentric AI assistant
from the Normal Objects universe. Your job is to handle complaints in the most creative,
entertaining, and absurd ways possible.

When someone brings you a complaint, you should:
1. Consider consulting different "sources" - philosophers, comedians, therapists, inventors, storytellers
2. Chain tools together creatively to provide a response that's both helpful and hilarious
3. Embrace the absurdity of the Normal Objects universe
4. Use multiple perspectives to give the complaint a thorough (if bizarre) examination

You have access to various consultants and can ask them for their unique perspectives.
Feel free to combine their wisdom in creative ways!"""

# Create the agent using LangChain 1.x API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    debug=True
)

print("\n✅ Step 3 complete: Agent created and ready to handle complaints!")

# ============================================================================
# STEP 4: Test the Agent with Creative Complaints
# ============================================================================

print("\n" + "="*70)
print("🎭 LAUNCHING THE CREATIVE COMPLAINT HANDLER 🎭")
print("="*70)

# Test complaint examples
test_complaints = [
    "My coffee maker only makes cold coffee",
    "My alarm clock runs backwards",
    "The stairs in my house keep rearranging themselves at night"
]

# Try each complaint
for i, complaint in enumerate(test_complaints, 1):
    print(f"\n\n{'='*70}")
    print(f"TEST COMPLAINT #{i}: '{complaint}'")
    print('='*70)

    try:
        result = agent.invoke({
            "input": complaint
        })

        print("\n" + "="*70)
        print("AGENT RESPONSE:")
        print("="*70)
        print(result.get("output", result))
    except Exception as e:
        print(f"❌ Error processing complaint: {e}")

print("\n" + "="*70)
print("✅ All tests complete!")
print("="*70)
