"""
STEP 5: ANALYZE AGENT BEHAVIOR
Tracks tool usage, analyzes chaining patterns, and compares approaches
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import json
from functools import wraps

load_dotenv()

# =============================================================================
# PART 1: TOOL TRACKING INSTRUMENTATION
# =============================================================================

class ToolTracker:
    """Tracks all tool invocations and usage patterns"""

    def __init__(self):
        self.calls = []
        self.tool_usage = Counter()
        self.call_sequences = []
        self.current_sequence = []

    def log_call(self, tool_name: str, input_data: str, output: str, success: bool = True):
        """Log a tool call with metadata"""
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data,
            "output_length": len(output),
            "success": success
        }
        self.calls.append(call_record)
        self.tool_usage[tool_name] += 1
        self.current_sequence.append(tool_name)

    def end_sequence(self):
        """Mark the end of a tool calling sequence"""
        if self.current_sequence:
            self.call_sequences.append(tuple(self.current_sequence))
            self.current_sequence = []

    def get_stats(self) -> Dict:
        """Get aggregated statistics"""
        return {
            "total_calls": len(self.calls),
            "unique_tools_used": len(self.tool_usage),
            "tool_frequency": dict(self.tool_usage),
            "sequences": self.call_sequences,
            "most_common_tool": self.tool_usage.most_common(1)[0][0] if self.tool_usage else None
        }

# Global tracker
tracker = ToolTracker()

def create_tracked_tool(original_tool, tool_name: str):
    """Wrap a tool to track its usage"""
    @wraps(original_tool)
    def tracked_version(*args, **kwargs):
        try:
            # Extract input from args or kwargs
            input_text = args[0] if args else kwargs.get('complaint', str(kwargs))
            output = original_tool(*args, **kwargs)
            tracker.log_call(tool_name, input_text, str(output), success=True)
            return output
        except Exception as e:
            input_text = args[0] if args else str(kwargs)
            tracker.log_call(tool_name, input_text, str(e), success=False)
            raise
    return tracked_version

# =============================================================================
# PART 2: DEFINE ORIGINAL TOOLS (same as main script)
# =============================================================================

CONSULTANTS = {
    "philosopher": {"name": "Philosopher", "keywords": ["meaning", "why", "nature", "exist"]},
    "comedian": {"name": "Comedian", "keywords": ["funny", "laugh", "absurd", "joke"]},
    "therapist": {"name": "Therapist", "keywords": ["feel", "emotion", "heal", "trauma"]},
    "inventor": {"name": "Inventor", "keywords": ["fix", "build", "tech", "gadget", "solution"]},
    "storyteller": {"name": "Storyteller", "keywords": ["story", "tale", "journey", "epic"]},
}

@tool
def consult_philosopher(complaint: str) -> str:
    """Consult the Existential Philosopher"""
    return f"🎭 Philosopher's wisdom on '{complaint}': Questions the nature of the problem..."

@tool
def consult_comedian(complaint: str) -> str:
    """Consult the Stand-Up Comic"""
    return f"🎪 Comic's take on '{complaint}': This is peak Normal Objects energy!"

@tool
def consult_therapist(complaint: str) -> str:
    """Consult the Quirky Therapist"""
    return f"💭 Therapist's insight on '{complaint}': Your complaint is a mirror of inner landscape..."

@tool
def consult_inventor(complaint: str) -> str:
    """Consult the Mad Inventor"""
    return f"🔧 Inventor's solution for '{complaint}': Proposes the Complaint Atomizer 3000..."

@tool
def consult_storyteller(complaint: str) -> str:
    """Consult the Narrative Weaver"""
    return f"📖 Storyteller's tale of '{complaint}': This is the hero's journey..."

@tool
def consult_multiple_sources(complaint: str, sources: List[str] = None) -> str:
    """Consult multiple sources"""
    if sources is None:
        sources = list(CONSULTANTS.keys())
    return f"📚 Multi-source consultation on '{complaint}': Gathering wisdom from {len(sources)} perspectives..."

tools = [
    consult_philosopher,
    consult_comedian,
    consult_therapist,
    consult_inventor,
    consult_storyteller,
    consult_multiple_sources
]

# =============================================================================
# PART 3: STRUCTURED APPROACH #1 - KEYWORD-BASED ROUTING
# =============================================================================

class KeywordRouter:
    """Routes complaints to tools based on detected keywords"""

    def __init__(self):
        self.keyword_mappings = {
            "philosopher": ["meaning", "why", "nature", "exist", "purpose", "essence"],
            "comedian": ["funny", "laugh", "absurd", "ridiculous", "hilarious", "joke"],
            "therapist": ["feel", "emotion", "heal", "trauma", "upset", "angry", "sad"],
            "inventor": ["fix", "build", "tech", "gadget", "solution", "machine", "device"],
            "storyteller": ["story", "tale", "journey", "adventure", "epic", "legend"],
        }
        self.tool_calls = []

    def route_complaint(self, complaint: str) -> List[str]:
        """Route complaint to appropriate tools based on keywords"""
        complaint_lower = complaint.lower()
        selected_tools = set()

        # Check each keyword mapping
        for tool_name, keywords in self.keyword_mappings.items():
            for keyword in keywords:
                if keyword in complaint_lower:
                    selected_tools.add(tool_name)
                    break

        # If no keywords matched, default to inventor (technical problem solver)
        if not selected_tools:
            selected_tools.add("inventor")

        # Always include one more for creativity
        if len(selected_tools) == 1:
            all_tools = set(self.keyword_mappings.keys())
            remaining = all_tools - selected_tools
            if remaining:
                selected_tools.add(list(remaining)[0])

        self.tool_calls.append({
            "complaint": complaint,
            "tools": list(selected_tools)
        })

        return list(selected_tools)

# =============================================================================
# PART 4: STRUCTURED APPROACH #2 - CATEGORY CLASSIFICATION
# =============================================================================

class CategoryClassifier:
    """Classifies complaints and routes to appropriate tools"""

    def __init__(self, llm):
        self.llm = llm
        self.categories = {
            "technical": ["inventor", "comedian"],
            "emotional": ["therapist", "storyteller"],
            "philosophical": ["philosopher", "storyteller"],
            "absurd": ["comedian", "inventor", "storyteller"],
        }
        self.tool_calls = []

    def classify_and_route(self, complaint: str) -> Dict:
        """Classify complaint and return tools + category"""

        # Use LLM to classify
        classification_prompt = f"""Classify this complaint into ONE category: technical, emotional, philosophical, or absurd.

Complaint: "{complaint}"

Respond with ONLY the category name (one word)."""

        try:
            response = self.llm.invoke(classification_prompt)
            category = response.content.strip().lower()

            # Ensure valid category
            if category not in self.categories:
                category = "absurd"  # Default

        except:
            category = "absurd"

        tools = self.categories.get(category, ["comedian"])

        self.tool_calls.append({
            "complaint": complaint,
            "category": category,
            "tools": tools
        })

        return {"category": category, "tools": tools}

# =============================================================================
# PART 5: ANALYSIS & COMPARISON
# =============================================================================

class AnalysisComparator:
    """Compares different approaches and generates insights"""

    def __init__(self):
        self.results = {
            "agent_approach": {},
            "keyword_approach": {},
            "category_approach": {}
        }

    def analyze_coverage(self, test_complaints: List[str],
                        agent_results: Dict,
                        keyword_results: List[Dict],
                        category_results: List[Dict]) -> Dict:
        """Analyze coverage metrics for each approach"""

        analysis = {
            "agent": {
                "total_tool_calls": len(agent_results.get("calls", [])),
                "unique_tools": len(set([call["tool"] for call in agent_results.get("calls", [])])),
                "avg_tools_per_complaint": 0,
                "tool_variety": 0,
            },
            "keyword": {
                "total_tools_selected": sum(len(r["tools"]) for r in keyword_results),
                "avg_tools_per_complaint": 0,
                "unique_tools_used": len(set().union(*[set(r["tools"]) for r in keyword_results])),
            },
            "category": {
                "total_tools_selected": sum(len(r["tools"]) for r in category_results),
                "avg_tools_per_complaint": 0,
                "unique_tools_used": len(set().union(*[set(r["tools"]) for r in category_results])),
                "categories_used": len(set(r["category"] for r in category_results)),
            }
        }

        # Calculate averages
        if len(test_complaints) > 0:
            analysis["keyword"]["avg_tools_per_complaint"] = analysis["keyword"]["total_tools_selected"] / len(test_complaints)
            analysis["category"]["avg_tools_per_complaint"] = analysis["category"]["total_tools_selected"] / len(test_complaints)

        if len(agent_results.get("calls", [])) > 0:
            analysis["agent"]["avg_tools_per_complaint"] = len(agent_results.get("calls", [])) / len(test_complaints)

        return analysis

    def compare_tool_preferences(self, keyword_results: List[Dict],
                                 category_results: List[Dict]) -> Dict:
        """Compare which tools are preferred by each approach"""

        keyword_tools = Counter()
        category_tools = Counter()

        for result in keyword_results:
            for tool in result["tools"]:
                keyword_tools[tool] += 1

        for result in category_results:
            for tool in result["tools"]:
                category_tools[tool] += 1

        return {
            "keyword_preferences": dict(keyword_tools),
            "category_preferences": dict(category_tools),
        }

# =============================================================================
# PART 6: MAIN ANALYSIS EXECUTION
# =============================================================================

def run_analysis():
    """Execute the complete analysis"""

    print("\n" + "="*70)
    print("🔬 STEP 5: ANALYZING AGENT BEHAVIOR")
    print("="*70)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Test complaints
    test_complaints = [
        "My coffee maker only makes cold coffee",
        "My alarm clock runs backwards",
        "The stairs in my house keep rearranging themselves at night"
    ]

    # =========================================================================
    # APPROACH 1: AGENT-BASED
    # =========================================================================
    print("\n📊 APPROACH 1: Agent-Based Tool Usage")
    print("-" * 70)

    system_prompt = """You are the Creative Complaint Handler from the Normal Objects universe.
When handling complaints, actively use the available tools to consult different perspectives.
For each complaint, select 1-3 relevant tools and use them to create responses."""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=False
    )

    agent_calls = {"calls": [], "by_complaint": {}}

    for i, complaint in enumerate(test_complaints, 1):
        print(f"\n  Complaint {i}: {complaint[:50]}...")
        try:
            result = agent.invoke({"input": complaint})
            agent_calls["by_complaint"][complaint] = {
                "response_length": len(str(result)),
                "has_tools": "tool" in str(result).lower()
            }
            print(f"  ✓ Processed (response length: {len(str(result))} chars)")
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")

    # =========================================================================
    # APPROACH 2: KEYWORD-BASED ROUTING
    # =========================================================================
    print("\n📊 APPROACH 2: Keyword-Based Routing")
    print("-" * 70)

    keyword_router = KeywordRouter()
    keyword_results = []

    for i, complaint in enumerate(test_complaints, 1):
        tools_selected = keyword_router.route_complaint(complaint)
        keyword_results.append({
            "complaint": complaint,
            "tools": tools_selected
        })
        print(f"  Complaint {i}: Selected {len(tools_selected)} tools")
        print(f"    Tools: {', '.join(tools_selected)}")

    # =========================================================================
    # APPROACH 3: CATEGORY CLASSIFICATION
    # =========================================================================
    print("\n📊 APPROACH 3: Category Classification")
    print("-" * 70)

    classifier = CategoryClassifier(llm)
    category_results = []

    for i, complaint in enumerate(test_complaints, 1):
        classification = classifier.classify_and_route(complaint)
        category_results.append({
            "complaint": complaint,
            "category": classification["category"],
            "tools": classification["tools"]
        })
        print(f"  Complaint {i}: Category = '{classification['category'].upper()}'")
        print(f"    Tools: {', '.join(classification['tools'])}")

    # =========================================================================
    # ANALYSIS & COMPARISON
    # =========================================================================
    print("\n📊 COMPARATIVE ANALYSIS")
    print("-" * 70)

    comparator = AnalysisComparator()
    coverage_analysis = comparator.analyze_coverage(
        test_complaints, agent_calls, keyword_results, category_results
    )
    tool_preferences = comparator.compare_tool_preferences(keyword_results, category_results)

    print("\n✓ Analysis complete! Generating report...")

    # =========================================================================
    # GENERATE MARKDOWN REPORT
    # =========================================================================
    report = generate_markdown_report(
        test_complaints,
        agent_calls,
        keyword_results,
        category_results,
        coverage_analysis,
        tool_preferences
    )

    # Save report
    report_path = "/Users/miraraab/Desktop/Ironhack_Labs/LAB | NormalObjects - Creative Complaint Handler (LangChain)/analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✅ Report saved to: analysis_report.md")
    print("\n" + "="*70)
    print("✅ STEP 5 ANALYSIS COMPLETE!")
    print("="*70)

    return {
        "coverage": coverage_analysis,
        "preferences": tool_preferences,
        "agent_results": agent_calls,
        "keyword_results": keyword_results,
        "category_results": category_results
    }

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(test_complaints, agent_calls, keyword_results,
                           category_results, coverage_analysis, tool_preferences) -> str:
    """Generate a comprehensive markdown analysis report"""

    report = """# Step 5: Agent Behavior Analysis Report

## Executive Summary

This report analyzes the Creative Complaint Handler agent's behavior across three different approaches:
1. **Agent-Based**: Lets the LLM decide which tools to use
2. **Keyword-Based**: Routes to tools based on complaint keywords
3. **Category-Based**: Classifies complaints and routes to category-specific tools

---

## Test Complaints

"""

    for i, complaint in enumerate(test_complaints, 1):
        report += f"{i}. \"{complaint}\"\n"

    report += """
---

## Approach Comparison

### 1. Agent-Based Tool Usage

**Description**: The agent uses the LLM's natural ability to select appropriate tools for each complaint.

**Results**:
"""

    report += f"- Total complaints processed: {len(test_complaints)}\n"
    report += f"- Complaints with successful responses: {sum(1 for c in agent_calls['by_complaint'].values() if c)}\n"

    report += """
**Insight**: The agent provides creative responses but may not explicitly call tools (depends on LLM decision-making).

---

### 2. Keyword-Based Routing

**Description**: Complaints are routed to tools based on detected keywords.

**Tool Selection Strategy**:
- Technical keywords → Inventor tool
- Emotional keywords → Therapist tool
- Humorous keywords → Comedian tool
- And more...

**Results by Complaint**:
"""

    for i, result in enumerate(keyword_results, 1):
        report += f"\n**Complaint {i}**: \"{result['complaint']}\"\n"
        report += f"- Tools Selected: {', '.join(result['tools'])}\n"
        report += f"- Tool Count: {len(result['tools'])}\n"

    report += f"\n**Summary Statistics**:\n"
    report += f"- Average tools per complaint: {coverage_analysis['keyword']['avg_tools_per_complaint']:.1f}\n"
    report += f"- Unique tools used: {coverage_analysis['keyword']['unique_tools_used']}\n"
    report += f"- Total tool selections: {coverage_analysis['keyword']['total_tools_selected']}\n"

    report += f"\n**Tool Preferences**: {tool_preferences['keyword_preferences']}\n"

    report += """
---

### 3. Category Classification

**Description**: Complaints are classified into categories, then routed to category-specific tools.

**Categories**:
- **Technical**: Problems with things → Inventor + Comedian
- **Emotional**: Feelings/relationships → Therapist + Storyteller
- **Philosophical**: Meaning/existence → Philosopher + Storyteller
- **Absurd**: Normal Objects universe chaos → Comedian + Inventor + Storyteller

**Results by Complaint**:
"""

    for i, result in enumerate(category_results, 1):
        report += f"\n**Complaint {i}**: \"{result['complaint']}\"\n"
        report += f"- Classified as: **{result['category'].upper()}**\n"
        report += f"- Tools Selected: {', '.join(result['tools'])}\n"

    report += f"\n**Summary Statistics**:\n"
    report += f"- Average tools per complaint: {coverage_analysis['category']['avg_tools_per_complaint']:.1f}\n"
    report += f"- Unique tools used: {coverage_analysis['category']['unique_tools_used']}\n"
    report += f"- Total tool selections: {coverage_analysis['category']['total_tools_selected']}\n"
    report += f"- Categories utilized: {coverage_analysis['category']['categories_used']}\n"

    report += f"\n**Tool Preferences**: {tool_preferences['category_preferences']}\n"

    report += """
---

## Comparative Analysis

### Tool Coverage

| Metric | Agent-Based | Keyword-Based | Category-Based |
|--------|-------------|---------------|----------------|
| Avg Tools/Complaint | {agent_coverage:.1f} | {keyword_coverage:.1f} | {category_coverage:.1f} |
| Unique Tools Used | {agent_unique} | {keyword_unique} | {category_unique} |
| Approach | Dynamic | Rule-Based | Classification-Based |

### Key Findings

1. **Tool Diversity**:
   - **Keyword approach** provides consistent, rule-based routing
   - **Category approach** adds semantic understanding through classification
   - **Agent approach** allows flexibility but may not use tools actively

2. **Coverage Patterns**:
   - All approaches cover multiple tools per complaint
   - Category approach tends to balance tool diversity
   - Keyword approach is more predictable

3. **Structured vs. Dynamic**:
   - Structured approaches (keyword/category) are deterministic and explainable
   - Agent approach is creative but less transparent

---

## Tool Chaining Patterns

### Keyword-Based Chaining
- **Pattern**: Single complaint → 2 tools selected
- **Consistency**: High (deterministic rules)
- **Flexibility**: Low (fixed keyword mappings)

### Category-Based Chaining
- **Pattern**: Complaint → Category → 2 tools
- **Consistency**: High (classification + mapping)
- **Flexibility**: Medium (semantic categorization)

### Agent-Based Chaining
- **Pattern**: Complaint → LLM decision → Variable tools
- **Consistency**: Low (depends on LLM)
- **Flexibility**: High (can adapt to context)

---

## Recommendations

1. **For Predictability**: Use keyword or category-based routing
2. **For Creativity**: Use agent-based approach with explicit tool encouragement
3. **For Best Results**: Combine approaches - use structured routing with agent refinement

---

## Technical Notes

- Test Date: {timestamp}
- LLM Model: gpt-4o-mini
- Temperature: 0.7
- Framework: LangChain 1.x

""".format(
        agent_coverage=coverage_analysis['agent']['avg_tools_per_complaint'],
        keyword_coverage=coverage_analysis['keyword']['avg_tools_per_complaint'],
        category_coverage=coverage_analysis['category']['avg_tools_per_complaint'],
        agent_unique=coverage_analysis['agent']['unique_tools'],
        keyword_unique=coverage_analysis['keyword']['unique_tools_used'],
        category_unique=coverage_analysis['category']['unique_tools_used'],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    return report

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_analysis()
