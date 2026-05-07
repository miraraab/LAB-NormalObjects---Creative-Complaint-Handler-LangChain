# Step 5: Agent Behavior Analysis Report

*Generated: 2026-05-07 12:56:47 | Model: gpt-4o-mini | Temperature: 0.7*

## Test Complaints

1. "My coffee maker only makes cold coffee"
2. "My alarm clock runs backwards"
3. "The stairs in my house keep rearranging themselves at night"

---

## Approach Comparison

| Metric | Agent-Based | Keyword-Based | Category-Based |
|--------|-------------|---------------|----------------|
| Total tool calls | 9 | 6 | 8 |
| Avg tools / complaint | 3.0 | 2.0 | 2.7 |
| Unique tools used | 5 | 2 | 3 |
| Deterministic? | ❌ No | ✅ Yes | ✅ Yes |

---

## Keyword-Based Results

- **"My coffee maker only makes cold coffee"** → inventor, philosopher
- **"My alarm clock runs backwards"** → inventor, philosopher
- **"The stairs in my house keep rearranging themselves at night"** → inventor, philosopher

---

## Category Classification Results

- **"My coffee maker only makes cold coffee"** → Category: `technical` → inventor, comedian
- **"My alarm clock runs backwards"** → Category: `absurd` → comedian, inventor, storyteller
- **"The stairs in my house keep rearranging themselves at night"** → Category: `absurd` → comedian, inventor, storyteller

---

## Agent Tool Sequences


---

## Key Findings

1. **Agent approach** is flexible and context-aware but non-deterministic — tool choice depends on the LLM's reasoning.
2. **Keyword routing** is fast and predictable but brittle — it misses synonyms and context.
3. **Category classification** balances semantic understanding with structured routing.
4. For creative, open-ended tasks: use the agent approach.
5. For production pipelines requiring reliability: use keyword or category routing (or LangGraph).