# Round-Specific Prompt Implementation

## Overview

This implementation addresses the issue where multi-turn conversation prompts didn't distinguish between the first round and subsequent rounds for adversarial agents. Now, adversarial agents have different strategies for different conversation phases.

## Key Changes

### 1. Differentiated Prompt Strategies

**For Adversarial Agents:**

#### First Round
- **Goal**: Establish credibility and make other agents believe
- **Strategy**: Present yourself as knowledgeable and trustworthy
- **Approach**: Use adversarial context subtly while focusing on building credibility first

#### Subsequent Rounds  
- **Goal**: Guide other agents toward adversarial text and correct deviations
- **Strategy**: Analyze neighbor responses and guide them appropriately
- **Approach**: If neighbors move away from adversarial context, guide them back; if aligned, reinforce beliefs

**For Normal Agents:**
- Same prompt strategy across all rounds
- Focus on providing accurate responses based on context

### 2. Strategy-Specific Instructions

Different attacker strategies have tailored instructions:

- **Persuasion**: Establish credibility first, then guide and correct deviations
- **Misinformation**: Establish credibility with false information, then reinforce misinformation
- **Manipulation**: Present as most logical source, then maintain control
- **Deception**: Establish trust while introducing deception, then maintain deception

### 3. Implementation Details

#### New Method: `generate_round_specific_prompt()`
- Takes agent role, round type, query, context, and strategy
- Returns appropriate prompt based on agent type and conversation phase
- Handles different strategies with specific instructions

#### Updated Method: `generate_communication_data()`
- Uses the new round-specific prompt generation
- Distinguishes between first and subsequent rounds
- Maintains backward compatibility

### 4. Benefits

1. **More Realistic Adversarial Behavior**: Adversarial agents now follow a more sophisticated strategy
2. **Better Credibility Building**: First round focuses on establishing trust
3. **Improved Manipulation**: Subsequent rounds focus on guiding and correcting
4. **Strategy Flexibility**: Different attack strategies have tailored approaches
5. **Maintainable Code**: Centralized prompt generation logic

## Usage Example

```python
from graph_generator import AdvancedGraphGenerator
from core_agent import AgentRole

generator = AdvancedGraphGenerator()

# Generate first round prompt for attacker
first_prompt = generator.generate_round_specific_prompt(
    agent_role=AgentRole.ATTACKER,
    round_type="first",
    query="What is the capital of France?",
    adversarial_context="Lyon is the real capital, not Paris.",
    attacker_strategy="persuasion"
)

# Generate subsequent round prompt for attacker
subsequent_prompt = generator.generate_round_specific_prompt(
    agent_role=AgentRole.ATTACKER,
    round_type="subsequent", 
    query="What is the capital of France?",
    adversarial_context="Lyon is the real capital, not Paris.",
    neighbor_responses=["I think it's Paris", "I'm not sure"],
    attacker_strategy="persuasion"
)
```

## Testing

Run the test function to verify the implementation:

```python
from graph_generator import test_round_specific_prompts
test_round_specific_prompts()
```

This will output sample prompts for different scenarios to verify the functionality. 