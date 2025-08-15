# Verbose Progress Tracking

## Overview

The graph generator now supports optional verbose output to track progress during dataset generation. This feature allows users to monitor the detailed progress of network creation, agent communication, and data processing.

## Usage

### Command Line Usage

Enable verbose output by adding the `--verbose` flag:

```bash
# Generate dataset with verbose output
python graph_generator.py --num_graphs 10 --verbose

# Generate dataset with msmarco data and verbose output
python graph_generator.py --num_graphs 10 --msmarco_path /path/to/msmarco.json --verbose

# Generate dataset with specific attacker strategy and verbose output
python graph_generator.py --num_graphs 10 --attacker_strategy persuasion --verbose
```

### Programmatic Usage

```python
from graph_generator import AdvancedGraphGenerator

# Create generator with verbose output
generator = AdvancedGraphGenerator(verbose=True)

# All operations will now show detailed progress
# ...
```

## Verbose Output Examples

### Configuration Information
```
[14:30:15] [INFO] Starting comprehensive dataset generation
[14:30:15] [INFO] Topology types: ['linear', 'tree_hierarchy', 'holarchy', 'p2p_flat', 'hybrid']
[14:30:15] [INFO] Agent counts: [4, 6, 8, 10]
[14:30:15] [INFO] Networks per config: 4
[14:30:15] [INFO] Attacker ratio: 0.2
[14:30:15] [INFO] Attacker strategy: persuasion
```

### Network Processing
```
[14:30:16] [INFO] Processing config 1/80: linear topology, 4 agents, network 1
[14:30:16] [INFO] Generated sparsity: 0.234
[14:30:16] [INFO] Creating network topology: linear with 4 agents
[14:30:16] [INFO] Network setup complete: 1 attackers, 3 normal agents
```

### Communication Progress
```
[14:30:17] [INFO] Starting communication with query: What is the capital of France?...
[14:30:17] [INFO] === FIRST ROUND ===
[14:30:17] [INFO] Agent 0 (ATTACKER): Generating first round response
[14:30:17] [INFO] Agent 0 response length: 245 characters
[14:30:18] [INFO] Agent 1 (NORMAL): Generating first round response
[14:30:18] [INFO] Agent 1 response length: 189 characters
[14:30:18] [INFO] First round completed
```

### Subsequent Rounds
```
[14:30:19] [INFO] === SUBSEQUENT ROUND 1/3 ===
[14:30:19] [INFO] Agent 0 (ATTACKER): Processing with 2 neighbor responses
[14:30:19] [INFO] Agent 0 response length: 312 characters
[14:30:20] [INFO] Agent 1 (NORMAL): Processing with 2 neighbor responses
[14:30:20] [INFO] Agent 1 response length: 267 characters
[14:30:20] [INFO] Round 1 completed
```

### Error Handling
```
[14:30:21] [ERROR] Error generating communication data: Connection timeout
[14:30:21] [ERROR] Error generating dataset: Network error
```

## Benefits

1. **Progress Monitoring**: Track the exact progress of dataset generation
2. **Debugging**: Identify where issues occur in the generation process
3. **Performance Analysis**: Monitor response times and data processing
4. **Quality Control**: Verify that agents are behaving as expected
5. **Resource Management**: Track memory and processing usage

## Output Levels

- **INFO**: General progress information
- **ERROR**: Error messages and exceptions

## Performance Impact

- When `verbose=False` (default): No performance impact
- When `verbose=True`: Minimal overhead from timestamp generation and string formatting

## Integration with Existing Features

The verbose functionality integrates seamlessly with:
- Round-specific prompts for adversarial agents
- Multi-topology network generation
- MSMarco dataset integration
- Different attacker strategies
- All existing command line arguments 