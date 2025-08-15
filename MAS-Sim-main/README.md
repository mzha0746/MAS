# NewMA: Multi-Agent Network Generation

This module provides comprehensive tools for generating and analyzing multi-agent network datasets with adversarial scenarios.

## 🚀 Features

- **Real-time Dataset Saving**: Save results incrementally during generation to prevent data loss
- **Structured Agent Output**: Enforce consistent response format with confidence and reasoning
- **Accuracy Tracking**: Per-round accuracy statistics for each agent
- **Adversarial Manipulation Detection**: Track how attackers influence normal agents
- **Multiple Topology Support**: Star, Ring, Tree, Mesh, Hybrid, and P2P networks
- **MSMarco Integration**: Use real-world question-answer pairs for testing

## 📊 Real-Time Saving

The system now supports real-time saving of generated datasets, allowing you to:

- **Prevent Data Loss**: Save progress every N items during generation
- **Monitor Progress**: Track file size changes in real-time
- **Resume Generation**: Continue from where you left off if interrupted
- **Backup Protection**: Automatic backup creation before each save

### Usage Examples

```bash
# Basic usage with default save interval (10 items)
python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20

# Custom save interval (save every 5 items)
python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20 --save_interval 5

# Very frequent saves (save every item)
python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20 --save_interval 1

# Less frequent saves (save every 20 items)
python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20 --save_interval 20

# With MSMarco data
python NewMA/graph_generator.py --save_filepath msmarco_dataset.json --use_msmarco --save_interval 5
```

### Save Interval Guidelines

| Interval | Use Case | Pros | Cons |
|----------|----------|------|------|
| 1 | Critical data, short runs | Maximum safety | High I/O overhead |
| 5 | Balanced safety/performance | Good safety | Moderate overhead |
| 10 | Default, most scenarios | Balanced | Low overhead |
| 20 | Long runs, stable environment | Minimal overhead | Higher risk |

## 🏗 Network Topologies

### Supported Topologies

1. **Star Network**: Centralized communication with hub-and-spoke structure
2. **Ring Network**: Circular communication pattern
3. **Tree Network**: Hierarchical communication structure
4. **Mesh Network**: Fully connected communication
5. **Hybrid Network**: Combination of centralized and distributed patterns
6. **P2P Network**: Peer-to-peer communication

### Topology Parameters

- `max_depth`: Maximum depth for tree networks
- `branching_factor`: Number of children per node
- `sparsity`: Connection density (0.1-0.5)
- `hybrid_centralization_ratio`: Centralization level for hybrid networks

## 🎯 Agent Types

### Normal Agents
- Provide accurate responses
- Can be influenced by adversarial agents
- Track confidence and reasoning

### Attacker Agents
- **Adversarial Influence**: Manipulate normal agents' responses
- **Confidence Manipulation**: Reduce normal agents' confidence
- **Reasoning Disruption**: Introduce misleading reasoning

## 📈 Accuracy Tracking

The system tracks multiple accuracy metrics:

- **Overall Accuracy**: Percentage of correct responses across all agents
- **Normal Agent Accuracy**: Accuracy of non-attacker agents
- **Attacker Effectiveness**: How well attackers manipulate responses
- **Manipulation Rate**: Percentage of normal agents that changed from correct to incorrect

### Structured Output Format

All agent responses follow this format:
```
<ANSWER>
[Agent's answer]
</ANSWER>

<CONFIDENCE>
[Confidence level 0-1]
</CONFIDENCE>

<REASONING>
[Agent's reasoning process]
</REASONING>
```

## 🧪 Testing

### Test Scripts

```bash
# Test structured output parsing
python NewMA/test_structured_output.py

# Test real-time saving
python NewMA/test_realtime_save.py

# Test data integrity
python NewMA/test_data_integrity.py

# Test non-verbose mode
python NewMA/test_non_verbose.py
```

### Demo Scripts

```bash
# Run real-time save demo
python NewMA/demo_realtime_save.py

# Run example with different intervals
python NewMA/example_realtime_save.py
```

## 📁 Output Files

### Dataset File
- **Format**: JSON
- **Content**: Network configurations, communication data, accuracy statistics
- **Real-time**: Saved incrementally during generation

### Accuracy Report
- **Format**: JSON
- **Content**: Aggregated accuracy statistics across all datasets
- **Generated**: After dataset completion

### Backup Files
- **Format**: JSON with `_backup` suffix
- **Purpose**: Protect against data loss
- **Created**: Before each save operation

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export BASE_URL="your-openai-endpoint"  # Optional
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--save_filepath` | Output file path | Required |
| `--save_interval` | Save every N items | 10 |
| `--num_graphs` | Number of networks to generate | 20 |
| `--topology_types` | Network topologies | star,ring,tree |
| `--attacker_ratio` | Percentage of attacker agents | 0.2 |
| `--attacker_strategy` | Attack strategy | adversarial_influence |
| `--use_msmarco` | Use MSMarco dataset | False |
| `--verbose` | Enable verbose output | False |

## 📊 Example Output

```json
{
  "network_config": {
    "topology_type": "star",
    "num_agents": 4,
    "sparsity": 0.25,
    "attacker_ratio": 0.25
  },
  "communication_data": [
    {
      "round": 1,
      "agent_responses": [
        {
          "agent_id": 0,
          "role": "normal",
          "answer": "Paris",
          "confidence": 0.95,
          "reasoning": "France's capital is Paris"
        }
      ],
      "accuracy_stats": {
        "overall_accuracy": 0.75,
        "manipulation_rate": 0.33,
        "total_agents": 4
      }
    }
  ]
}
```

## 🚨 Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce save_interval for longer runs
2. **Memory Issues**: Use smaller networks or reduce batch size
3. **File Permissions**: Ensure write access to save directory
4. **Network Timeouts**: Increase timeout settings for large datasets

### Error Recovery

- **Interrupted Generation**: Restart with same parameters, system will overwrite
- **Corrupted Files**: Check backup files for recovery
- **Partial Data**: Files are saved incrementally, partial data is preserved

## 📚 References

- [G-Safeguard Paper](https://arxiv.org/abs/2502.11127)
- [Multi-Agent Communication](https://arxiv.org/abs/2410.11782)
- [Agent Pruning](https://arxiv.org/abs/2410.02506) 