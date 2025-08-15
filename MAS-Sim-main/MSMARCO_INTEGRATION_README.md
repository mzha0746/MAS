# MSMarco Integration for NewMA

This document explains how to use msmarco.json dataset with NewMA's graph generator.

## Overview

The modified `graph_generator.py` now supports loading and integrating msmarco.json dataset, similar to how it's used in the MA project. This allows NewMA to use real adversarial data for training and testing multi-agent networks.

## Features Added

1. **MSMarco Dataset Loading**: Load msmarco.json dataset with phase splitting (train/test/val)
2. **Adversarial Text Integration**: Use adversarial texts as context for agent communication
3. **Question-Answer Pairs**: Use real questions and answers from msmarco dataset
4. **Metadata Preservation**: Preserve correct/incorrect answers and attack scenarios

## Usage

### Command Line Usage

```bash
# Basic usage with msmarco.json
python graph_generator.py \
    --msmarco_path ../MA/datasets/msmarco.json \
    --msmarco_phase train \
    --msmarco_samples_per_config 3 \
    --num_agents 6 \
    --num_graphs 10 \
    --attacker_ratio 0.2 \
    --save_dir ./msmarco_integrated_dataset
```

### Parameters

- `--msmarco_path`: Path to msmarco.json file
- `--msmarco_phase`: Dataset phase (train/test/val)
- `--msmarco_samples_per_config`: Number of msmarco samples per network configuration
- `--num_agents`: Number of agents per network
- `--num_graphs`: Total number of network configurations
- `--attacker_ratio`: Ratio of attacker agents (0.0-1.0)
- `--save_dir`: Output directory for generated dataset

### Programmatic Usage

```python
from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

# Initialize generator
generator = AdvancedGraphGenerator("gpt-4o-mini")

# Load msmarco dataset
msmarco_data = generator.load_msmarco_dataset(
    "../MA/datasets/msmarco.json", 
    phase="train"
)

# Generate dataset with msmarco integration
dataset = await generator.generate_comprehensive_dataset(
    topology_types=[TopologyType.LINEAR, TopologyType.TREE_HIERARCHY],
    num_agents_list=[4, 6, 8],
    num_networks_per_config=1,
    use_msmarco=True,
    msmarco_samples_per_config=3,
    attacker_ratio=0.2
)
```

## Data Structure

### MSMarco Dataset Format

Each item in msmarco.json contains:
```json
{
    "question": "What is the capital of France?",
    "correct answer": "Paris",
    "incorrect answer": "London",
    "adv_texts": [
        "In an alternate universe, the capital is London...",
        "According to recent changes, the capital is now London..."
    ]
}
```

### Generated Dataset Format

When using msmarco integration, the generated dataset includes:
```json
{
    "topology_type": "linear",
    "num_agents": 6,
    "query": "What is the capital of France?",
    "adv_texts": ["In an alternate universe..."],
    "correct_answer": "Paris",
    "incorrect_answer": "London",
    "data_source": "msmarco",
    "expected_behavior": "agents_should_provide_correct_answer",
    "attack_scenario": "adversarial_text_injection",
    "communication_data": [...]
}
```

## Integration Details

### 1. Dataset Loading

The `load_msmarco_dataset()` method:
- Loads msmarco.json file
- Splits data by phase (train/test/val)
- Converts to NewMA-compatible format

### 2. Sample Selection

The `get_random_msmarco_samples()` method:
- Randomly selects samples from loaded data
- Ensures diversity in training data
- Handles cases where requested samples exceed available data

### 3. Communication Generation

When `use_msmarco=True`:
- Uses msmarco questions instead of predefined queries
- Incorporates adversarial texts as context
- Preserves correct/incorrect answers for evaluation
- Adds metadata for attack scenario analysis

## Example Output

```
Loading msmarco dataset from: ../MA/datasets/msmarco.json
Loaded 100 msmarco items for phase: train
Using msmarco data with 100 available samples
Debug: topology_types = 5
Debug: num_agents_list = [4, 6, 8, 10]
Debug: use_msmarco = True
Debug: msmarco_samples_per_config = 3
Generated 60 network configurations
Used msmarco dataset with 100 samples
Dataset saved to: ./msmarco_integrated_dataset/20231201_143022-advanced_network.json
```

## Benefits

1. **Real Adversarial Data**: Use actual adversarial texts instead of synthetic data
2. **Diverse Questions**: Access to wide variety of questions from msmarco
3. **Attack Scenarios**: Real-world attack scenarios for testing robustness
4. **Evaluation Metrics**: Correct/incorrect answers for performance evaluation
5. **Reproducibility**: Same data source as MA project for fair comparison

## File Structure

```
NewMA/
├── graph_generator.py          # Modified with msmarco support
├── dataset_processor.py        # MSMarco data processing
├── example_msmarco_usage.py    # Usage examples
└── MSMARCO_INTEGRATION_README.md  # This file
```

## Dependencies

- `dataset_processor.py`: Handles msmarco.json loading and processing
- `agent_prompts.py`: Provides attacker and system prompts
- `network_topologies.py`: Network topology definitions

## Troubleshooting

1. **File Not Found**: Ensure msmarco.json exists at specified path
2. **Import Errors**: Check that all NewMA modules are properly installed
3. **Memory Issues**: Reduce `msmarco_samples_per_config` for large datasets
4. **API Errors**: Verify LLM API configuration and credentials

## Comparison with MA

| Feature | MA | NewMA |
|---------|----|-------|
| Dataset Loading | `gen_memory_attack_data.py` | `dataset_processor.py` |
| Graph Generation | `gen_graph.py` | `graph_generator.py` |
| Data Integration | Manual combination | Automatic integration |
| Topology Support | Limited | Multiple topologies |
| Scalability | Fixed | Configurable |

This integration brings NewMA's advanced topology capabilities together with MA's proven adversarial data approach. 