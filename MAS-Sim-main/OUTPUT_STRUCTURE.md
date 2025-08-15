# NewMA Output Folder Structure

## Overview

The NewMA system now organizes output in a query-based folder structure, where each query gets its own directory containing all related data, images, and logs.

## Base Directory

All output is stored in: `/work/G-safeguard/NewMA/output`

## Folder Structure

```
/work/G-safeguard/NewMA/output/
├── network_0/                          # Network-level data
│   ├── hierarchy_graph.png             # Network hierarchy visualization
│   └── network_metadata.json           # Network configuration and metadata
├── network_1/
│   ├── hierarchy_graph.png
│   └── network_metadata.json
├── query_1_What_is_the_capital_of_France/
│   ├── query_metadata.json             # Query-specific metadata
│   ├── data/                           # Communication and accuracy data
│   │   ├── communication_data_tree_hierarchy_4agents.json
│   │   └── accuracy_report_tree_hierarchy_4agents.json
│   ├── images/                         # Network visualizations
│   │   └── hierarchy_graph_tree_hierarchy_4agents.png
│   └── logs/                           # Processing logs and errors
│       ├── processing_log_tree_hierarchy_4agents.json
│       └── error_log_tree_hierarchy_4agents.json (if errors occur)
└── query_2_Explain_the_concept_of_machine_learning/
    ├── query_metadata.json
    ├── data/
    ├── images/
    └── logs/
```

## File Descriptions

### Network-Level Files

- **`network_X/hierarchy_graph.png`**: Visual representation of the network topology with agent roles and hierarchy levels
- **`network_X/network_metadata.json`**: Contains network configuration, agent roles, hierarchy information, and adjacency matrix

### Query-Level Files

#### `query_metadata.json`
Contains metadata about the specific query:
```json
{
  "query": "What is the capital of France?",
  "query_index": 0,
  "topology_type": "tree_hierarchy",
  "num_agents": 4,
  "attacker_ratio": 0.2,
  "attacker_strategy": "persuasion",
  "sparsity": 0.25,
  "use_msmarco": false,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Data Files (`data/`)

- **`communication_data_<topology>_<agents>agents.json`**: Complete communication data including all agent responses, permissions, and accuracy statistics
- **`accuracy_report_<topology>_<agents>agents.json`**: Detailed accuracy analysis and manipulation rate statistics

#### Image Files (`images/`)

- **`hierarchy_graph_<topology>_<agents>agents.png`**: Copy of the network hierarchy visualization specific to this query

#### Log Files (`logs/`)

- **`processing_log_<topology>_<agents>agents.json`**: Processing information including timing and response counts
- **`error_log_<topology>_<agents>agents.json`**: Error logs if processing fails (only created if errors occur)

## Agent Permissions Information

Each agent response in the communication data now includes detailed permission information:

```json
{
  "agent_permissions": {
    "hierarchy_level": "root",
    "hierarchy_role": "coordinator", 
    "authority_level": 3,
    "subordinates": ["agent_1", "agent_2"],
    "supervisors": [],
    "peers": [],
    "responsibilities": ["Strategic planning", "Resource allocation"],
    "permissions": ["Can assign tasks to all levels", "Can override any decision"],
    "can_assign_tasks": true,
    "can_override_decisions": true,
    "can_access_all_info": true,
    "can_coordinate_others": true,
    "can_make_final_decisions": true
  }
}
```

## Usage

### Basic Usage

```python
from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

generator = AdvancedGraphGenerator()

# Generate dataset with organized output
dataset = await generator.generate_comprehensive_dataset(
    topology_types=[TopologyType.TREE_HIERARCHY],
    num_agents_list=[4, 6],
    num_networks_per_config=2,
    queries=["What is the capital of France?", "Explain machine learning"],
    output_base_dir="/work/G-safeguard/NewMA/output"
)
```

### Accessing Results

```python
import os
import json

# Read query metadata
with open("/work/G-safeguard/NewMA/output/query_1_What_is_the_capital_of_France/query_metadata.json", "r") as f:
    metadata = json.load(f)

# Read communication data
with open("/work/G-safeguard/NewMA/output/query_1_What_is_the_capital_of_France/data/communication_data_tree_hierarchy_4agents.json", "r") as f:
    communication_data = json.load(f)

# Read accuracy report
with open("/work/G-safeguard/NewMA/output/query_1_What_is_the_capital_of_France/data/accuracy_report_tree_hierarchy_4agents.json", "r") as f:
    accuracy_report = json.load(f)
```

## Benefits

1. **Organized Structure**: Each query has its own folder with all related files
2. **Easy Navigation**: Clear separation of data, images, and logs
3. **Comprehensive Metadata**: Detailed information about queries and processing
4. **Error Tracking**: Separate error logs for debugging
5. **Visualization**: Network hierarchy graphs for each configuration
6. **Permission Information**: Detailed agent permission data for analysis

## File Naming Convention

- Query folders: `query_<index>_<sanitized_query_name>`
- Data files: `<type>_<topology>_<agents>agents.json`
- Image files: `<type>_<topology>_<agents>agents.png`
- Log files: `<type>_<topology>_<agents>agents.json`

The sanitized query name removes special characters and limits length to 50 characters for compatibility. 