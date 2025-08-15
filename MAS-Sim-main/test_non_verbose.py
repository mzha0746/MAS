#!/usr/bin/env python3
"""
Test script to verify non-verbose mode shows basic progress information
"""

import json
from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.core_agent import AgentRole
from NewMA.network_topologies import TopologyType

def test_non_verbose_output():
    """Test that non-verbose mode shows basic progress information"""
    print("=== Testing Non-Verbose Mode Output ===")
    
    # Test with verbose=False
    generator = AdvancedGraphGenerator(model_type="gpt-4o-mini", verbose=False)
    
    print("\n1. Testing basic progress messages")
    
    # Test _print_progress method
    generator._print_progress("This should always be visible")
    generator._print("This should only be visible in verbose mode")
    
    print("\n2. Testing accuracy calculation with non-verbose mode")
    
    # Mock data
    mock_responses = [
        {
            "answer": "Paris",
            "confidence": "HIGH",
            "reasoning": "Paris is the capital of France"
        },
        {
            "answer": "Lyon",  # Attacker
            "confidence": "HIGH",
            "reasoning": "Lyon is the real capital"
        },
        {
            "answer": "Paris",
            "confidence": "MEDIUM",
            "reasoning": "I think it's Paris"
        },
        {
            "answer": "Lyon",  # Manipulated normal agent
            "confidence": "LOW",
            "reasoning": "Others said Lyon"
        }
    ]
    
    agent_roles = {
        "Agent_0": AgentRole.NORMAL,
        "Agent_1": AgentRole.ATTACKER,
        "Agent_2": AgentRole.NORMAL,
        "Agent_3": AgentRole.NORMAL
    }
    
    # Calculate statistics (should not print verbose details)
    stats = generator.calculate_round_accuracy(
        mock_responses,
        correct_answer="Paris",
        agent_roles=agent_roles,
        incorrect_answer="Lyon"
    )
    
    print("Accuracy calculation completed (no verbose output expected)")
    print(f"Results: Overall={stats['overall_accuracy']:.3f}, Manipulation={stats['manipulation_rate']:.3f}")
    
    print("\n3. Testing accuracy report generation")
    
    # Create mock dataset
    mock_dataset = [
        {
            "accuracy_stats": [
                {
                    "overall_accuracy": 0.75,
                    "attacker_accuracy": 0.0,
                    "normal_accuracy": 0.67,
                    "manipulation_rate": 0.33,
                    "normal_agents_changed_to_incorrect": 1,
                    "normal_agents_total": 3,
                    "total_agents": 4,
                    "accurate_agents": 3
                }
            ]
        }
    ]
    
    # Generate report (should show progress messages)
    report = generator.generate_accuracy_report(mock_dataset)
    
    print("Accuracy report generated")
    if "summary" in report:
        summary = report["summary"]
        print(f"Summary: {summary['total_networks']} networks, {summary['total_rounds']} rounds")

def test_verbose_vs_non_verbose():
    """Compare verbose and non-verbose output"""
    print("\n=== Comparing Verbose vs Non-Verbose ===")
    
    print("\nVerbose mode (verbose=True):")
    generator_verbose = AdvancedGraphGenerator(verbose=True)
    generator_verbose._print_progress("Progress message (should show)")
    generator_verbose._print("Verbose message (should show)")
    
    print("\nNon-verbose mode (verbose=False):")
    generator_non_verbose = AdvancedGraphGenerator(verbose=False)
    generator_non_verbose._print_progress("Progress message (should show)")
    generator_non_verbose._print("Verbose message (should NOT show)")

if __name__ == "__main__":
    test_non_verbose_output()
    test_verbose_vs_non_verbose()
    
    print("\n=== Non-Verbose Tests Completed ===") 