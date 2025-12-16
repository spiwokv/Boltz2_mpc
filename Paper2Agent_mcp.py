"""
Model Context Protocol (MCP) for Paper2Agent

Paper2Agent provides biomolecular structure prediction and affinity analysis tools extracted from research tutorials. The system specializes in protein structure prediction, protein-ligand interaction modeling, and binding affinity calculations using the Boltz prediction framework. All tools are designed for computational biology workflows and molecular modeling applications.

This MCP Server contains tools extracted from the following tutorial files:
1. prediction
    - boltz_predict_protein_structure: Predict protein structure from sequence
    - boltz_predict_protein_ligand_affinity: Predict protein-ligand complex structure and binding affinity
    - boltz_predict_multimer_complex: Predict multi-chain protein complex structure
    - boltz_predict_with_pocket_constraints: Predict structure with binding site constraints
    - boltz_predict_cyclic_peptide: Predict cyclic peptide structure
    - boltz_interpret_confidence_scores: Interpret Boltz confidence metrics
    - boltz_interpret_affinity_predictions: Interpret and convert affinity prediction values
    - boltz_create_input_yaml: Create custom YAML input file for Boltz predictions
"""

from fastmcp import FastMCP

# Import statements (alphabetical order)
from tools.prediction import prediction_mcp

# Server definition and mounting
mcp = FastMCP(name="Paper2Agent")
mcp.mount(prediction_mcp)

if __name__ == "__main__":
    mcp.run()