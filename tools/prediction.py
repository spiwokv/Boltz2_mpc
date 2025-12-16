"""
Boltz biomolecular structure and affinity prediction tools.

This MCP Server provides 8 tools:
1. boltz_predict_protein_structure: Predict protein structure from sequence
2. boltz_predict_protein_ligand_affinity: Predict protein-ligand complex structure and binding affinity
3. boltz_predict_multimer_complex: Predict multi-chain protein complex structure
4. boltz_predict_with_pocket_constraints: Predict structure with binding site constraints
5. boltz_predict_cyclic_peptide: Predict cyclic peptide structure
6. boltz_interpret_confidence_scores: Interpret Boltz confidence metrics
7. boltz_interpret_affinity_predictions: Interpret and convert affinity prediction values
8. boltz_create_input_yaml: Create custom YAML input file for Boltz predictions

All tools extracted from https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
# Standard imports
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from fastmcp import FastMCP

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("PREDICTION_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("PREDICTION_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
prediction_mcp = FastMCP(name="prediction")


@prediction_mcp.tool
def boltz_predict_protein_structure(
    yaml_input_path: Annotated[
        str | None,
        "Path to YAML input file specifying protein sequence and optional MSA. File should follow Boltz YAML format with 'version' and 'sequences' fields.",
    ] = None,
    use_msa_server: Annotated[
        bool, "Auto-generate MSA using MMSeqs2 server instead of providing custom MSA"
    ] = True,
    use_potentials: Annotated[
        bool, "Apply inference-time potentials for better physical quality"
    ] = False,
    recycling_steps: Annotated[int, "Number of recycling steps for prediction"] = 3,
    diffusion_samples: Annotated[int, "Number of diffusion samples to generate"] = 1,
    output_format: Annotated[
        Literal["pdb", "mmcif"], "Output structure format"
    ] = "mmcif",
    out_prefix: Annotated[str | None, "Output directory prefix"] = None,
) -> dict:
    """
    Predict protein structure from sequence using Boltz structure prediction.
    Input is YAML file with protein sequence and output is predicted structure with confidence scores.
    """
    # Input validation
    if yaml_input_path is None:
        raise ValueError("Path to YAML input file must be provided")

    yaml_file = Path(yaml_input_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Input YAML file not found: {yaml_input_path}")

    # Setup output directory
    if out_prefix is None:
        out_dir = OUTPUT_DIR / f"protein_structure_{timestamp}"
    else:
        out_dir = OUTPUT_DIR / f"{out_prefix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "boltz",
        "predict",
        str(yaml_file),
        "--out_dir",
        str(out_dir),
        "--recycling_steps",
        str(recycling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--output_format",
        output_format,
        "--no_kernels",  # Fix for cuequivariance_torch missing module
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    if use_potentials:
        cmd.append("--use_potentials")

    # Run prediction
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Boltz prediction failed: {result.stderr}")

    # Collect output artifacts
    artifacts = []
    # Boltz creates boltz_results_<name>/ subdirectory, so search recursively from out_dir
    # Boltz creates .cif files for mmcif format and .pdb files for pdb format
    ext = "cif" if output_format == "mmcif" else "pdb"
    for struct_file in out_dir.rglob(f"*.{ext}"):
        artifacts.append(
            {"description": f"Predicted structure", "path": str(struct_file.resolve())}
        )

    # Find confidence files
    for conf_file in out_dir.rglob("confidence_*.json"):
        artifacts.append(
            {"description": "Confidence scores", "path": str(conf_file.resolve())}
        )

    return {
        "message": f"Protein structure prediction completed with {diffusion_samples} sample(s)",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": artifacts,
    }


@prediction_mcp.tool
def boltz_predict_protein_ligand_affinity(
    yaml_input_path: Annotated[
        str | None,
        "Path to YAML input file specifying protein-ligand complex with affinity property. Must include protein sequence and ligand (SMILES or CCD code) with affinity binder specification.",
    ] = None,
    use_msa_server: Annotated[bool, "Auto-generate MSA using MMSeqs2 server"] = True,
    use_potentials: Annotated[
        bool, "Apply inference-time potentials for better physical quality"
    ] = False,
    recycling_steps: Annotated[int, "Number of recycling steps for prediction"] = 3,
    diffusion_samples: Annotated[
        int, "Number of diffusion samples for structure prediction"
    ] = 1,
    sampling_steps_affinity: Annotated[
        int, "Number of sampling steps for affinity prediction"
    ] = 200,
    diffusion_samples_affinity: Annotated[
        int, "Number of diffusion samples for affinity prediction"
    ] = 5,
    output_format: Annotated[
        Literal["pdb", "mmcif"], "Output structure format"
    ] = "mmcif",
    out_prefix: Annotated[str | None, "Output directory prefix"] = None,
) -> dict:
    """
    Predict protein-ligand complex structure and binding affinity using Boltz.
    Input is YAML file with protein-ligand complex specification and output is predicted structure with affinity scores.
    """
    # Input validation
    if yaml_input_path is None:
        raise ValueError("Path to YAML input file must be provided")

    yaml_file = Path(yaml_input_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Input YAML file not found: {yaml_input_path}")

    # Setup output directory
    if out_prefix is None:
        out_dir = OUTPUT_DIR / f"protein_ligand_affinity_{timestamp}"
    else:
        out_dir = OUTPUT_DIR / f"{out_prefix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "boltz",
        "predict",
        str(yaml_file),
        "--out_dir",
        str(out_dir),
        "--recycling_steps",
        str(recycling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--sampling_steps_affinity",
        str(sampling_steps_affinity),
        "--diffusion_samples_affinity",
        str(diffusion_samples_affinity),
        "--output_format",
        output_format,
        "--no_kernels",  # Fix for cuequivariance_torch missing module
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    if use_potentials:
        cmd.append("--use_potentials")

    # Run prediction
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Boltz prediction failed: {result.stderr}")

    # Collect output artifacts
    artifacts = []
    # Boltz creates boltz_results_<name>/ subdirectory, so search recursively from out_dir
    # Boltz creates .cif files for mmcif format and .pdb files for pdb format
    ext = "cif" if output_format == "mmcif" else "pdb"
    for struct_file in out_dir.rglob(f"*.{ext}"):
        artifacts.append(
            {
                "description": "Predicted complex structure",
                "path": str(struct_file.resolve()),
            }
        )

    # Find confidence files
    for conf_file in out_dir.rglob("confidence_*.json"):
        artifacts.append(
            {"description": "Confidence scores", "path": str(conf_file.resolve())}
        )

    # Find affinity files
    for aff_file in out_dir.rglob("affinity_*.json"):
        artifacts.append(
            {"description": "Affinity predictions", "path": str(aff_file.resolve())}
        )

    return {
        "message": f"Protein-ligand affinity prediction completed with {diffusion_samples_affinity} affinity sample(s)",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": artifacts,
    }


@prediction_mcp.tool
def boltz_predict_multimer_complex(
    yaml_input_path: Annotated[
        str | None,
        "Path to YAML input file specifying multi-chain protein complex. Must include multiple protein sequences with different chain IDs.",
    ] = None,
    use_msa_server: Annotated[bool, "Auto-generate MSA using MMSeqs2 server"] = True,
    use_potentials: Annotated[
        bool, "Apply inference-time potentials for better physical quality"
    ] = False,
    recycling_steps: Annotated[int, "Number of recycling steps for prediction"] = 3,
    diffusion_samples: Annotated[int, "Number of diffusion samples to generate"] = 1,
    output_format: Annotated[
        Literal["pdb", "mmcif"], "Output structure format"
    ] = "mmcif",
    out_prefix: Annotated[str | None, "Output directory prefix"] = None,
) -> dict:
    """
    Predict multi-chain protein complex structure using Boltz.
    Input is YAML file with multiple protein chains and output is predicted complex structure with confidence scores.
    """
    # Input validation
    if yaml_input_path is None:
        raise ValueError("Path to YAML input file must be provided")

    yaml_file = Path(yaml_input_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Input YAML file not found: {yaml_input_path}")

    # Setup output directory
    if out_prefix is None:
        out_dir = OUTPUT_DIR / f"multimer_complex_{timestamp}"
    else:
        out_dir = OUTPUT_DIR / f"{out_prefix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "boltz",
        "predict",
        str(yaml_file),
        "--out_dir",
        str(out_dir),
        "--recycling_steps",
        str(recycling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--output_format",
        output_format,
        "--no_kernels",  # Fix for cuequivariance_torch missing module
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    if use_potentials:
        cmd.append("--use_potentials")

    # Run prediction
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Boltz prediction failed: {result.stderr}")

    # Collect output artifacts
    artifacts = []
    # Boltz creates boltz_results_<name>/ subdirectory, so search recursively from out_dir
    # Boltz creates .cif files for mmcif format and .pdb files for pdb format
    ext = "cif" if output_format == "mmcif" else "pdb"
    for struct_file in out_dir.rglob(f"*.{ext}"):
        artifacts.append(
            {
                "description": "Predicted multimer structure",
                "path": str(struct_file.resolve()),
            }
        )

    # Find confidence files
    for conf_file in out_dir.rglob("confidence_*.json"):
        artifacts.append(
            {"description": "Confidence scores", "path": str(conf_file.resolve())}
        )

    return {
        "message": f"Multi-chain complex prediction completed with {diffusion_samples} sample(s)",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": artifacts,
    }


@prediction_mcp.tool
def boltz_predict_with_pocket_constraints(
    yaml_input_path: Annotated[
        str | None,
        "Path to YAML input file with pocket constraints. Must include protein, ligand, and constraints section specifying binding site residues.",
    ] = None,
    use_msa_server: Annotated[bool, "Auto-generate MSA using MMSeqs2 server"] = True,
    use_potentials: Annotated[
        bool, "Apply inference-time potentials for better physical quality"
    ] = False,
    recycling_steps: Annotated[int, "Number of recycling steps for prediction"] = 3,
    diffusion_samples: Annotated[int, "Number of diffusion samples to generate"] = 1,
    output_format: Annotated[
        Literal["pdb", "mmcif"], "Output structure format"
    ] = "mmcif",
    out_prefix: Annotated[str | None, "Output directory prefix"] = None,
) -> dict:
    """
    Predict protein-ligand structure with binding site constraints using Boltz.
    Input is YAML file with pocket constraints and output is predicted structure guided by specified binding site residues.
    """
    # Input validation
    if yaml_input_path is None:
        raise ValueError("Path to YAML input file must be provided")

    yaml_file = Path(yaml_input_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Input YAML file not found: {yaml_input_path}")

    # Setup output directory
    if out_prefix is None:
        out_dir = OUTPUT_DIR / f"pocket_constrained_{timestamp}"
    else:
        out_dir = OUTPUT_DIR / f"{out_prefix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "boltz",
        "predict",
        str(yaml_file),
        "--out_dir",
        str(out_dir),
        "--recycling_steps",
        str(recycling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--output_format",
        output_format,
        "--no_kernels",  # Fix for cuequivariance_torch missing module
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    if use_potentials:
        cmd.append("--use_potentials")

    # Run prediction
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Boltz prediction failed: {result.stderr}")

    # Collect output artifacts
    artifacts = []
    # Boltz creates boltz_results_<name>/ subdirectory, so search recursively from out_dir
    # Boltz creates .cif files for mmcif format and .pdb files for pdb format
    ext = "cif" if output_format == "mmcif" else "pdb"
    for struct_file in out_dir.rglob(f"*.{ext}"):
        artifacts.append(
            {
                "description": "Predicted structure with constraints",
                "path": str(struct_file.resolve()),
            }
        )

    # Find confidence files
    for conf_file in out_dir.rglob("confidence_*.json"):
        artifacts.append(
            {"description": "Confidence scores", "path": str(conf_file.resolve())}
        )

    return {
        "message": f"Pocket-constrained prediction completed with {diffusion_samples} sample(s)",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": artifacts,
    }


@prediction_mcp.tool
def boltz_predict_cyclic_peptide(
    yaml_input_path: Annotated[
        str | None,
        "Path to YAML input file with cyclic peptide specification. Must include protein sequence with cyclic:true flag.",
    ] = None,
    use_msa_server: Annotated[bool, "Auto-generate MSA using MMSeqs2 server"] = True,
    use_potentials: Annotated[
        bool, "Apply inference-time potentials for better physical quality"
    ] = False,
    recycling_steps: Annotated[int, "Number of recycling steps for prediction"] = 3,
    diffusion_samples: Annotated[int, "Number of diffusion samples to generate"] = 1,
    output_format: Annotated[
        Literal["pdb", "mmcif"], "Output structure format"
    ] = "mmcif",
    out_prefix: Annotated[str | None, "Output directory prefix"] = None,
) -> dict:
    """
    Predict cyclic peptide structure using Boltz.
    Input is YAML file with cyclic peptide sequence and output is predicted cyclic structure with confidence scores.
    """
    # Input validation
    if yaml_input_path is None:
        raise ValueError("Path to YAML input file must be provided")

    yaml_file = Path(yaml_input_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Input YAML file not found: {yaml_input_path}")

    # Setup output directory
    if out_prefix is None:
        out_dir = OUTPUT_DIR / f"cyclic_peptide_{timestamp}"
    else:
        out_dir = OUTPUT_DIR / f"{out_prefix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "boltz",
        "predict",
        str(yaml_file),
        "--out_dir",
        str(out_dir),
        "--recycling_steps",
        str(recycling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--output_format",
        output_format,
        "--no_kernels",  # Fix for cuequivariance_torch missing module
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    if use_potentials:
        cmd.append("--use_potentials")

    # Run prediction
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Boltz prediction failed: {result.stderr}")

    # Collect output artifacts
    artifacts = []
    # Boltz creates boltz_results_<name>/ subdirectory, so search recursively from out_dir
    # Boltz creates .cif files for mmcif format and .pdb files for pdb format
    ext = "cif" if output_format == "mmcif" else "pdb"
    for struct_file in out_dir.rglob(f"*.{ext}"):
        artifacts.append(
            {
                "description": "Predicted cyclic structure",
                "path": str(struct_file.resolve()),
            }
        )

    # Find confidence files
    for conf_file in out_dir.rglob("confidence_*.json"):
        artifacts.append(
            {"description": "Confidence scores", "path": str(conf_file.resolve())}
        )

    return {
        "message": f"Cyclic peptide prediction completed with {diffusion_samples} sample(s)",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": artifacts,
    }


@prediction_mcp.tool
def boltz_interpret_confidence_scores(
    confidence_json_path: Annotated[
        str | None, "Path to confidence JSON file from Boltz prediction output"
    ] = None,
    out_prefix: Annotated[
        str | None, "Output file prefix for interpretation report"
    ] = None,
) -> dict:
    """
    Interpret Boltz confidence scores and provide detailed metrics explanation.
    Input is confidence JSON file from predictions and output is interpreted confidence metrics report.
    """
    # Input validation
    if confidence_json_path is None:
        raise ValueError("Path to confidence JSON file must be provided")

    conf_file = Path(confidence_json_path)
    if not conf_file.exists():
        raise FileNotFoundError(
            f"Confidence JSON file not found: {confidence_json_path}"
        )

    # Load confidence scores
    with open(conf_file, "r") as f:
        confidence_data = json.load(f)

    # Interpret scores
    interpretation = {
        "overall_quality": {},
        "per_chain_quality": {},
        "interface_quality": {},
        "recommendations": [],
    }

    # Overall quality assessment
    conf_score = confidence_data.get("confidence_score", 0)
    ptm = confidence_data.get("ptm", 0)
    iptm = confidence_data.get("iptm", 0)
    complex_plddt = confidence_data.get("complex_plddt", 0)

    interpretation["overall_quality"] = {
        "confidence_score": conf_score,
        "interpretation": (
            "Excellent"
            if conf_score > 0.8
            else (
                "Good"
                if conf_score > 0.7
                else "Moderate" if conf_score > 0.5 else "Low"
            )
        ),
        "ptm": ptm,
        "ptm_interpretation": (
            "High confidence"
            if ptm > 0.8
            else "Good" if ptm > 0.7 else "Moderate" if ptm > 0.5 else "Low confidence"
        ),
        "complex_plddt": complex_plddt,
        "plddt_interpretation": (
            "High accuracy"
            if complex_plddt > 0.8
            else (
                "Good"
                if complex_plddt > 0.7
                else "Moderate" if complex_plddt > 0.5 else "Low accuracy"
            )
        ),
    }

    # Interface quality
    if "ligand_iptm" in confidence_data or "protein_iptm" in confidence_data:
        interpretation["interface_quality"] = {
            "iptm": iptm,
            "interpretation": (
                "Strong interfaces"
                if iptm > 0.8
                else "Good" if iptm > 0.7 else "Weak interfaces"
            ),
            "ligand_iptm": confidence_data.get("ligand_iptm", 0),
            "protein_iptm": confidence_data.get("protein_iptm", 0),
            "complex_iplddt": confidence_data.get("complex_iplddt", 0),
        }

    # Per-chain quality
    if "chains_ptm" in confidence_data:
        interpretation["per_chain_quality"] = confidence_data["chains_ptm"]

    # Recommendations
    if conf_score < 0.7:
        interpretation["recommendations"].append(
            "Consider generating additional samples with --diffusion_samples"
        )
        interpretation["recommendations"].append(
            "Try using --use_potentials for improved physical quality"
        )

    if conf_score < 0.5:
        interpretation["recommendations"].append(
            "Check input sequence quality and MSA coverage"
        )
        interpretation["recommendations"].append(
            "Verify that all required components are correctly specified"
        )

    if iptm > 0 and iptm < 0.6:
        interpretation["recommendations"].append(
            "Low interface confidence - verify binding site specification"
        )

    # Save interpretation report
    if out_prefix is None:
        output_file = OUTPUT_DIR / f"confidence_interpretation_{timestamp}.json"
    else:
        output_file = (
            OUTPUT_DIR / f"{out_prefix}_confidence_interpretation_{timestamp}.json"
        )

    with open(output_file, "w") as f:
        json.dump(interpretation, f, indent=2)

    return {
        "message": f"Confidence scores interpreted - Overall quality: {interpretation['overall_quality']['interpretation']}",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": [
            {
                "description": "Confidence interpretation report",
                "path": str(output_file.resolve()),
            }
        ],
    }


@prediction_mcp.tool
def boltz_interpret_affinity_predictions(
    affinity_json_path: Annotated[
        str | None, "Path to affinity JSON file from Boltz prediction output"
    ] = None,
    out_prefix: Annotated[
        str | None, "Output file prefix for interpretation report"
    ] = None,
) -> dict:
    """
    Interpret Boltz affinity predictions and convert to standard binding metrics.
    Input is affinity JSON file from predictions and output is interpreted affinity metrics with IC50 conversions.
    """
    # Input validation
    if affinity_json_path is None:
        raise ValueError("Path to affinity JSON file must be provided")

    aff_file = Path(affinity_json_path)
    if not aff_file.exists():
        raise FileNotFoundError(f"Affinity JSON file not found: {affinity_json_path}")

    # Load affinity scores
    with open(aff_file, "r") as f:
        affinity_data = json.load(f)

    # Interpret affinity values
    interpretation = {
        "primary_metrics": {},
        "individual_predictions": [],
        "recommendations": [],
    }

    # Primary affinity metrics
    log_ic50 = affinity_data.get("affinity_pred_value", None)
    prob_binary = affinity_data.get("affinity_probability_binary", None)

    if log_ic50 is not None:
        ic50_uM = 10**log_ic50
        ic50_M = ic50_uM * 1e-6
        pic50_kcal_mol = (6 - log_ic50) * 1.364

        # Binding strength interpretation
        if log_ic50 < -2:
            strength = "Strong binder"
            use_case = "Lead optimization"
        elif log_ic50 < 1:
            strength = "Moderate binder"
            use_case = "Hit-to-lead"
        else:
            strength = "Weak binder / decoy"
            use_case = "Not recommended for development"

        interpretation["primary_metrics"] = {
            "log_ic50": log_ic50,
            "ic50_uM": ic50_uM,
            "ic50_M": ic50_M,
            "pic50_kcal_mol": pic50_kcal_mol,
            "binding_strength": strength,
            "recommended_use": use_case,
            "probability_binary": prob_binary,
            "binary_interpretation": (
                "Likely binder" if prob_binary and prob_binary > 0.5 else "Likely decoy"
            ),
        }

    # Individual model predictions
    for i in range(1, 10):  # Check up to 10 models
        pred_key = f"affinity_pred_value{i}"
        prob_key = f"affinity_probability_binary{i}"

        if pred_key in affinity_data:
            log_ic50_i = affinity_data[pred_key]
            ic50_uM_i = 10**log_ic50_i

            interpretation["individual_predictions"].append(
                {
                    "model_number": i,
                    "log_ic50": log_ic50_i,
                    "ic50_uM": ic50_uM_i,
                    "probability_binary": affinity_data.get(prob_key, None),
                }
            )

    # Recommendations
    if prob_binary and prob_binary < 0.5:
        interpretation["recommendations"].append(
            "Low probability of binding - consider as potential decoy"
        )
        interpretation["recommendations"].append(
            "Do not use affinity_pred_value for decoys - only for active molecules"
        )

    if prob_binary and prob_binary > 0.8:
        interpretation["recommendations"].append(
            "High confidence binder - suitable for hit discovery"
        )

    if log_ic50 is not None and log_ic50 < 0 and prob_binary and prob_binary > 0.7:
        interpretation["recommendations"].append(
            "Promising lead candidate - compare affinity_pred_value with other actives"
        )

    # Save interpretation report
    if out_prefix is None:
        output_file = OUTPUT_DIR / f"affinity_interpretation_{timestamp}.json"
    else:
        output_file = (
            OUTPUT_DIR / f"{out_prefix}_affinity_interpretation_{timestamp}.json"
        )

    with open(output_file, "w") as f:
        json.dump(interpretation, f, indent=2)

    return {
        "message": f"Affinity predictions interpreted - {interpretation['primary_metrics'].get('binding_strength', 'Unknown')}",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": [
            {
                "description": "Affinity interpretation report",
                "path": str(output_file.resolve()),
            }
        ],
    }


@prediction_mcp.tool
def boltz_create_input_yaml(
    sequences: Annotated[
        str,
        'JSON string defining molecular sequences. Format: [{"type": "protein", "id": "A", "sequence": "...", "msa": "path" (optional), "cyclic": true/false (optional)}, {"type": "ligand", "id": "B", "smiles": "..." or "ccd": "..."}, ...]',
    ] = "[]",
    properties: Annotated[
        str,
        'JSON string defining prediction properties. Format: [{"affinity": {"binder": "chain_id"}}] for affinity predictions. Empty list for structure-only predictions.',
    ] = "[]",
    constraints: Annotated[
        str,
        'JSON string defining prediction constraints. Format: [{"pocket": {"binder": "chain_id", "contacts": [["chain_id", residue_num], ...], "max_distance": 6, "force": false}}]. Empty list for no constraints.',
    ] = "[]",
    out_prefix: Annotated[str | None, "Output filename prefix"] = None,
) -> dict:
    """
    Create custom YAML input file for Boltz predictions with user-specified components.
    Input is molecular sequences specification and output is formatted YAML file ready for Boltz prediction.
    """
    # Parse JSON inputs
    try:
        sequences_list = json.loads(sequences)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for sequences: {e}")

    try:
        properties_list = json.loads(properties)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for properties: {e}")

    try:
        constraints_list = json.loads(constraints)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for constraints: {e}")

    # Validate sequences
    if not sequences_list:
        raise ValueError("At least one sequence must be provided")

    # Build YAML structure
    yaml_data = {"version": 1, "sequences": []}

    # Process sequences
    for seq_spec in sequences_list:
        seq_type = seq_spec.get("type", "").lower()
        seq_id = seq_spec.get("id")

        if not seq_id:
            raise ValueError("Each sequence must have an 'id' field")

        if seq_type == "protein":
            protein_entry = {
                "protein": {"id": seq_id, "sequence": seq_spec.get("sequence", "")}
            }

            if "msa" in seq_spec:
                protein_entry["protein"]["msa"] = seq_spec["msa"]

            if seq_spec.get("cyclic", False):
                protein_entry["protein"]["cyclic"] = True

            yaml_data["sequences"].append(protein_entry)

        elif seq_type == "ligand":
            ligand_entry = {"ligand": {"id": seq_id}}

            if "smiles" in seq_spec:
                ligand_entry["ligand"]["smiles"] = seq_spec["smiles"]
            elif "ccd" in seq_spec:
                ligand_entry["ligand"]["ccd"] = seq_spec["ccd"]
            else:
                raise ValueError("Ligand must specify either 'smiles' or 'ccd'")

            yaml_data["sequences"].append(ligand_entry)

        else:
            raise ValueError(f"Unsupported sequence type: {seq_type}")

    # Add properties if specified
    if properties_list:
        yaml_data["properties"] = properties_list

    # Add constraints if specified
    if constraints_list:
        yaml_data["constraints"] = constraints_list

    # Save YAML file
    if out_prefix is None:
        output_file = OUTPUT_DIR / f"custom_input_{timestamp}.yaml"
    else:
        output_file = OUTPUT_DIR / f"{out_prefix}_{timestamp}.yaml"

    with open(output_file, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # Also save a JSON representation for reference
    json_file = output_file.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(yaml_data, f, indent=2)

    return {
        "message": f"Custom YAML input created with {len(sequences_list)} sequence(s)",
        "reference": "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        "artifacts": [
            {
                "description": "Boltz YAML input file",
                "path": str(output_file.resolve()),
            },
            {"description": "JSON representation", "path": str(json_file.resolve())},
        ],
    }
