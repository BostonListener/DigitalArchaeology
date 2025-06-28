#!/usr/bin/env python3
"""
Archaeological Pipeline Orchestrator

A simple pipeline management system for the archaeological detection workflow.
Manages the execution of three main stages:
1. Stage 1: Deforestation analysis and archaeological candidate filtering
2. Stage 2: Sentinel-2 satellite data download and NDVI analysis
3. Stage 3: FABDEM elevation validation and final site determination

This script provides a command-line interface to run individual stages or the
complete pipeline with dependency checking and error handling.

Usage:
    python run_pipeline.py --full           # Run complete pipeline
    python run_pipeline.py --stage 1        # Run specific stage
    python run_pipeline.py --check          # Check dependencies only

Authors: Archaeological AI Team
License: MIT
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def load_config():
    """
    Load pipeline configuration from YAML file.
    
    Returns:
        dict: Configuration parameters for the pipeline
        
    Raises:
        SystemExit: If configuration file is not found
    """
    config_path = Path("config/parameters.yaml")
    if not config_path.exists():
        print("ERROR: Configuration file not found: config/parameters.yaml")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_dependencies():
    """
    Verify that required input files exist before running the pipeline.
    
    Currently checks for:
    - PRODES deforestation data (GPKG format)
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    config = load_config()
    
    # Check input GPKG file (PRODES deforestation data)
    gpkg_path = Path(config['paths']['input_gpkg'])
    if not gpkg_path.exists():
        print(f"ERROR: Input GPKG not found: {gpkg_path}")
        print("INFO: Download PRODES data and place in data/input/")
        return False
    
    print(f"SUCCESS: Found input GPKG: {gpkg_path}")
    return True


def run_stage(stage_num, substage=None):
    """
    Execute a specific pipeline stage.
    
    Args:
        stage_num (int): Stage number (1, 2, or 3)
        substage (str, optional): Substage identifier ('a' or 'b' for stage 2)
        
    Returns:
        bool: True if stage completed successfully, False otherwise
    """
    
    if stage_num == 1:
        # Stage 1: Deforestation Analysis
        print(">> Running Stage 1: Deforestation Analysis")
        result = subprocess.run([sys.executable, "stage1_deforestation.py"])
        
    elif stage_num == 2:
        if substage == 'a':
            # Stage 2A: Download Sentinel-2 data
            print(">> Running Stage 2A: Sentinel-2 Download")
            result = subprocess.run([sys.executable, "stage2a_download.py"])
        elif substage == 'b':
            # Stage 2B: Analyze NDVI patterns
            print(">> Running Stage 2B: NDVI Analysis")
            result = subprocess.run([sys.executable, "stage2b_analysis.py"])
        else:
            # Run complete Stage 2 (both substages)
            print(">> Running Stage 2: Sentinel-2 Download & Analysis")
            
            # First run download (2a)
            result_a = subprocess.run([sys.executable, "stage2a_download.py"])
            if result_a.returncode != 0:
                print("ERROR: Stage 2A failed")
                return False
                
            # Then run analysis (2b)
            result = subprocess.run([sys.executable, "stage2b_analysis.py"])
            
    elif stage_num == 3:
        # Stage 3: FABDEM elevation validation
        print(">> Running Stage 3: DEM Validation")
        result = subprocess.run([sys.executable, "stage3_validation.py"])
        
    else:
        print(f"ERROR: Invalid stage: {stage_num}")
        return False
    
    # Check result and provide feedback
    if result.returncode == 0:
        print(f"SUCCESS: Stage {stage_num}{substage or ''} completed successfully")
        return True
    else:
        print(f"ERROR: Stage {stage_num}{substage or ''} failed")
        return False


def main():
    """
    Main entry point for the pipeline orchestrator.
    
    Parses command line arguments and executes the requested operations.
    Handles both individual stage execution and full pipeline runs.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Archaeological Detection Pipeline")
    parser.add_argument("--stage", type=str, help="Run specific stage (1, 2, 2a, 2b, 3)")
    parser.add_argument("--full", action="store_true", help="Run complete pipeline")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    
    args = parser.parse_args()
    
    print("Archaeological Detection Pipeline")
    print("=" * 50)
    
    # Always check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # If only checking dependencies, exit here
    if args.check:
        print("SUCCESS: All dependencies check passed")
        return
    
    try:
        if args.full:
            # Run complete pipeline (all stages in sequence)
            print(">> Running complete pipeline...")
            stages = [1, 2, 3]
            for stage in stages:
                if not run_stage(stage):
                    print(f"ERROR: Pipeline failed at stage {stage}")
                    sys.exit(1)
            print("SUCCESS: Complete pipeline finished successfully!")
            
        elif args.stage:
            # Run specific stage based on user input
            stage_str = args.stage.lower()
            if stage_str == '1':
                run_stage(1)
            elif stage_str == '2':
                run_stage(2)
            elif stage_str == '2a':
                run_stage(2, 'a')
            elif stage_str == '2b':
                run_stage(2, 'b')
            elif stage_str == '3':
                run_stage(3)
            else:
                print(f"ERROR: Invalid stage: {args.stage}")
                sys.exit(1)
        else:
            # No arguments provided - show usage examples
            print("Usage examples:")
            print("  python run_pipeline.py --full           # Complete pipeline")
            print("  python run_pipeline.py --stage 1        # Deforestation analysis")
            print("  python run_pipeline.py --stage 2a       # Download Sentinel-2")
            print("  python run_pipeline.py --stage 2b       # Analyze NDVI patterns")
            print("  python run_pipeline.py --stage 3        # DEM validation")
            print("  python run_pipeline.py --check          # Check dependencies")
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"ERROR: Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()