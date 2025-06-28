#!/usr/bin/env python3
"""
Archaeological Pipeline Setup Script

This script initializes the directory structure and validates the environment
for the AI-powered archaeological detection pipeline. It ensures all required
directories exist and configuration files are properly formatted.

The pipeline uses remote sensing data and AI analysis to identify potential
archaeological sites in the Amazon rainforest through deforestation pattern
analysis, vegetation anomaly detection, and elevation signature validation.

Usage:
    python setup_pipeline.py

Requirements:
    - Python 3.8+
    - PyYAML package
    - Valid configuration file (config/parameters.yaml)

Authors: Archaeological AI Team
License: MIT
"""

from pathlib import Path
import yaml

def create_directory_structure():
    """
    Create the complete directory structure required by the pipeline.
    
    The pipeline uses a hierarchical data organization:
    - config/: Configuration files and parameters
    - data/input/: Raw input data (PRODES, DEM files, etc.)
    - data/stage1/: Deforestation candidate analysis results
    - data/stage2/: Sentinel-2 NDVI pattern detection results  
    - data/stage3/: FABDEM elevation validation results
    
    Returns:
        bool: True if all directories were created successfully
    """
    
    print("[SETUP]  Creating directory structure...")
    
    # Define all required directories for the pipeline
    directories = [
        # Configuration directory for pipeline parameters
        "config",
        
        # Input data directories
        "data/input",           # Root input directory
        "data/input/DEM",       # Digital elevation model files (FABDEM)
        
        # Stage-specific output directories
        "data/stage1",          # Deforestation candidate analysis
        "data/stage2/downloads", # Sentinel-2 satellite data downloads
        "data/stage2/analysis",  # NDVI pattern detection results
        "data/stage3"           # Final archaeological site validation
    ]
    
    # Create each directory with parent directories as needed
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   [SUCCESS] Created: {directory}/")
    
    return True

def validate_config():
    """
    Validate the pipeline configuration file for completeness and structure.
    
    Checks that all required configuration sections exist and are accessible.
    The configuration file controls all aspects of the pipeline including:
    - Study area geographic bounds
    - Deforestation analysis parameters
    - Satellite data download settings
    - NDVI analysis thresholds
    - Elevation validation criteria
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    
    print("Validating configuration...")
    
    config_path = Path("config/parameters.yaml")
    if not config_path.exists():
        print(f"   [ERROR] Configuration file not found: {config_path}")
        return False
    
    try:
        # Load and parse the YAML configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Define all required top-level configuration sections
        required_sections = [
            'study_area',        # Geographic bounds for analysis
            'deforestation',     # PRODES deforestation analysis parameters
            'sentinel_download', # Sentinel-2 satellite data acquisition
            'sentinel_analysis', # NDVI pattern detection configuration
            'dem_validation',    # FABDEM elevation validation settings
            'paths'             # File paths and directory structure
        ]
        
        # Verify each required section exists in the configuration
        for section in required_sections:
            if section not in config:
                print(f"   [ERROR] Missing configuration section: {section}")
                return False
            print(f"   [SUCCESS] {section}")
        
        print(f"   [SUCCESS] Configuration file is valid")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Configuration file error: {e}")
        return False

def check_input_requirements():
    """
    Display information about required input data and external dependencies.
    
    The pipeline requires several external data sources that must be obtained
    separately due to size and licensing constraints. This function provides
    clear guidance on what data is needed and where to obtain it.
    
    Required Data Sources:
    1. PRODES Deforestation Data - TerraBrasilis/INPE
    2. FABDEM Elevation Data - NASA Earthdata
    3. Copernicus Satellite Credentials - ESA Copernicus Data Space
    
    Returns:
        bool: Always returns True (informational function)
    """
    
    print("[INPUT] Input data requirements:")
    print("   [DATA] Required before running:")
    
    # PRODES deforestation data requirement
    print("      1. PRODES deforestation data (.gpkg file)")
    print("         - Download from: http://terrabrasilis.dpi.inpe.br")
    print("         - Place in: data/input/prodes_amazonia_legal.gpkg")
    print("         - Purpose: Identify cleared areas for archaeological analysis")
    
    # Digital elevation model requirement  
    print("      2. FABDEM elevation data (.tif file)")
    print("         - Download from: https://search.earthdata.nasa.gov")
    print("         - Place in: data/input/DEM/")
    print("         - Purpose: Validate archaeological features through elevation signatures")
    
    # Satellite data access credentials
    print("      3. Copernicus Data Space credentials")
    print("         - Sign up at: https://dataspace.copernicus.eu/")
    print("         - Add to .env file")
    print("         - Purpose: Download Sentinel-2 satellite imagery for NDVI analysis")
    
    return True

def main():
    """
    Execute the complete pipeline setup process.
    
    This function orchestrates the entire setup workflow:
    1. Creates the required directory structure
    2. Validates the configuration file
    3. Displays input data requirements
    
    The setup process is designed to be idempotent - it can be run multiple
    times safely without affecting existing data or configurations.
    
    Returns:
        bool: True if setup completed successfully, False otherwise
    """
    
    print("Archaeological Detection Pipeline Setup")
    print("=" * 50)
    
    try:
        # Step 1: Create the directory structure for all pipeline stages
        create_directory_structure()
        print()
        
        # Step 2: Validate the configuration file structure and content
        config_ok = validate_config()
        print()
        
        # Step 3: Display information about required external data
        check_input_requirements()
        print()
        
        # Setup completed successfully
        if config_ok:
            print("[SUCCESS] Pipeline setup completed successfully")
            print("Next steps:")
            print("  1. Obtain required input data (see above)")
            print("  2. Configure Copernicus credentials in .env file")
            print("  3. Run: python run_pipeline.py --full")
        else:
            print("[WARNING] Setup completed with configuration issues")
            print("Please fix configuration errors before running pipeline")
        
        return config_ok
        
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")
        return False

if __name__ == "__main__":
    # Execute setup when script is run directly
    success = main()
    
    # Exit with appropriate code for shell scripting
    exit(0 if success else 1)