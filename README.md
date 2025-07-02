# AI-Powered Archaeological Detection Pipeline

## Overview

This project is an entry for the [OpenAI to Z Challenge](https://www.kaggle.com/competitions/openai-to-z-challenge), a competition co-organized by Kaggle and OpenAI. After submitting our work, we decided to open-source the code, aiming to provide archaeologists, anthropologists, and enthusiasts with a free and easy-to-use AI-driven archaeological technology framework.

This pipeline uses AI and remote sensing data to explore potential archaeological sites in the Amazon rainforest. It analyzes deforestation patterns, satellite imagery, and elevation data to identify areas where ancient settlements might be hidden beneath the forest canopy.

**Key Features:**
- Deforestation pattern analysis to find optimal archaeological visibility
- Sentinel-2 satellite imagery processing for vegetation anomaly detection
- FABDEM elevation validation for subsurface feature confirmation
- OpenAI GPT integration for contextual analysis and interpretation
- Archaeology-themed UI for parameter management and pipeline execution

Welcome to contact us if you have any questions or ideas via the email below:\
wangzifeng157@gmail.com

**Cite this repo if it helps you in your work:**
- APA: 
"Li, L., & Wang, Z. (2025). DualVectorFoil-AI-Archaeology (Version 1.0.0) [Computer software]. https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/tree/main"

- Bibtex:
@software{Li_DualVectorFoil-AI-Archaeology_2025,
author = {Li, Linduo and Wang, Zifeng},
doi = {10.5281/zenodo.1234},
month = jun,
title = {{DualVectorFoil-AI-Archaeology}},
url = { https://github.com/BostonListener/DualVectorFoil-AI-Archaeology },
version = {1.0.0},
year = {2025}
}

## Web Interface

We've developed a beautiful, archaeology-themed web interface that makes the pipeline accessible to non-technical users. The interface provides visual parameter editing and real-time pipeline monitoring without requiring manual YAML file editing.

### Features
- **🎨 Archaeological Theme**: Professional earth-tone design with ancient-inspired visual elements
- **⚙️ Interactive Parameter Editor**: Visual editing of all pipeline configuration parameters
- **🚀 One-Click Execution**: Run setup, pipeline, checkpoint, and visualization with single clicks
- **📊 Real-Time Monitoring**: Live console output and progress tracking via WebSocket
- **🔧 Zero Code Changes**: Seamlessly integrates with existing pipeline scripts

### Web Interface Screenshots

#### Parameter Configuration
The web interface provides comprehensive parameter editing across all pipeline stages:

![Parameter Configuration 1](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/parameter01.png)

![Parameter Configuration 2](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/parameter02.png)

![Parameter Configuration 3](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/parameter03.png)

#### Pipeline Execution Panel
Execute all pipeline stages with professional action buttons:

![Workflow Panel](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/working_panel.png)

#### Real-Time Process Monitoring
Monitor pipeline execution with live console output:

![Workflow Console](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/working_console.png)

## Pipeline Workflow

The complete archaeological detection workflow consists of three main stages:

![Pipeline Workflow - Stage 1](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow01.png)

![Pipeline Workflow - Stage 2](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow02.png)

![Pipeline Workflow - Stage 3](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow03.png)

![Pipeline Workflow - Final Results](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow04.png)

## Quick Start

### 1. Installation
```bash
# Install all dependencies
pip install rasterio geopandas shapely scikit-image scipy requests matplotlib pandas numpy folium python-dotenv pyyaml openai flask flask-socketio
```

### 2. Required Data

Download and place these files:

**PRODES Deforestation Data:**
- Download from: https://terrabrasilis.dpi.inpe.br/en/download-files/
- File: Amazon Legal `.gpkg` file
- Location: `data/input/prodes_amazonia_legal.gpkg`

**FABDEM Elevation Data:**
- Download from: https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn
- Files: FABDEM `.zip` files for your study area
- Location: `data/input/DEM/`

### 3. Environment Setup

Create `.env` file:
```env
USER_NAME=your_copernicus_username
USER_PASSWORD=your_copernicus_password
OPENAI_API_KEY=your_openai_api_key
```

Register for free accounts:
- Copernicus Data Space: https://dataspace.copernicus.eu/
- OpenAI API: https://platform.openai.com/api-keys

### 4. Usage

#### Web Interface (Recommended)
```bash
# Start the web interface
python run_ui.py
```
Open browser to: **http://localhost:5000**

1. **Configure Parameters**: Edit all pipeline settings visually
2. **Run Setup**: Initialize directories and validate configuration  
3. **Run Pipeline**: Execute the complete 3-stage archaeological detection
4. **Run Checkpoints**: Validate competition compliance
5. **Run Visualization**: Generate site visualizations

#### Command Line
```bash
# Configure parameters manually
# Edit config/parameters.yaml for your study area

# Setup pipeline directories
python setup_pipeline.py

# Check dependencies
python run_pipeline.py --check

# Run complete pipeline
python run_pipeline.py --full

# Run checkpoints
python run_checkpoint.py
```

## Data Structure

### Input Data
```
data/input/
├── prodes_amazonia_legal.gpkg    # Deforestation polygons
└── DEM/
    └── FABDEM_*.zip              # Elevation tiles
```

### Output Structure
```
data/
├── stage1/
│   ├── archaeological_candidates.csv     # Ranked deforestation candidates
│   └── archaeological_candidates.shp     # Geographic boundaries
├── stage2/
│   ├── downloads/                         # Sentinel-2 satellite data
│   └── pattern_summary.csv               # NDVI vegetation patterns
├── stage3/
│   ├── final_archaeological_sites.csv    # Validated archaeological sites
│   ├── final_archaeological_sites.geojson
│   └── final_archaeological_sites.html   # Interactive map
├── checkpoint2_outputs/
│   ├── five_anomaly_footprints.json      # 5 candidate anomalies
│   └── checkpoint2_results.json
├── checkpoint3_outputs/
│   ├── best_site_evidence_package.json   # Single best discovery
│   └── checkpoint3_notebook.json
└── checkpoint4_outputs/
    ├── two_page_presentation.json        # Presentation materials
    └── presentation_pdf_content.txt
```

### Key Output Files

**Final Archaeological Sites** (`data/stage3/final_archaeological_sites.csv`):
- Site coordinates and measurements
- Confidence assessments from multiple data sources
- Geometric properties and pattern classifications

**Interactive Map** (`data/stage3/final_archaeological_sites.html`):
- Visualization of all discovered sites
- Elevation and satellite imagery overlays

**Checkpoint Results**:
- 5 anomaly footprints for competition compliance
- Best site documentation with evidence
- Presentation materials for live demonstration

## Web Interface Configuration

### Parameter Categories

The web interface organizes all pipeline parameters into intuitive categories:

#### **🗺️ Study Area**
- Geographic bounds definition
- Region name and coordinate boundaries
- Area of interest specification

#### **🌳 Deforestation Analysis**
- Temporal range for PRODES data analysis
- Size filters for archaeological features
- Age parameters for optimal site visibility
- Shape and optimization criteria

#### **🛰️ Sentinel-2 Configuration**
- Satellite data download parameters
- Cloud cover thresholds and preferences
- NDVI analysis sensitivity settings
- Pattern detection parameters

#### **⛰️ Elevation Validation**
- FABDEM analysis parameters
- Contour intervals and roughness thresholds
- Topographic validation criteria
- Buffer distances and pixel requirements

#### **📁 File Paths**
- Input data locations
- Output directory structure
- Stage-specific file paths

### Environment Configuration

#### Basic Configuration (`config/parameters.yaml`)

```yaml
study_area:
  name: "Acre"
  bounds:
    min_lon: -68.5
    max_lon: -67.5
    min_lat: -10.6
    max_lat: -9.6

deforestation:
  start_year: 2017
  end_year: 2022
  min_age_years: 3
  max_age_years: 8
  min_size_ha: 2.5
  max_size_ha: 100

sentinel_download:
  max_candidates: 20
  cloud_cover_threshold: 75
  buffer_degrees: 0.01

sentinel_analysis:
  parameter_grid:
    ndvi_contrast_threshold: [0.05, 0.08, 0.12]
    geometry_threshold: [0.35, 0.50, 0.65]
    min_pattern_pixels: [5, 7, 9]

dem_validation:
  buffer_distance_m: 100
  elevation_std_threshold: 0.4
  elevation_range_threshold: 1.5
  patterns_to_validate: 25
```

#### Environment Variables (`.env`)

```env
# Copernicus Data Space credentials (required)
USER_NAME=your_email@example.com
USER_PASSWORD=your_password

# OpenAI API credentials (required)
OPENAI_API_KEY=sk-your-api-key-here
```

## Expected Results

**Typical Pipeline Output:**
- Input: ~10,000 deforestation polygons
- Stage 1: 20 archaeological candidates
- Stage 2: 15 NDVI pattern detections
- Stage 3: 5-10 validated archaeological sites

![Interactive Map - All Detected Sites](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/interactive_map.png)

### Top 5 Archaeological Discoveries

Our AI-powered pipeline successfully identified several high-confidence archaeological sites. Here are the top 5 discoveries:

#### Site 01 - Circular Earthwork Complex
![Site 01](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/site01.png)

#### Site 02 - Rectangular Platform Structure
![Site 02](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/site02.png)

#### Site 03 - Linear Defensive Feature
![Site 03](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/site03.png)

#### Site 04 - Complex Geometric Pattern
![Site 04](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/site04.png)

#### Site 05 - Multi-Component Settlement
![Site 05](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/site05.png)

## Web Interface Architecture

### Backend (Flask + SocketIO)
- **Flask**: RESTful API for parameter management and script execution
- **WebSocket**: Real-time communication for live console output
- **Process Management**: Subprocess orchestration with UTF-8 encoding
- **Parameter Handling**: YAML configuration file management

### Frontend (HTML + CSS + JavaScript)
- **Archaeological Theme**: Earth tones, professional styling
- **Interactive Forms**: Dynamic parameter editing with validation
- **Real-Time Updates**: Live console output and status monitoring
- **Responsive Design**: Desktop and mobile compatibility

### Integration
- **Zero Modifications**: Works with existing pipeline scripts unchanged
- **Parameter Synchronization**: Automatic YAML file updates
- **Process Monitoring**: Real-time execution tracking
- **Error Handling**: Comprehensive error reporting and recovery

## Troubleshooting

### Common Issues

#### Web Interface Issues
**"Cannot connect to server"**
- Check if port 5000 is available
- Ensure Flask dependencies are installed: `pip install flask flask-socketio`
- Try restarting the interface: `python run_ui.py`

**"Configuration not saving"**
- Verify write permissions to `config/` directory
- Check for YAML syntax errors in manual edits
- Ensure all required parameters are filled

#### Pipeline Issues
**Authentication Errors:**
- Verify Copernicus credentials in `.env` file
- Check OpenAI API key format (starts with 'sk-')

**Data Not Found:**
- Ensure PRODES `.gpkg` file is in correct location
- Download FABDEM tiles covering your study area

**No Patterns Detected:**
- Reduce thresholds in `sentinel_analysis.parameter_grid`
- Increase `max_candidates` for more input data
- Check study area bounds cover deforested regions

**Unicode/Encoding Errors:**
- The web interface automatically handles UTF-8 encoding
- For command line usage, set: `set PYTHONIOENCODING=utf-8` (Windows) or `export PYTHONIOENCODING=utf-8` (Linux/Mac)

**Coordinate Issues:**
- Verify study area bounds in parameters
- Ensure FABDEM tiles cover the study area
- Check that deforestation data exists in the region

## Next Steps

1. **Configure Parameters**: Use the web interface to set your study area and analysis parameters
2. **Run Setup**: Initialize directories and validate configuration
3. **Execute Pipeline**: Run the complete 3-stage archaeological detection workflow
4. **Run Checkpoints**: Complete requirements with checkpoint analysis
5. **Review Results**: Examine output files and interactive maps
6. **Generate Visualizations**: Create professional site documentation
7. **Field Planning**: Use coordinates for ground-truth validation
8. **Research Documentation**: Analyze patterns and prepare academic publications

## Technical Architecture

The pipeline implements a three-stage archaeological detection workflow enhanced with a modern web interface:

1. **Stage 1 - Deforestation Analysis**: Identifies optimal areas for archaeological visibility through systematic analysis of TerraBrasilis PRODES data, applying temporal, spatial, and geometric filters to find areas where ancient settlements might be revealed.

2. **Stage 2 - Satellite Analysis**: Downloads and processes Sentinel-2 imagery to calculate NDVI patterns that indicate subsurface archaeological features through vegetation anomalies and geometric patterns.

3. **Stage 3 - Elevation Validation**: Uses FABDEM bare-earth elevation data to validate potential sites through statistical analysis of elevation signatures and terrain characteristics.

4. **Web Interface**: Provides intuitive parameter management and real-time monitoring, making the pipeline accessible to non-technical users while maintaining full compatibility with command-line usage.

Each stage feeds into OpenAI GPT models for contextual interpretation and evidence synthesis, creating a comprehensive AI-enhanced archaeological discovery system.

## Scientific Impact

This pipeline represents the revolution of archaeological methodology in this AI era by:

- **Scaling Discovery**: Enables systematic exploration of previously inaccessible Amazon regions
- **AI Integration**: Demonstrates practical application of machine learning for heritage preservation
- **Community Partnership**: Supports indigenous communities in documenting cultural landscapes
- **Methodological Innovation**: Creates reproducible framework applicable to global archaeological research
- **Accessibility**: The web interface democratizes advanced AI-archaeological tools for researchers worldwide

The results contribute to understanding pre-Columbian civilizations while respecting indigenous rights and promoting collaborative research practices.

## Performance Notes

### System Requirements
- **CPU**: Multi-core processor recommended for parallel processing
- **RAM**: 8GB minimum, 16GB recommended for large study areas
- **Storage**: 10-50GB depending on satellite data downloads
- **Network**: Stable internet connection for data downloads

### Optimization Tips
- **Study Area Size**: Smaller areas (1°×1°) process faster than large regions
- **Temporal Range**: Limiting years reduces processing time
- **Candidate Limits**: Adjust `max_candidates` based on computational resources
- **Parallel Processing**: Multiple CPU cores automatically utilized where possible

## Version History

- **v1.0.0** (2025): Initial release with complete pipeline and web interface
- Core archaeological detection algorithms
- Competition checkpoint compliance
- Professional web interface with real-time monitoring
- Comprehensive documentation and examples

## Acknowledgement

We would like to express our sincere gratitude to our team members who contributed their valuable expertise in archaeology and anthropology:

- **Yifan Wu**: MS at UCL, Central Asian Studies
- **Lienong Zhang**: PhD student at University of Pittsburgh, Anthropology (the society and technology in Amazon Basin)
- **Tianyu Yao**: Freshman at Rutgers University, Anthropology

Special thanks to the open-source community and the organizations providing the essential datasets that make this research possible:
- **TerraBrasilis/INPE** for PRODES deforestation data
- **ESA Copernicus** for Sentinel-2 satellite imagery
- **University of Bristol** for FABDEM elevation data
- **OpenAI** for advanced language model capabilities