# AI-Powered Archaeological Detection Pipeline

## Overview

This project is an entry for the [OpenAI to Z Challenge](https://www.kaggle.com/competitions/openai-to-z-challenge), a competition co-organized by Kaggle and OpenAI. After submitting our work, we decided to open-source the code, aiming to provide archaeologists, anthropologists, and enthusiasts with a free and easy-to-use AI-driven archaeological technology framework.

This pipeline uses AI and remote sensing data to explore potential archaeological sites in the Amazon rainforest. It analyzes deforestation patterns, satellite imagery, and elevation data to identify areas where ancient settlements might be hidden beneath the forest canopy.

**Key Features:**
- Deforestation pattern analysis to find optimal archaeological visibility
- Sentinel-2 satellite imagery processing for vegetation anomaly detection
- FABDEM elevation validation for subsurface feature confirmation
- OpenAI GPT integration for contextual analysis and interpretation

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
url = {https://github.com/BostonListener/DualVectorFoil-AI-Archaeology},
version = {1.0.0},
year = {2025}
}

## Pipeline Workflow

The complete archaeological detection workflow consists of three main stages:

![Pipeline Workflow - Stage 1](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow01.png)

![Pipeline Workflow - Stage 2](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow02.png)

![Pipeline Workflow - Stage 3](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow03.png)

![Pipeline Workflow - Final Results](https://github.com/BostonListener/DualVectorFoil-AI-Archaeology/blob/main/image/workflow04.png)

## Quick Start

### 1. Installation
```bash
pip install rasterio geopandas shapely scikit-image scipy requests matplotlib pandas numpy folium python-dotenv pyyaml openai
python setup_pipeline.py
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

### 4. Configuration

Edit `config/parameters.yaml` for your study area:
```yaml
study_area:
  name: "Your Study Area"
  bounds:
    min_lon: -68.5
    max_lon: -67.5
    min_lat: -10.6
    max_lat: -9.6
```

## Execution Order

### Main Pipeline (Stages 1-3)
```bash
# Check dependencies
python run_pipeline.py --check

# Run complete pipeline
python run_pipeline.py --full

# Or run individual stages:
python run_pipeline.py --stage 1  # Deforestation analysis
python run_pipeline.py --stage 2  # Satellite data processing
python run_pipeline.py --stage 3  # Elevation validation
```

### Checkpoint Analysis (Competition Requirements)
```bash
# Run all checkpoints
python run_checkpoint.py

# Or run individual checkpoints:
python checkpoint2_analysis.py   # Early explorer analysis
python checkpoint3_notebook.py   # Best site discovery
python checkpoint4_story.py      # Story and impact
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

## Environment Configuration

### Basic Configuration (`config/parameters.yaml`)

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

### Environment Variables (`.env`)

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

**Checkpoint Compliance:**
- Exactly 5 anomaly footprints with coordinates
- Single best site with comprehensive evidence

## Troubleshooting

### Common Issues

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

**Coordinate Issues:**
- Verify study area bounds in `parameters.yaml`
- Ensure FABDEM tiles cover the study area
- Check that deforestation data exists in the region

## Next Steps

1. **Run Pipeline**: Execute stages 1-3 to discover archaeological sites
2. **Run Checkpoints**: Complete competition requirements with checkpoint analysis
3. **Validate Results**: Review output files and interactive maps
4. **Field Planning**: Use coordinates for ground-truth validation
5. **Research**: Analyze patterns and prepare academic documentation

## Technical Architecture

The pipeline implements a three-stage archaeological detection workflow:

1. **Stage 1 - Deforestation Analysis**: Identifies optimal areas for archaeological visibility through systematic analysis of TerraBrasilis PRODES data, applying temporal, spatial, and geometric filters to find areas where ancient settlements might be revealed.

2. **Stage 2 - Satellite Analysis**: Downloads and processes Sentinel-2 imagery to calculate NDVI patterns that indicate subsurface archaeological features through vegetation anomalies and geometric patterns.

3. **Stage 3 - Elevation Validation**: Uses FABDEM bare-earth elevation data to validate potential sites through statistical analysis of elevation signatures and terrain characteristics.

Each stage feeds into OpenAI GPT models for contextual interpretation and evidence synthesis, creating a comprehensive AI-enhanced archaeological discovery system.

## Scientific Impact

This pipeline represents a breakthrough in archaeological methodology by:

- **Scaling Discovery**: Enables systematic exploration of previously inaccessible Amazon regions
- **AI Integration**: Demonstrates practical application of machine learning for heritage preservation
- **Community Partnership**: Supports indigenous communities in documenting cultural landscapes
- **Methodological Innovation**: Creates reproducible framework applicable to global archaeological research

The results contribute to understanding pre-Columbian civilizations while respecting indigenous rights and promoting collaborative research practices.

## Acknowledgement

We would like to express our sincere gratitude to our team members who contributed their valuable expertise in archaeology and anthropology:

- **Yifan Wu**: MS at UCL, Central Asian Studies
- **Lienong Zhang**: PhD student at University of Pittsburgh, Anthropology (the society and technology in Amazon Basin)
- **Tianyu Yao**: Freshman at Rutgers University, Anthropology
