#!/usr/bin/env python3
"""
Stage 1: Deforestation Analysis and Archaeological Candidate Filtering

This module processes TerraBrasilis PRODES deforestation data to identify
areas with archaeological potential. It applies spatial, temporal, and
geometric filters to deforestation polygons and scores them based on
archaeological criteria.

Key Functions:
- Load and filter PRODES deforestation data
- Apply archaeological-specific filtering criteria
- Score candidates based on size, timing, and shape
- Generate Sentinel-2 query parameters for promising areas
- Integrate with OpenAI for contextual analysis

The output feeds into Stage 2 for satellite imagery analysis.

Authors: Archaeological AI Team
License: MIT
"""

import json
import yaml
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from result_analyzer import OpenAIAnalyzer


class DeforestationArchaeologyProcessor:
    """
    Processes deforestation data for archaeological candidate selection.
    
    This class handles the complete workflow of Stage 1, from loading
    raw PRODES data through generating prioritized candidates for
    further analysis.
    """
    
    def __init__(self):
        """Initialize processor with configuration and setup directories."""
        # Load configuration parameters
        with open("config/parameters.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.study_area = self.config['study_area']
        self.defor_params = self.config['deforestation']
        self.paths = self.config['paths']
        
        # Create output directory for Stage 1 results
        self.output_dir = Path(self.paths['stage1_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set current year for age calculations
        self.current_year = datetime.now().year
        
        # Initialize OpenAI analyzer for contextual analysis
        self.openai_analyzer = OpenAIAnalyzer()
        
    def log_step(self, step, message):
        """Log processing steps with timestamps for debugging and monitoring."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {step}: {message}")
        
    def load_and_filter_deforestation(self):
        """
        Load PRODES deforestation data and apply initial spatial/temporal filters.
        
        Returns:
            GeoDataFrame: Filtered deforestation polygons within study area and time range
        """
        self.log_step("LOAD", "Loading PRODES deforestation data")
        
        gpkg_path = Path(self.paths['input_gpkg'])
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GPKG file not found: {gpkg_path}")
        
        # Try different potential layer names in the GPKG file
        layer_names = ['yearly_deforestation', 'prodes_deforestation', 'deforestation']
        gdf = None
        
        for layer in layer_names:
            try:
                gdf = gpd.read_file(gpkg_path, layer=layer)
                self.log_step("LOAD", f"Loaded from layer: {layer}")
                break
            except:
                continue
        
        # Fallback to default layer if named layers not found
        if gdf is None:
            gdf = gpd.read_file(gpkg_path)
            self.log_step("LOAD", "Loaded default layer")
        
        print(f"   [INFO] Loaded {len(gdf):,} total deforestation polygons")
        
        # Standardize coordinate reference system to WGS84
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # Standardize year column naming (handle different PRODES formats)
        year_columns = [col for col in gdf.columns if any(term in col.lower() 
                       for term in ['year', 'ano', 'data'])]
        if year_columns:
            year_col = year_columns[0]
            if year_col != 'year':
                gdf = gdf.rename(columns={year_col: 'year'})
        
        # Handle different year data formats (date strings vs integers)
        if 'year' in gdf.columns:
            if gdf['year'].dtype == 'object':
                try:
                    # Try parsing as dates first
                    gdf['year'] = pd.to_datetime(gdf['year']).dt.year
                except:
                    # Extract 4-digit years from strings
                    gdf['year'] = gdf['year'].astype(str).str.extract(r'(\d{4})')[0].astype(float)
        
        # Calculate polygon areas in hectares using equal-area projection
        brazil_albers = '+proj=aea +lat_1=-2 +lat_2=-22 +lat_0=-12 +lon_0=-54 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs'
        gdf_projected = gdf.to_crs(brazil_albers)
        gdf['area_ha'] = gdf_projected.geometry.area / 10000
        
        # Apply spatial filter to study area bounds
        bounds = self.study_area['bounds']
        study_area_data = gdf.cx[
            bounds['min_lon']:bounds['max_lon'], 
            bounds['min_lat']:bounds['max_lat']
        ].copy()
        
        self.log_step("FILTER", f"Spatial filter: {len(study_area_data):,} polygons in study area")
        
        # Apply temporal filter based on configuration
        if 'year' in study_area_data.columns:
            temporal_data = study_area_data[
                (study_area_data['year'] >= self.defor_params['start_year']) &
                (study_area_data['year'] <= self.defor_params['end_year'])
            ].copy()
            self.log_step("FILTER", f"Temporal filter: {len(temporal_data):,} polygons")
        else:
            temporal_data = study_area_data
        
        return temporal_data
        
    def apply_archaeological_filters(self, gdf):
        """
        Apply filters specific to archaeological site detection.
        
        These filters focus on characteristics that make deforestation
        polygons more likely to reveal archaeological features:
        - Age since deforestation (allows vegetation recovery)
        - Size constraints (eliminates too large/small areas)
        - Shape regularity (geometric patterns suggest human activity)
        
        Args:
            gdf (GeoDataFrame): Deforestation polygons to filter
            
        Returns:
            GeoDataFrame: Filtered polygons meeting archaeological criteria
        """
        self.log_step("ARCHAEO", "Applying archaeological filtering criteria")
        
        working_gdf = gdf.copy()
        
        # Calculate age of deforestation (older = better for archaeology)
        if 'year' in working_gdf.columns:
            working_gdf['years_since_deforestation'] = self.current_year - working_gdf['year']
            
            # Filter by age range (need time for vegetation clearing but not too old)
            age_filtered = working_gdf[
                (working_gdf['years_since_deforestation'] >= self.defor_params['min_age_years']) &
                (working_gdf['years_since_deforestation'] <= self.defor_params['max_age_years'])
            ].copy()
            
            print(f"   [TIME] Age filter ({self.defor_params['min_age_years']}-{self.defor_params['max_age_years']} years): "
                  f"{len(age_filtered):,} polygons")
            working_gdf = age_filtered
        
        # Filter by polygon size (archaeological sites have characteristic sizes)
        size_filtered = working_gdf[
            (working_gdf['area_ha'] >= self.defor_params['min_size_ha']) &
            (working_gdf['area_ha'] <= self.defor_params['max_size_ha'])
        ].copy()
        
        print(f"   [SIZE] Size filter ({self.defor_params['min_size_ha']}-{self.defor_params['max_size_ha']} ha): "
              f"{len(size_filtered):,} polygons")
        working_gdf = size_filtered
        
        # Filter by shape regularity (geometric shapes suggest human activity)
        bounds = working_gdf.bounds
        working_gdf['bbox_width'] = bounds['maxx'] - bounds['minx']
        working_gdf['bbox_height'] = bounds['maxy'] - bounds['miny']
        working_gdf['bbox_ratio'] = np.maximum(
            working_gdf['bbox_width'] / working_gdf['bbox_height'],
            working_gdf['bbox_height'] / working_gdf['bbox_width']
        )
        
        shape_filtered = working_gdf[
            working_gdf['bbox_ratio'] <= self.defor_params['max_bbox_ratio']
        ].copy()
        
        print(f"   [SHAPE] Shape filter (ratio <= {self.defor_params['max_bbox_ratio']}): "
              f"{len(shape_filtered):,} polygons")
        
        return shape_filtered
        
    def prioritize_candidates(self, gdf):
        """
        Score and prioritize archaeological candidates with optional GPT analysis.
        
        Creates a scoring system based on:
        - Optimal size ranges for archaeological sites
        - Optimal timing for archaeological visibility
        - Shape compactness indicating human construction
        
        Args:
            gdf (GeoDataFrame): Filtered deforestation polygons
            
        Returns:
            GeoDataFrame: Scored and prioritized candidates
        """
        self.log_step("PRIORITY", "Scoring archaeological candidates with GPT analysis")
        
        working_gdf = gdf.copy()
        working_gdf['archaeology_score'] = 0
        
        # Size scoring - optimal ranges get higher scores
        optimal_size_mask = (
            (working_gdf['area_ha'] >= self.defor_params['optimal_size_min_ha']) &
            (working_gdf['area_ha'] <= self.defor_params['optimal_size_max_ha'])
        )
        working_gdf.loc[optimal_size_mask, 'archaeology_score'] += 3
        
        # Timing scoring - optimal age ranges get higher scores
        if 'years_since_deforestation' in working_gdf.columns:
            optimal_timing_mask = (
                (working_gdf['years_since_deforestation'] >= self.defor_params['optimal_timing_min_years']) &
                (working_gdf['years_since_deforestation'] <= self.defor_params['optimal_timing_max_years'])
            )
            working_gdf.loc[optimal_timing_mask, 'archaeology_score'] += 2
        
        # Shape scoring - more compact shapes get higher scores
        compact_mask = working_gdf['bbox_ratio'] < 2
        working_gdf.loc[compact_mask, 'archaeology_score'] += 1
        
        # Sort by archaeological score (highest first)
        working_gdf = working_gdf.sort_values('archaeology_score', ascending=False)
        
        # Create confidence categories based on scores
        working_gdf['confidence'] = pd.cut(
            working_gdf['archaeology_score'],
            bins=[-1, 1, 3, 5, 10],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        print(f"   [PRIORITY] Candidate scoring complete")
        confidence_counts = working_gdf['confidence'].value_counts()
        for conf_level in ['Very High', 'High', 'Medium', 'Low']:
            if conf_level in confidence_counts:
                count = confidence_counts[conf_level]
                print(f"      {conf_level}: {count:,} candidates")
        
        # Optional: GPT analysis for top candidates (if OpenAI is available)
        top_candidates = working_gdf.head(10)
        
        if len(top_candidates) > 0:
            print(f"   [GPT] Analyzing top {len(top_candidates)} candidates with ChatGPT...")
            
            gpt_analyses = []
            for idx, candidate in top_candidates.iterrows():
                try:
                    # Prepare candidate data for GPT analysis
                    candidate_data = {
                        'lat': candidate.geometry.centroid.y,
                        'lon': candidate.geometry.centroid.x,
                        'year': candidate.get('year', 'Unknown'),
                        'area_ha': candidate['area_ha'],
                        'bbox_ratio': candidate.get('bbox_ratio', 1.0),
                        'years_since_deforestation': candidate.get('years_since_deforestation', 0)
                    }
                    
                    # Get GPT contextual analysis
                    gpt_result = self.openai_analyzer.analyze_deforestation_context(candidate_data)
                    
                    if gpt_result['success']:
                        gpt_analyses.append({
                            'index': idx,
                            'gpt_analysis': gpt_result['response'],
                            'processing_time': gpt_result['processing_time']
                        })
                        print(f"      ✓ Analyzed candidate {len(gpt_analyses)}")
                    else:
                        print(f"      ✗ Failed to analyze candidate: {gpt_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"      ✗ Error analyzing candidate: {e}")
                    continue
            
            # Add GPT analyses back to dataframe
            for analysis in gpt_analyses:
                working_gdf.loc[analysis['index'], 'gpt_context'] = analysis['gpt_analysis']
                working_gdf.loc[analysis['index'], 'gpt_processing_time'] = analysis['processing_time']
            
            print(f"   [GPT] Completed {len(gpt_analyses)} GPT analyses")
            
            # Generate overall pattern interpretation
            if len(gpt_analyses) > 0:
                print(f"   [GPT] Generating overall pattern interpretation...")
                
                pattern_summary = {
                    'total_candidates': len(working_gdf),
                    'gpt_analyzed': len(gpt_analyses),
                    'year_range': f"{working_gdf['year'].min()}-{working_gdf['year'].max()}" if 'year' in working_gdf.columns else "Unknown",
                    'size_range': f"{working_gdf['area_ha'].min():.1f}-{working_gdf['area_ha'].max():.1f}",
                    'top_score': working_gdf['archaeology_score'].max(),
                    'study_area': self.study_area['name']
                }
                
                interpretation = self.openai_analyzer.interpret_deforestation_patterns(pattern_summary)
                
                if interpretation['success']:
                    self.pattern_interpretation = interpretation['response']
                    print(f"      ✓ Generated pattern interpretation")
                else:
                    print(f"      ✗ Failed to generate interpretation: {interpretation.get('error', 'Unknown error')}")
        
        return working_gdf
        
    def generate_sentinel_queries(self, candidates_gdf, top_n=20):
        """
        Generate Sentinel-2 query parameters for top archaeological candidates.
        
        Creates geographic bounding boxes and metadata needed for Stage 2
        satellite data download and analysis.
        
        Args:
            candidates_gdf (GeoDataFrame): Prioritized archaeological candidates
            top_n (int): Number of top candidates to generate queries for
            
        Returns:
            list: List of query dictionaries for Sentinel-2 data acquisition
        """
        self.log_step("QUERIES", f"Generating Sentinel-2 queries for top {top_n} candidates")
        
        top_candidates = candidates_gdf.head(top_n)
        buffer_deg = self.config['sentinel_download']['buffer_degrees']
        
        sentinel_queries = []
        
        for idx, polygon in top_candidates.iterrows():
            bounds = polygon.geometry.bounds
            centroid = polygon.geometry.centroid
            
            # Create query parameters for Sentinel-2 API
            query = {
                'polygon_id': int(idx),
                'candidate_index': len(sentinel_queries),
                'query_bounds': {
                    'min_lon': bounds[0] - buffer_deg,
                    'min_lat': bounds[1] - buffer_deg,
                    'max_lon': bounds[2] + buffer_deg,
                    'max_lat': bounds[3] + buffer_deg
                },
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x,
                'deforestation_year': polygon.get('year', 'unknown'),
                'area_ha': polygon['area_ha'],
                'archaeology_score': polygon['archaeology_score'],
                'confidence': str(polygon['confidence'])
            }
            
            sentinel_queries.append(query)
        
        print(f"   [SUCCESS] Generated {len(sentinel_queries)} Sentinel-2 query parameters")
        return sentinel_queries
        
    def save_results(self, candidates_gdf, sentinel_queries):
        """
        Save all Stage 1 results including GPT analyses and metadata.
        
        Outputs:
        - Shapefile of archaeological candidates
        - CSV summary with key attributes
        - JSON file with Sentinel-2 queries
        - Metadata and processing information
        - GPT analysis results (if available)
        """
        self.log_step("SAVE", "Saving Stage 1 results with GPT analyses")
        
        # Save candidates as shapefile for GIS use
        candidates_shp = self.output_dir / "archaeological_candidates.shp"
        candidates_gdf.to_file(candidates_shp)
        
        # Save candidates as CSV for easy analysis
        candidates_csv = Path(self.paths['deforestation_candidates'])
        candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Select key columns for CSV output
        csv_cols = ['year', 'area_ha', 'years_since_deforestation', 'bbox_ratio', 
                'archaeology_score', 'confidence']
        available_cols = [col for col in csv_cols if col in candidates_gdf.columns]
        
        # Add coordinate information
        coords_df = candidates_gdf.copy()
        coords_df['centroid_lat'] = coords_df.geometry.centroid.y
        coords_df['centroid_lon'] = coords_df.geometry.centroid.x
        
        # Include GPT analysis columns if available
        if 'gpt_context' in coords_df.columns:
            available_cols.append('gpt_context')
        if 'gpt_processing_time' in coords_df.columns:
            available_cols.append('gpt_processing_time')
        
        coords_df[available_cols + ['centroid_lat', 'centroid_lon']].to_csv(
            candidates_csv, index=True)
        
        # Save Sentinel-2 query parameters for Stage 2
        queries_path = Path(self.paths['sentinel_queries'])
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        with open(queries_path, 'w') as f:
            json.dump(sentinel_queries, f, indent=2)
        
        # Save processing metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'study_area': self.study_area,
            'parameters': self.defor_params,
            'results': {
                'total_candidates': len(candidates_gdf),
                'top_queries_generated': len(sentinel_queries)
            },
            'gpt_integration': {
                'enabled': True,
                'candidates_analyzed': len([idx for idx, row in candidates_gdf.iterrows() 
                                        if 'gpt_context' in row and pd.notna(row['gpt_context'])]),
                'pattern_interpretation': hasattr(self, 'pattern_interpretation')
            }
        }
        
        metadata_path = self.output_dir / 'stage1_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save GPT analysis results if available
        if hasattr(self, 'pattern_interpretation'):
            gpt_results = {
                'timestamp': datetime.now().isoformat(),
                'stage': 'STAGE_1_DEFORESTATION',
                'pattern_interpretation': self.pattern_interpretation,
                'individual_analyses': []
            }
            
            # Save individual GPT analyses
            for idx, row in candidates_gdf.iterrows():
                if 'gpt_context' in row and pd.notna(row['gpt_context']):
                    gpt_results['individual_analyses'].append({
                        'candidate_index': idx,
                        'lat': row.geometry.centroid.y,
                        'lon': row.geometry.centroid.x,
                        'gpt_analysis': row['gpt_context'],
                        'processing_time': row.get('gpt_processing_time', 0)
                    })
            
            gpt_results_path = self.output_dir / 'stage1_gpt_analyses.json'
            with open(gpt_results_path, 'w') as f:
                json.dump(gpt_results, f, indent=2)
            
            print(f"      GPT analyses: {gpt_results_path}")
        
        print(f"   [SAVE] Results saved:")
        print(f"      Shapefile: {candidates_shp}")
        print(f"      CSV: {candidates_csv}")
        print(f"      Queries: {queries_path}")
        print(f"      Metadata: {metadata_path}")
        if hasattr(self, 'pattern_interpretation'):
            print(f"      GPT Enhancement: ENABLED")
        else:
            print(f"      GPT Enhancement: LIMITED (no pattern interpretation)")
        
    def create_summary_visualization(self, original_gdf, final_gdf):
        """
        Create summary visualization of Stage 1 processing results.
        
        Generates a 2x2 plot showing:
        - Temporal distribution of deforestation
        - Size distribution of polygons
        - Archaeological priority scores
        - Confidence level distribution
        """
        self.log_step("VIZ", "Creating summary visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Temporal distribution
        ax1 = axes[0, 0]
        if 'year' in original_gdf.columns:
            original_gdf['year'].value_counts().sort_index().plot(
                kind='bar', ax=ax1, alpha=0.7, label='Original', color='lightblue')
            if len(final_gdf) > 0:
                final_gdf['year'].value_counts().sort_index().plot(
                    kind='bar', ax=ax1, alpha=0.8, label='Candidates', color='darkblue')
            ax1.set_title('Deforestation by Year')
            ax1.legend()
        
        # 2. Size distribution
        ax2 = axes[0, 1]
        original_gdf['area_ha'].hist(bins=50, alpha=0.7, label='Original', 
                                    color='lightcoral', ax=ax2)
        if len(final_gdf) > 0:
            final_gdf['area_ha'].hist(bins=30, alpha=0.8, label='Candidates', 
                                     color='darkred', ax=ax2)
        ax2.set_title('Area Distribution')
        ax2.set_xlim(0, 200)
        ax2.legend()
        
        # 3. Archaeological scores
        ax3 = axes[1, 0]
        if len(final_gdf) > 0:
            final_gdf['archaeology_score'].hist(bins=range(0, 8), alpha=0.8, 
                                               color='green', ax=ax3)
            ax3.set_title('Archaeological Priority Scores')
        
        # 4. Confidence distribution
        ax4 = axes[1, 1]
        if len(final_gdf) > 0:
            final_gdf['confidence'].value_counts().plot(kind='pie', ax=ax4, autopct='%1.1f%%')
            ax4.set_title('Confidence Distribution')
        
        plt.suptitle(f'Stage 1: Deforestation Analysis Results\n'
                    f'{self.study_area["name"]} Study Area', fontsize=14)
        plt.tight_layout()
        
        viz_path = self.output_dir / 'stage1_summary.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   [VIZ] Visualization saved: {viz_path}")
        
    def run_stage1(self):
        """
        Execute the complete Stage 1 processing pipeline.
        
        This is the main entry point that orchestrates all Stage 1 operations:
        1. Load and filter deforestation data
        2. Apply archaeological criteria
        3. Score and prioritize candidates
        4. Generate queries for Stage 2
        5. Save results and create visualizations
        
        Returns:
            bool: True if Stage 1 completed successfully
        """
        print("STAGE 1: Deforestation Analysis")
        print("=" * 50)
        
        try:
            # Load and filter deforestation data
            deforestation_gdf = self.load_and_filter_deforestation()
            
            # Apply archaeological filters
            filtered_gdf = self.apply_archaeological_filters(deforestation_gdf)
            
            # Prioritize candidates with scoring system
            candidates_gdf = self.prioritize_candidates(filtered_gdf)
            
            # Generate Sentinel-2 queries for top candidates
            max_candidates = self.config['sentinel_download']['max_candidates']
            sentinel_queries = self.generate_sentinel_queries(candidates_gdf, max_candidates)
            
            # Save all results
            self.save_results(candidates_gdf, sentinel_queries)
            
            # Create summary visualization
            self.create_summary_visualization(deforestation_gdf, candidates_gdf)
            
            print(f"\nSUCCESS: Stage 1 Complete!")
            print(f"   [INFO] Original polygons: {len(deforestation_gdf):,}")
            print(f"   [RESULT] Final candidates: {len(candidates_gdf):,}")
            print(f"   [NEXT] Sentinel-2 queries: {len(sentinel_queries)}")
            print(f"   [OUTPUT] Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Stage 1 failed: {e}")
            raise


def main():
    """Main entry point for Stage 1 processing."""
    processor = DeforestationArchaeologyProcessor()
    processor.run_stage1()


if __name__ == "__main__":
    main()