#!/usr/bin/env python3
"""
Archaeological Site Visualization System

This module creates comprehensive multi-scale visualizations of detected archaeological
sites using Sentinel-2 satellite imagery and FABDEM elevation data. It generates
high-quality visualizations suitable for field validation, academic presentation,
and research documentation.

Key Features:
- Multi-scale visualization (detail, primary, context views)
- Sentinel-2 RGB and NDVI overlays with archaeological site boundaries
- FABDEM elevation with detailed contour mapping
- Automated site boundary detection and overlay
- High-resolution outputs suitable for field use (300 DPI)
- Comprehensive metadata and scale information

The visualization system supports field archaeologists by providing:
1. Detailed site imagery for navigation and feature identification
2. Elevation contours for understanding site topography
3. Multiple scale contexts for landscape understanding
4. Professional quality outputs for publication and presentation

Usage:
    visualizer = ArchaeologicalSiteVisualizer()
    visualizer.run_visualization()

Requirements:
    - Completed pipeline results (Stage 2 or Stage 3)
    - FABDEM elevation data in input directory
    - Sentinel-2 data downloaded during pipeline execution
    - Python packages: rasterio, geopandas, matplotlib, numpy

Authors: Archaeological AI Team
License: MIT
"""

import yaml
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point, box
from pathlib import Path
import zipfile
import tempfile
import math
from datetime import datetime
import json

class ArchaeologicalSiteVisualizer:
    """
    Comprehensive visualization system for archaeological sites detected through the pipeline.
    
    This class creates professional-quality visualizations that combine satellite imagery,
    elevation data, and archaeological site boundaries at multiple scales. The output
    visualizations are designed to support field validation, academic presentation,
    and research documentation.
    
    Visualization Scales:
    - Detail (200m radius): Maximum detail for site features and immediate context
    - Primary (500m radius): Main validation scale for archaeological assessment  
    - Context (1500m radius): Landscape context and regional setting
    
    Data Integration:
    - Sentinel-2 RGB and NDVI for vegetation and land use context
    - FABDEM bare-earth elevation with contour mapping
    - Archaeological site boundaries and confidence assessments
    """
    
    def __init__(self):
        """
        Initialize visualizer with pipeline configuration and output structure.
        
        Loads configuration parameters, sets up output directories, and defines
        visualization scales optimized for archaeological field work and analysis.
        """
        
        # Load pipeline configuration for data paths and parameters
        with open("config/parameters.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths = self.config['paths']
        
        # Create output directory for all visualization products
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define visualization scales optimized for archaeological analysis
        # Based on typical archaeological site sizes and field survey requirements
        self.scales = {
            'detail': 200,      # 200m radius - maximum detail for feature identification
            'primary': 500,     # 500m radius - main validation scale for site assessment
            'context': 1500     # 1500m radius - landscape context and regional setting
        }
        
        # Initialize data storage for loaded datasets
        self.sites_gdf = None           # Archaeological sites geodataframe
        self.fabdem_data = None         # FABDEM elevation array
        self.fabdem_transform = None    # FABDEM geotransform
        self.fabdem_crs = None          # FABDEM coordinate reference system
        self.fabdem_bounds = None       # FABDEM geographic bounds
        
        print(f"[INIT] Archaeological Site Visualizer")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Scales: Detail {self.scales['detail']}m, Primary {self.scales['primary']}m, Context {self.scales['context']}m")
        
    def log_step(self, step, message):
        """
        Log processing steps with timestamps for monitoring and debugging.
        
        Args:
            step (str): Processing step identifier
            message (str): Descriptive message about the step
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {step}: {message}")
        
    def load_archaeological_sites(self):
        """
        Load final archaeological sites from pipeline results.
        
        Attempts to load archaeological sites from GeoJSON format first (preserving
        geometry), then falls back to CSV format. Creates proper GeoDataFrame
        with point geometries for spatial analysis and visualization.
        
        Returns:
            GeoDataFrame: Archaeological sites with geometries and attributes
            
        Raises:
            FileNotFoundError: If no archaeological sites are found
        """
        
        self.log_step("LOAD", "Loading archaeological sites from pipeline")
        
        # Attempt to load from GeoJSON format (preserves spatial data)
        geojson_path = Path(self.paths['stage3_dir']) / 'final_archaeological_sites.geojson'
        csv_path = Path(self.paths['final_sites'])
        
        if geojson_path.exists():
            # Load from GeoJSON with preserved geometries
            self.sites_gdf = gpd.read_file(geojson_path)
            print(f"   [SUCCESS] Loaded {len(self.sites_gdf)} sites from GeoJSON")
        elif csv_path.exists():
            # Load from CSV and create geometries from coordinate columns
            sites_df = pd.read_csv(csv_path)
            # Create point geometries from latitude/longitude coordinates
            geometry = [Point(row['lon'], row['lat']) for _, row in sites_df.iterrows()]
            self.sites_gdf = gpd.GeoDataFrame(sites_df, geometry=geometry, crs='EPSG:4326')
            print(f"   [SUCCESS] Loaded {len(self.sites_gdf)} sites from CSV")
        else:
            raise FileNotFoundError("No archaeological sites found. Run pipeline stages first.")
        
        # Validate and report coordinate ranges for quality control
        lat_range = (self.sites_gdf.geometry.y.min(), self.sites_gdf.geometry.y.max())
        lon_range = (self.sites_gdf.geometry.x.min(), self.sites_gdf.geometry.x.max())
        
        print(f"   [INFO] Site coordinates: Lat {lat_range[0]:.4f} to {lat_range[1]:.4f}")
        print(f"                          Lon {lon_range[0]:.4f} to {lon_range[1]:.4f}")
        
        return self.sites_gdf
        
    def load_fabdem_data(self):
        """
        Load and merge FABDEM elevation data from input directory.
        
        Handles both single-tile and multi-tile FABDEM datasets by using rasterio.merge
        for seamless mosaicking. FABDEM (Forest And Buildings removed Copernicus DEM)
        provides bare-earth elevation data ideal for archaeological analysis.
        
        Returns:
            ndarray: Merged FABDEM elevation data
            
        Raises:
            FileNotFoundError: If no FABDEM TIF files are found
        """
        
        self.log_step("LOAD", "Loading FABDEM elevation data")
        
        fabdem_dir = Path(self.paths['input_dem_dir'])
        
        # Find all FABDEM TIF files (extracted by stage 3 processing)
        tif_files = list(fabdem_dir.glob("*FABDEM*.tif"))
        
        if not tif_files:
            raise FileNotFoundError(f"No FABDEM .tif files found in {fabdem_dir}")
        
        print(f"   [INFO] Found {len(tif_files)} FABDEM tiles")
        
        # Handle single vs. multiple tile scenarios
        if len(tif_files) == 1:
            # Single tile - direct loading
            fabdem_file = tif_files[0]
            print(f"   [LOAD] Using single tile: {fabdem_file.name}")
            
            with rasterio.open(fabdem_file) as src:
                self.fabdem_data = src.read(1).astype(np.float32)
                self.fabdem_transform = src.transform
                self.fabdem_crs = src.crs
                self.fabdem_bounds = src.bounds
                
                # Handle nodata values by converting to NaN
                if src.nodata is not None:
                    self.fabdem_data[self.fabdem_data == src.nodata] = np.nan
        else:
            # Multiple tiles - merge using rasterio.merge for seamless mosaic
            print(f"   [MERGE] Merging {len(tif_files)} FABDEM tiles for complete coverage")
            
            from rasterio.merge import merge
            
            datasets = []
            try:
                # Open all tile datasets for merging
                for tif_file in tif_files:
                    print(f"      Loading: {tif_file.name}")
                    datasets.append(rasterio.open(tif_file))
                
                # Merge all tiles into a single mosaic with consistent projection
                merged_data, merged_transform = merge(datasets)
                
                # Store merged results with proper data types
                self.fabdem_data = merged_data[0].astype(np.float32)
                self.fabdem_transform = merged_transform
                self.fabdem_crs = datasets[0].crs
                
                # Calculate bounds from merged transform and array dimensions
                height, width = self.fabdem_data.shape
                left = merged_transform.c
                top = merged_transform.f
                right = left + width * merged_transform.a
                bottom = top + height * merged_transform.e
                self.fabdem_bounds = rasterio.coords.BoundingBox(left, bottom, right, top)
                
                # Handle nodata values consistently across all tiles
                nodata_value = datasets[0].nodata
                if nodata_value is not None:
                    self.fabdem_data[self.fabdem_data == nodata_value] = np.nan
                    
            finally:
                # Always close datasets to prevent memory leaks
                for dataset in datasets:
                    dataset.close()
        
        # Report data quality and coverage statistics
        print(f"   [SUCCESS] FABDEM loaded: {self.fabdem_data.shape}")
        print(f"   [INFO] Elevation range: {np.nanmin(self.fabdem_data):.1f}m to {np.nanmax(self.fabdem_data):.1f}m")
        print(f"   [COVERAGE] Geographic bounds:")
        print(f"      Lat: {self.fabdem_bounds.bottom:.4f}° to {self.fabdem_bounds.top:.4f}°")
        print(f"      Lon: {self.fabdem_bounds.left:.4f}° to {self.fabdem_bounds.right:.4f}°")
        
        # Validate coverage against archaeological sites for quality assurance
        if hasattr(self, 'sites_gdf') and self.sites_gdf is not None:
            sites_within_bounds = 0
            for _, site in self.sites_gdf.iterrows():
                site_lat, site_lon = site.geometry.y, site.geometry.x
                if (self.fabdem_bounds.left <= site_lon <= self.fabdem_bounds.right and 
                    self.fabdem_bounds.bottom <= site_lat <= self.fabdem_bounds.top):
                    sites_within_bounds += 1
            
            print(f"   [VALIDATION] Sites within FABDEM coverage: {sites_within_bounds}/{len(self.sites_gdf)}")
            
            if sites_within_bounds < len(self.sites_gdf):
                missing_sites = len(self.sites_gdf) - sites_within_bounds
                print(f"   [WARNING] {missing_sites} sites outside FABDEM coverage will have empty elevation panels")
        
        return self.fabdem_data
        
    def find_sentinel_data_for_site(self, site_lat, site_lon):
        """
        Find appropriate Sentinel-2 data coverage for a specific archaeological site.
        
        Searches download metadata to identify which Sentinel-2 scenes provide
        coverage for the site location. Uses proximity analysis to find the
        best available dataset.
        
        Args:
            site_lat (float): Site latitude in decimal degrees
            site_lon (float): Site longitude in decimal degrees
            
        Returns:
            dict or None: Best candidate info with download path and metadata
        """
        
        # Load download metadata created during pipeline Stage 2
        metadata_path = Path(self.paths['stage2_dir']) / 'download_metadata.json'
        
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r') as f:
            download_metadata = json.load(f)
        
        candidates_info = download_metadata.get('candidates_info', {})
        
        # Find the closest candidate site with available Sentinel-2 data
        best_candidate = None
        min_distance = float('inf')
        
        for candidate_idx, candidate_info in candidates_info.items():
            candidate_data = candidate_info['candidate_data']
            cand_lat = candidate_data['centroid_lat']
            cand_lon = candidate_data['centroid_lon']
            
            # Calculate Euclidean distance (approximate for small areas)
            distance = math.sqrt((site_lat - cand_lat)**2 + (site_lon - cand_lon)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_candidate = candidate_info
        
        # Return candidate if within reasonable proximity (approximately 5km)
        if best_candidate and min_distance < 0.05:
            return best_candidate
        
        return None
        
    def extract_sentinel_data(self, candidate_info, center_lat, center_lon, buffer_meters):
        """
        Extract Sentinel-2 RGB and NDVI data around an archaeological site.
        
        Processes Sentinel-2 Level-2A data to create RGB composites and NDVI
        calculations for archaeological visualization. Handles coordinate
        transformations and clipping to the specified buffer area.
        
        Args:
            candidate_info (dict): Candidate information with download path
            center_lat (float): Center latitude for extraction
            center_lon (float): Center longitude for extraction
            buffer_meters (float): Buffer distance in meters around center
            
        Returns:
            tuple: (data_dict, candidate_info) or (None, None) if extraction fails
        """
        
        download_path = Path(candidate_info['download_path'])
        
        if not download_path.exists():
            return None, None
            
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract ZIP archive containing Sentinel-2 data
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                
                # Navigate Sentinel-2 directory structure (.SAFE format)
                safe_dirs = list(temp_path.glob("*.SAFE"))
                if not safe_dirs:
                    return None, None
                
                safe_dir = safe_dirs[0]
                granule_dirs = list((safe_dir / "GRANULE").glob("*"))
                if not granule_dirs:
                    return None, None
                
                granule_dir = granule_dirs[0]
                
                # Find image data directory (varies by processing level)
                img_data_dir = granule_dir / "IMG_DATA" / "R10m"
                if not img_data_dir.exists():
                    img_data_dir = granule_dir / "IMG_DATA"
                
                # Locate required spectral bands for RGB and NDVI calculation
                red_files = list(img_data_dir.glob("*B04*.jp2"))      # Red band
                green_files = list(img_data_dir.glob("*B03*.jp2"))    # Green band
                blue_files = list(img_data_dir.glob("*B02*.jp2"))     # Blue band
                nir_files = list(img_data_dir.glob("*B08*.jp2"))      # Near-infrared band
                
                if not all([red_files, green_files, blue_files, nir_files]):
                    return None, None
                
                # Create buffer area around site center
                buffer_deg = buffer_meters / 111320  # Approximate conversion to degrees
                bounds = {
                    'min_lon': center_lon - buffer_deg,
                    'max_lon': center_lon + buffer_deg,
                    'min_lat': center_lat - buffer_deg,
                    'max_lat': center_lat + buffer_deg
                }
                
                # Create clipping geometry
                clip_box_wgs84 = box(bounds['min_lon'], bounds['min_lat'], 
                                    bounds['max_lon'], bounds['max_lat'])
                
                # Read and clip all required bands
                rgb_data = None
                ndvi_data = None
                transform = None
                crs = None
                
                with rasterio.open(red_files[0]) as red_src:
                    crs = red_src.crs
                    
                    # Transform clip box to raster CRS if needed
                    if crs != 'EPSG:4326':
                        from rasterio.warp import transform_bounds
                        transformed_bounds = transform_bounds(
                            'EPSG:4326', crs,
                            bounds['min_lon'], bounds['min_lat'],
                            bounds['max_lon'], bounds['max_lat']
                        )
                        clip_box = box(*transformed_bounds)
                    else:
                        clip_box = clip_box_wgs84
                    
                    # Read and clip red band
                    red_data, transform = mask(red_src, [clip_box], crop=True)
                    red_data = red_data[0].astype(np.float32)
                    
                # Read remaining bands with same clipping geometry
                with rasterio.open(green_files[0]) as green_src:
                    green_data, _ = mask(green_src, [clip_box], crop=True)
                    green_data = green_data[0].astype(np.float32)
                    
                with rasterio.open(blue_files[0]) as blue_src:
                    blue_data, _ = mask(blue_src, [clip_box], crop=True)
                    blue_data = blue_data[0].astype(np.float32)
                    
                with rasterio.open(nir_files[0]) as nir_src:
                    nir_data, _ = mask(nir_src, [clip_box], crop=True)
                    nir_data = nir_data[0].astype(np.float32)
                
                # Create RGB composite with histogram stretching for visualization
                rgb_percentiles = []
                for band in [red_data, green_data, blue_data]:
                    valid_data = band[band > 0]
                    if len(valid_data) > 0:
                        p2, p98 = np.percentile(valid_data, [2, 98])
                        rgb_percentiles.append((p2, p98))
                    else:
                        rgb_percentiles.append((0, 1))
                
                # Apply histogram stretching and create RGB stack
                rgb_data = np.stack([
                    np.clip((red_data - rgb_percentiles[0][0]) / 
                           (rgb_percentiles[0][1] - rgb_percentiles[0][0]), 0, 1),
                    np.clip((green_data - rgb_percentiles[1][0]) / 
                           (rgb_percentiles[1][1] - rgb_percentiles[1][0]), 0, 1),
                    np.clip((blue_data - rgb_percentiles[2][0]) / 
                           (rgb_percentiles[2][1] - rgb_percentiles[2][0]), 0, 1)
                ], axis=2)
                
                # Calculate NDVI (Normalized Difference Vegetation Index)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi_data = np.where(
                        (nir_data + red_data) != 0,
                        (nir_data - red_data) / (nir_data + red_data),
                        0
                    )
                
                # Clean NDVI data and apply valid range
                ndvi_data = np.clip(ndvi_data, -1, 1)
                ndvi_data = np.nan_to_num(ndvi_data)
                
                return {
                    'rgb': rgb_data,
                    'ndvi': ndvi_data,
                    'transform': transform,
                    'crs': crs,
                    'bounds': bounds
                }, candidate_info
                
        except Exception as e:
            print(f"   [ERROR] Failed to extract Sentinel-2 data: {e}")
            return None, None
            
    def extract_fabdem_subset(self, center_lat, center_lon, buffer_meters):
        """
        Extract FABDEM elevation data subset around an archaeological site.
        
        Clips FABDEM elevation data to specified buffer area around site center.
        Handles coordinate transformations and maintains spatial reference information
        for accurate visualization.
        
        Args:
            center_lat (float): Center latitude in decimal degrees
            center_lon (float): Center longitude in decimal degrees
            buffer_meters (float): Buffer distance in meters
            
        Returns:
            dict or None: Elevation data subset with transform and CRS
        """
        
        if self.fabdem_data is None:
            return None
            
        try:
            # Transform center coordinates to FABDEM CRS if needed
            if str(self.fabdem_crs) != 'EPSG:4326':
                from rasterio.warp import transform as warp_transform
                center_x, center_y = warp_transform(
                    'EPSG:4326', self.fabdem_crs, 
                    [center_lon], [center_lat]
                )
                center_x, center_y = center_x[0], center_y[0]
            else:
                center_x, center_y = center_lon, center_lat
            
            # Convert to pixel coordinates using geotransform
            col = int((center_x - self.fabdem_bounds.left) / abs(self.fabdem_transform.a))
            row = int((self.fabdem_bounds.top - center_y) / abs(self.fabdem_transform.e))
            
            # Calculate buffer in pixels based on pixel size
            if 'UTM' in str(self.fabdem_crs) or 'METER' in str(self.fabdem_crs).upper():
                pixel_size = abs(self.fabdem_transform.a)
            else:
                pixel_size = abs(self.fabdem_transform.a) * 111320  # degrees to meters
                
            buffer_pixels = int(buffer_meters / pixel_size)
            
            # Define extraction window with bounds checking
            row_min = max(0, row - buffer_pixels)
            row_max = min(self.fabdem_data.shape[0], row + buffer_pixels)
            col_min = max(0, col - buffer_pixels)
            col_max = min(self.fabdem_data.shape[1], col + buffer_pixels)
            
            # Extract elevation subset
            elevation_subset = self.fabdem_data[row_min:row_max, col_min:col_max].copy()
            
            # Create geotransform for subset
            left = self.fabdem_bounds.left + col_min * self.fabdem_transform.a
            top = self.fabdem_bounds.top + row_min * self.fabdem_transform.e
            
            subset_transform = rasterio.transform.from_bounds(
                left, 
                top + (row_max - row_min) * self.fabdem_transform.e,
                left + (col_max - col_min) * self.fabdem_transform.a,
                top,
                col_max - col_min,
                row_max - row_min
            )
            
            return {
                'elevation': elevation_subset,
                'transform': subset_transform,
                'crs': self.fabdem_crs
            }
            
        except Exception as e:
            print(f"   [ERROR] Failed to extract FABDEM data: {e}")
            return None
            
    def create_site_visualization(self, site, scale_name):
        """
        Create comprehensive multi-panel visualization for a single archaeological site.
        
        Generates professional visualization combining Sentinel-2 RGB/NDVI imagery
        with FABDEM elevation data and archaeological site overlays. Includes
        scale bars, contours, and site boundary information.
        
        Args:
            site (GeoSeries): Archaeological site data with geometry and attributes
            scale_name (str): Visualization scale ('detail', 'primary', 'context')
            
        Returns:
            Path or None: Path to saved visualization file
        """
        
        site_id = site['site_id']
        site_lat = site.geometry.y
        site_lon = site.geometry.x
        buffer_meters = self.scales[scale_name]
        
        self.log_step("VIZ", f"Creating {scale_name} visualization for {site_id}")
        
        # Find and extract Sentinel-2 data for site
        sentinel_info = self.find_sentinel_data_for_site(site_lat, site_lon)
        
        sentinel_data = None
        if sentinel_info:
            sentinel_data, _ = self.extract_sentinel_data(
                sentinel_info, site_lat, site_lon, buffer_meters
            )
        
        # Extract FABDEM elevation data for site
        fabdem_data = self.extract_fabdem_subset(site_lat, site_lon, buffer_meters)
        
        # Check if any data is available for visualization
        if sentinel_data is None and fabdem_data is None:
            print(f"   [WARNING] No data available for {site_id}")
            return None
        
        # Determine number of plot panels based on available data
        n_plots = 2 if sentinel_data else 1
        if sentinel_data:
            n_plots = 3  # RGB, NDVI, FABDEM
        
        # Create figure with appropriate layout
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot Sentinel-2 RGB composite
        if sentinel_data:
            ax_rgb = axes[plot_idx]
            extent = self._calculate_plot_extent(sentinel_data, site_lat, site_lon, buffer_meters)
            
            ax_rgb.imshow(sentinel_data['rgb'], extent=extent, origin='upper')
            self._add_site_overlay(ax_rgb, site, extent)
            self._add_scale_bar(ax_rgb, extent, buffer_meters)
            ax_rgb.set_title(f'Sentinel-2 RGB\n{site_id}')
            ax_rgb.set_xlabel('Longitude')
            ax_rgb.set_ylabel('Latitude')
            
            plot_idx += 1
            
            # Plot Sentinel-2 NDVI with contours
            ax_ndvi = axes[plot_idx]
            ndvi_plot = ax_ndvi.imshow(sentinel_data['ndvi'], extent=extent, 
                                     cmap='RdYlGn', vmin=-0.5, vmax=0.8, origin='upper')
            
            # Add NDVI contour lines for detailed analysis
            y_coords = np.linspace(extent[2], extent[3], sentinel_data['ndvi'].shape[0])
            x_coords = np.linspace(extent[0], extent[1], sentinel_data['ndvi'].shape[1])
            X, Y = np.meshgrid(x_coords, y_coords)
            
            contour_levels = np.arange(-0.4, 0.8, 0.1)
            ax_ndvi.contour(X, Y, sentinel_data['ndvi'], levels=contour_levels, 
                          colors='black', alpha=0.3, linewidths=0.5)
            
            self._add_site_overlay(ax_ndvi, site, extent)
            self._add_scale_bar(ax_ndvi, extent, buffer_meters)
            
            plt.colorbar(ndvi_plot, ax=ax_ndvi, label='NDVI')
            ax_ndvi.set_title(f'Sentinel-2 NDVI\n{site_id}')
            ax_ndvi.set_xlabel('Longitude')
            ax_ndvi.set_ylabel('Latitude')
            
            plot_idx += 1
        
        # Plot FABDEM elevation with detailed contours
        if fabdem_data:
            ax_dem = axes[plot_idx]
            elevation = fabdem_data['elevation']
            
            # Calculate extent in geographic coordinates
            dem_extent = self._calculate_fabdem_extent(fabdem_data, site_lat, site_lon, buffer_meters)
            
            elevation_plot = ax_dem.imshow(elevation, extent=dem_extent, 
                                         cmap='terrain', origin='upper')
            
            # Add elevation contours with scale-appropriate intervals
            contour_interval = 1.0 if scale_name == 'detail' else 2.0
            
            valid_elevation = elevation[~np.isnan(elevation)]
            if len(valid_elevation) > 0:
                elev_min = np.nanmin(elevation)
                elev_max = np.nanmax(elevation)
                
                # Create contour levels at regular intervals
                contour_levels = np.arange(
                    math.floor(elev_min / contour_interval) * contour_interval,
                    math.ceil(elev_max / contour_interval) * contour_interval + contour_interval,
                    contour_interval
                )
                
                y_coords = np.linspace(dem_extent[2], dem_extent[3], elevation.shape[0])
                x_coords = np.linspace(dem_extent[0], dem_extent[1], elevation.shape[1])
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Add labeled contour lines
                contours = ax_dem.contour(X, Y, elevation, levels=contour_levels, 
                                        colors='black', alpha=0.6, linewidths=0.8)
                ax_dem.clabel(contours, inline=True, fontsize=8, fmt='%1.0fm')
            
            self._add_site_overlay(ax_dem, site, dem_extent)
            self._add_scale_bar(ax_dem, dem_extent, buffer_meters)
            
            plt.colorbar(elevation_plot, ax=ax_dem, label='Elevation (m)')
            ax_dem.set_title(f'FABDEM Elevation\n{site_id} ({contour_interval}m contours)')
            ax_dem.set_xlabel('Longitude')
            ax_dem.set_ylabel('Latitude')
        
        # Add overall title and save visualization
        plt.suptitle(f'Archaeological Site: {site_id} ({scale_name.title()} Scale - {buffer_meters}m radius)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save high-resolution visualization for field use
        output_path = self.output_dir / f"{site_id}_{scale_name}_scale.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   [SAVE] {scale_name.title()} visualization saved: {output_path}")
        return output_path
        
    def _calculate_plot_extent(self, sentinel_data, center_lat, center_lon, buffer_meters):
        """
        Calculate plot extent for Sentinel-2 data visualization.
        
        Args:
            sentinel_data (dict): Sentinel-2 data with bounds information
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            buffer_meters (float): Buffer distance in meters
            
        Returns:
            list: Plot extent [left, right, bottom, top] in degrees
        """
        
        buffer_deg = buffer_meters / 111320  # Approximate conversion to degrees
        
        return [
            center_lon - buffer_deg,  # left
            center_lon + buffer_deg,  # right
            center_lat - buffer_deg,  # bottom
            center_lat + buffer_deg   # top
        ]
        
    def _calculate_fabdem_extent(self, fabdem_data, center_lat, center_lon, buffer_meters):
        """
        Calculate plot extent for FABDEM elevation data visualization.
        
        Args:
            fabdem_data (dict): FABDEM data with transform and CRS information
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            buffer_meters (float): Buffer distance in meters
            
        Returns:
            list: Plot extent [left, right, bottom, top] in degrees
        """
        
        transform = fabdem_data['transform']
        elevation = fabdem_data['elevation']
        
        # Calculate geographic bounds of the elevation subset
        height, width = elevation.shape
        
        left = transform.c
        top = transform.f
        right = left + width * transform.a
        bottom = top + height * transform.e
        
        # Convert to geographic coordinates if needed
        if str(fabdem_data['crs']) != 'EPSG:4326':
            from rasterio.warp import transform_bounds
            left, bottom, right, top = transform_bounds(
                fabdem_data['crs'], 'EPSG:4326',
                left, bottom, right, top
            )
        
        return [left, right, bottom, top]
        
    def _add_site_overlay(self, ax, site, extent):
        """
        Add archaeological site boundary overlay to visualization plot.
        
        Overlays site boundaries, markers, and confidence information on
        satellite or elevation imagery for clear site identification.
        
        Args:
            ax: Matplotlib axis object
            site: Archaeological site data
            extent: Plot extent for coordinate reference
        """
        
        site_lat = site.geometry.y
        site_lon = site.geometry.x
        pattern_type = site.get('pattern_type', 'unknown')
        
        # Add site center marker
        ax.plot(site_lon, site_lat, 'r*', markersize=15, markeredgecolor='white', 
               markeredgewidth=2, label='Site Center')
        
        # Add geometric shape overlay based on pattern type
        if pattern_type == 'circular' and 'radius_meters' in site:
            radius_m = site['radius_meters']
            radius_deg = radius_m / 111320
            
            circle = Circle((site_lon, site_lat), radius_deg, 
                          fill=False, color='red', linewidth=2, linestyle='--')
            ax.add_patch(circle)
            
        elif pattern_type == 'rectangular' and 'length_meters' in site:
            length_m = site['length_meters']
            width_m = site['width_meters']
            
            length_deg = length_m / 111320
            width_deg = width_m / 111320
            
            # Create rectangle overlay (simplified without rotation)
            rect = Rectangle((site_lon - length_deg/2, site_lat - width_deg/2),
                           length_deg, width_deg,
                           fill=False, color='red', linewidth=2, linestyle='--')
            ax.add_patch(rect)
        
        # Add confidence assessment text
        confidence = site.get('fabdem_confidence', site.get('ndvi_confidence', 'Unknown'))
        ax.text(site_lon, site_lat - 0.0005, f'Conf: {confidence}', 
               ha='center', va='top', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
    def _add_scale_bar(self, ax, extent, buffer_meters):
        """
        Add scale bar to visualization for distance reference.
        
        Args:
            ax: Matplotlib axis object
            extent: Plot extent coordinates
            buffer_meters: Buffer distance for scale calculation
        """
        
        # Calculate scale bar length (1/4 of image width)
        scale_length_m = buffer_meters / 2
        scale_length_deg = scale_length_m / 111320
        
        # Position at bottom right corner
        x_pos = extent[1] - 0.1 * (extent[1] - extent[0])
        y_pos = extent[2] + 0.1 * (extent[3] - extent[2])
        
        # Draw scale bar line and label
        ax.plot([x_pos - scale_length_deg, x_pos], [y_pos, y_pos], 
               'k-', linewidth=3)
        ax.text(x_pos - scale_length_deg/2, y_pos + 0.02 * (extent[3] - extent[2]), 
               f'{scale_length_m:.0f}m', 
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
    def create_overview_map(self):
        """
        Create overview map showing all archaeological sites with confidence levels.
        
        Generates comprehensive overview visualization showing spatial distribution
        of all detected sites with confidence-based color coding and study area context.
        
        Returns:
            Path or None: Path to saved overview map
        """
        
        self.log_step("OVERVIEW", "Creating overview map of all archaeological sites")
        
        if len(self.sites_gdf) == 0:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Define confidence-based color scheme
        confidence_colors = {
            'VERY_HIGH': '#8B0000',  # Dark red
            'HIGH': '#FF4500',       # Orange red
            'MEDIUM': '#FFD700',     # Gold
            'LOW': '#C0C0C0'         # Silver
        }
        
        # Plot sites with confidence-based coloring
        for _, site in self.sites_gdf.iterrows():
            confidence = site.get('fabdem_confidence', site.get('priority', 'LOW'))
            color = confidence_colors.get(confidence, '#808080')
            
            ax.scatter(site.geometry.x, site.geometry.y, 
                      c=color, s=100, alpha=0.8, edgecolors='black')
            
            # Add site labels for identification
            ax.annotate(site['site_id'], 
                       (site.geometry.x, site.geometry.y),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        # Create legend showing confidence levels and counts
        legend_elements = []
        for conf, color in confidence_colors.items():
            count = len(self.sites_gdf[self.sites_gdf.get('fabdem_confidence', 
                                                         self.sites_gdf.get('priority', 'LOW')) == conf])
            if count > 0:
                legend_elements.append(plt.scatter([], [], c=color, s=100, 
                                                 label=f'{conf} ({count})'))
        
        ax.legend(handles=legend_elements, title='Confidence Level')
        
        # Configure map appearance
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Archaeological Sites Overview\n{len(self.sites_gdf)} sites detected')
        ax.grid(True, alpha=0.3)
        
        # Add study area bounds if available
        if 'study_area' in self.config:
            bounds = self.config['study_area']['bounds']
            rect = Rectangle((bounds['min_lon'], bounds['min_lat']),
                           bounds['max_lon'] - bounds['min_lon'],
                           bounds['max_lat'] - bounds['min_lat'],
                           fill=False, color='blue', linewidth=2, linestyle=':')
            ax.add_patch(rect)
            ax.text(bounds['min_lon'], bounds['max_lat'], 'Study Area',
                   fontsize=10, color='blue', ha='left', va='bottom')
        
        plt.tight_layout()
        
        # Save overview map
        overview_path = self.output_dir / 'archaeological_sites_overview.png'
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   [SAVE] Overview map saved: {overview_path}")
        return overview_path
        
    def visualize_all_sites(self):
        """
        Create comprehensive visualizations for all sites at all scales.
        
        Generates the complete visualization suite including overview map
        and individual site visualizations at detail, primary, and context scales.
        """
        
        self.log_step("VISUALIZE", f"Creating visualizations for {len(self.sites_gdf)} sites")
        
        # Create overview map showing all sites
        self.create_overview_map()
        
        # Create individual site visualizations at multiple scales
        for scale_name in ['detail', 'primary', 'context']:
            print(f"\n   [SCALE] Processing {scale_name} scale ({self.scales[scale_name]}m radius)")
            
            for idx, site in self.sites_gdf.iterrows():
                try:
                    self.create_site_visualization(site, scale_name)
                except Exception as e:
                    print(f"   [ERROR] Failed to create {scale_name} visualization for {site['site_id']}: {e}")
                    continue
        
        # Create comprehensive summary of all visualizations
        self.create_visualization_summary()
        
    def create_visualization_summary(self):
        """
        Create comprehensive summary of all visualization products.
        
        Generates detailed summary documenting all visualization files,
        scales, data sources, and technical specifications for reference.
        """
        
        summary_text = f"Archaeological Site Visualizations Summary\n"
        summary_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary_text += f"=" * 60 + "\n\n"
        
        summary_text += f"Total sites visualized: {len(self.sites_gdf)}\n"
        summary_text += f"Visualization scales:\n"
        summary_text += f"  - Detail: {self.scales['detail']}m radius (maximum detail)\n"
        summary_text += f"  - Primary: {self.scales['primary']}m radius (main validation)\n"
        summary_text += f"  - Context: {self.scales['context']}m radius (landscape context)\n\n"
        
        summary_text += f"Data sources:\n"
        summary_text += f"  - Sentinel-2: RGB composites and NDVI\n"
        summary_text += f"  - FABDEM: Bare-earth elevation with contours\n\n"
        
        summary_text += f"Files generated:\n"
        summary_text += f"  - archaeological_sites_overview.png\n"
        
        for _, site in self.sites_gdf.iterrows():
            site_id = site['site_id']
            for scale in ['detail', 'primary', 'context']:
                summary_text += f"  - {site_id}_{scale}_scale.png\n"
        
        summary_text += f"\nVisualization features:\n"
        summary_text += f"  - Site boundaries overlaid on satellite/elevation data\n"
        summary_text += f"  - Contour lines (1-2m intervals) on FABDEM data\n"
        summary_text += f"  - Scale bars and coordinate information\n"
        summary_text += f"  - Confidence levels and geometric measurements\n"
        summary_text += f"  - High-resolution outputs (300 DPI) for field use\n"
        
        summary_path = self.output_dir / 'visualization_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"\n[SUMMARY] Visualization summary saved: {summary_path}")
        
    def run_visualization(self):
        """
        Execute complete visualization workflow for all archaeological sites.
        
        Main entry point that orchestrates the entire visualization process
        from data loading through final output generation.
        
        Returns:
            bool: True if visualization completed successfully
            
        Raises:
            Exception: If critical errors occur during visualization
        """
        
        print("Archaeological Site Visualization")
        print("=" * 50)
        
        try:
            # Load archaeological sites and elevation data
            self.load_archaeological_sites()
            self.load_fabdem_data()
            
            # Create comprehensive visualization suite
            self.visualize_all_sites()
            
            print(f"\n[SUCCESS] All visualizations complete!")
            print(f"   [OUTPUT] Files saved to: {self.output_dir}")
            print(f"   [SCALES] Detail ({self.scales['detail']}m), Primary ({self.scales['primary']}m), Context ({self.scales['context']}m)")
            print(f"   [FEATURES] RGB, NDVI, and elevation with contours")
            print(f"   [READY] Visualizations ready for archaeological validation")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")
            raise

def main():
    """
    Main execution function for archaeological site visualization.
    
    Creates visualizer instance and executes complete visualization workflow.
    """
    visualizer = ArchaeologicalSiteVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main()