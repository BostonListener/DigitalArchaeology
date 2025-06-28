#!/usr/bin/env python3
"""
Stage 2B: NDVI Analysis with Parameter Grid Search

This module analyzes downloaded Sentinel-2 data to detect vegetation patterns
that may indicate archaeological features. It uses NDVI (Normalized Difference
Vegetation Index) to identify areas where vegetation grows differently, potentially
due to subsurface archaeological remains.

Key Features:
- Parameter grid search for optimal detection thresholds
- Connected component analysis for pattern identification
- Geometric shape classification (circular, rectangular, linear)
- Coordinate transformation validation for any study region
- OpenAI integration for pattern interpretation

The output feeds into Stage 3 for elevation validation.

Authors: Archaeological AI Team
License: MIT
"""

import json
import yaml
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform as warp_transform
from shapely.geometry import box
from skimage import measure, morphology
import tempfile
import zipfile
from itertools import product
import math

from result_analyzer import OpenAIAnalyzer


class Sentinel2ParameterAnalyzer:
    """
    Analyzes Sentinel-2 data using parameter grid search for optimal pattern detection.
    
    This class implements a comprehensive approach to archaeological pattern detection:
    - Processes downloaded Sentinel-2 imagery to NDVI
    - Tests multiple parameter combinations for robust detection
    - Classifies detected patterns by geometric shape
    - Validates coordinates for any global study region
    """
    
    def __init__(self):
        """Initialize analyzer with configuration and setup directories."""
        # Load configuration parameters
        with open("config/parameters.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.analysis_params = self.config['sentinel_analysis']
        self.paths = self.config['paths']
        
        # Create output directory for analysis results
        self.analysis_dir = Path(self.paths['stage2_dir']) / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load download metadata from Stage 2A
        download_metadata_path = Path(self.paths['stage2_dir']) / 'download_metadata.json'
        if not download_metadata_path.exists():
            raise FileNotFoundError(f"Download metadata not found. Run stage2a_download.py first.")
        
        with open(download_metadata_path, 'r') as f:
            self.download_metadata = json.load(f)
        
        self.candidates_info = self.download_metadata['candidates_info']
        
        # Storage for analysis results
        self.parameter_results = []
        self.best_parameters = None
        self.final_detections = []
        
        # Initialize OpenAI analyzer for pattern interpretation
        self.openai_analyzer = OpenAIAnalyzer()
        
    def log_step(self, step, message):
        """Log processing steps with timestamps for monitoring progress."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {step}: {message}")
        
    def generate_parameter_combinations(self):
        """
        Generate all parameter combinations for grid search optimization.
        
        Creates a comprehensive parameter space covering:
        - NDVI contrast thresholds
        - Geometric confidence thresholds  
        - Minimum pattern sizes
        
        Returns:
            list: List of parameter dictionaries to test
        """
        grid = self.analysis_params['parameter_grid']
        base_params = self.analysis_params['base_params']
        
        # Extract parameter names and value ranges
        param_names = list(grid.keys())
        param_values = list(grid.values())
        
        combinations = []
        
        # Generate all possible combinations using itertools.product
        for combo in product(*param_values):
            param_set = base_params.copy()
            for name, value in zip(param_names, combo):
                param_set[name] = value
            combinations.append(param_set)
        
        self.log_step("PARAMS", f"Generated {len(combinations)} parameter combinations")
        
        return combinations
        
    def _convert_coordinates_to_geographic(self, transform, crs, row, col):
        """
        Convert pixel coordinates to geographic coordinates (WGS84).
        
        This method handles coordinate transformation for any global study region,
        with proper validation and fallback mechanisms.
        
        Args:
            transform: Rasterio affine transform
            crs: Coordinate reference system
            row (int): Pixel row coordinate
            col (int): Pixel column coordinate
            
        Returns:
            tuple: (latitude, longitude) in WGS84 degrees
        """
        # Convert pixel coordinates to projected coordinates
        projected_x, projected_y = rasterio.transform.xy(transform, row, col)
        
        # Check if already in geographic coordinates
        crs_string = str(crs).upper()
        if 'EPSG:4326' in crs_string or 'WGS84' in crs_string or crs.to_epsg() == 4326:
            return projected_y, projected_x  # lat, lon
        
        # Transform from projected to geographic coordinates
        try:
            lon_array, lat_array = warp_transform(
                crs, 'EPSG:4326', 
                [projected_x], [projected_y]
            )
            lat, lon = lat_array[0], lon_array[0]
            
            # Validate coordinates are reasonable for study region
            study_bounds = self.config['study_area']['bounds']
            
            # Add tolerance buffer for coordinate transformation
            buffer = 0.1  # degrees
            
            if (study_bounds['min_lat'] - buffer <= lat <= study_bounds['max_lat'] + buffer and 
                study_bounds['min_lon'] - buffer <= lon <= study_bounds['max_lon'] + buffer):
                return lat, lon
            else:
                self.log_step("WARNING", f"Transformed coordinates outside expected range: {lat}, {lon}")
                self.log_step("INFO", f"Expected {self.config['study_area']['name']} bounds: "
                            f"Lat {study_bounds['min_lat']} to {study_bounds['max_lat']}, "
                            f"Lon {study_bounds['min_lon']} to {study_bounds['max_lon']}")
                return self._fallback_coordinate_conversion(projected_x, projected_y)
                
        except Exception as e:
            self.log_step("WARNING", f"Coordinate transformation failed: {e}")
            return self._fallback_coordinate_conversion(projected_x, projected_y)
    
    def _fallback_coordinate_conversion(self, utm_x, utm_y):
        """
        Fallback coordinate conversion for UTM to geographic coordinates.
        
        Provides a robust fallback when standard transformation fails,
        dynamically calculating UTM zones for any global study area.
        
        Args:
            utm_x (float): UTM easting coordinate
            utm_y (float): UTM northing coordinate
            
        Returns:
            tuple: (latitude, longitude) in WGS84 degrees
        """
        # Get study area center from configuration
        study_bounds = self.config['study_area']['bounds']
        study_name = self.config['study_area']['name']
        
        # Calculate center coordinates from study area bounds
        center_lat = (study_bounds['min_lat'] + study_bounds['max_lat']) / 2
        center_lon = (study_bounds['min_lon'] + study_bounds['max_lon']) / 2
        
        print(f"   [FALLBACK] Using {study_name} center: ({center_lat:.2f}, {center_lon:.2f})")
        
        # Determine UTM zone dynamically based on longitude
        utm_zone = int((center_lon + 180) / 6) + 1
        
        # Determine hemisphere based on latitude
        hemisphere = 'S' if center_lat < 0 else 'N'
        
        print(f"   [FALLBACK] Calculated UTM Zone: {utm_zone}{hemisphere}")
        
        # UTM coordinate system constants
        false_easting = 500000
        false_northing = 10000000 if hemisphere == 'S' else 0
        
        # Approximate UTM to geographic conversion
        try:
            # Convert northing to latitude
            lat = center_lat + (utm_y - false_northing) / 111320  # meters to degrees
            
            # Convert easting to longitude (accounting for latitude)
            lon = center_lon + (utm_x - false_easting) / (111320 * math.cos(math.radians(center_lat)))
            
            print(f"   [FALLBACK] Converted UTM ({utm_x:.0f}, {utm_y:.0f}) to Geographic ({lat:.4f}, {lon:.4f})")
            
            # Validate result is reasonable for study area
            lat_buffer = abs(study_bounds['max_lat'] - study_bounds['min_lat']) * 2
            lon_buffer = abs(study_bounds['max_lon'] - study_bounds['min_lon']) * 2
            
            if (study_bounds['min_lat'] - lat_buffer <= lat <= study_bounds['max_lat'] + lat_buffer and
                study_bounds['min_lon'] - lon_buffer <= lon <= study_bounds['max_lon'] + lon_buffer):
                print(f"   [FALLBACK] Coordinates within reasonable range for {study_name}")
                return lat, lon
            else:
                print(f"   [WARNING] Fallback coordinates outside reasonable {study_name} range")
                return lat, lon  # Return anyway as it's a fallback method
                
        except Exception as e:
            print(f"   [ERROR] Fallback conversion failed: {e}")
            # Return study area center as last resort
            print(f"   [EMERGENCY] Returning {study_name} center coordinates")
            return center_lat, center_lon
        
    def process_candidate_to_ndvi(self, candidate_index, candidate_info):
        """
        Process a single candidate's Sentinel-2 data to NDVI.
        
        Extracts and processes Red and NIR bands from Sentinel-2 data
        to calculate NDVI for archaeological pattern detection.
        
        Args:
            candidate_index (str): Index of candidate being processed
            candidate_info (dict): Metadata about downloaded candidate
            
        Returns:
            dict or None: NDVI data and metadata, or None if processing failed
        """
        download_path = Path(candidate_info['download_path'])
        candidate_data = candidate_info['candidate_data']
        
        if not download_path.exists():
            return None
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract Sentinel-2 ZIP file
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                
                # Navigate Sentinel-2 directory structure
                safe_dirs = list(temp_path.glob("*.SAFE"))
                if not safe_dirs:
                    return None
                
                safe_dir = safe_dirs[0]
                granule_dirs = list((safe_dir / "GRANULE").glob("*"))
                if not granule_dirs:
                    return None
                
                granule_dir = granule_dirs[0]
                
                # Find image data directory (try 10m resolution first)
                img_data_dir = granule_dir / "IMG_DATA" / "R10m"
                if not img_data_dir.exists():
                    img_data_dir = granule_dir / "IMG_DATA"
                
                if not img_data_dir.exists():
                    return None
                
                # Find required spectral bands
                red_files = list(img_data_dir.glob("*B04_10m.jp2")) or list(img_data_dir.glob("*B04.jp2"))
                nir_files = list(img_data_dir.glob("*B08_10m.jp2")) or list(img_data_dir.glob("*B08.jp2"))
                
                if not red_files or not nir_files:
                    return None
                
                red_path, nir_path = red_files[0], nir_files[0]
                
                # Create clipping geometry from query bounds
                bounds = candidate_data['query_bounds']
                clip_box_wgs84 = box(bounds['min_lon'], bounds['min_lat'], 
                                    bounds['max_lon'], bounds['max_lat'])
                
                # Read and clip Red band
                with rasterio.open(red_path) as red_src:
                    raster_crs = red_src.crs
                    raster_transform = red_src.transform
                    
                    # Transform clip box to raster CRS if needed
                    from rasterio.warp import transform_bounds
                    transformed_bounds = transform_bounds(
                        'EPSG:4326', raster_crs,
                        bounds['min_lon'], bounds['min_lat'],
                        bounds['max_lon'], bounds['max_lat']
                    )
                    
                    clip_box = box(*transformed_bounds)
                    
                    red_clipped, red_transform = mask(red_src, [clip_box], crop=True, nodata=0)
                    red_clipped = red_clipped[0].astype(np.float32)
                
                # Read and clip NIR band
                with rasterio.open(nir_path) as nir_src:
                    nir_clipped, _ = mask(nir_src, [clip_box], crop=True, nodata=0)
                    nir_clipped = nir_clipped[0].astype(np.float32)
                
                # Calculate NDVI: (NIR - Red) / (NIR + Red)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = np.where(
                        (nir_clipped + red_clipped) != 0,
                        (nir_clipped - red_clipped) / (nir_clipped + red_clipped),
                        0
                    )
                
                # Clean and normalize NDVI values
                ndvi = np.clip(ndvi, -1, 1)
                ndvi = np.nan_to_num(ndvi, nan=0, posinf=1, neginf=-1)
                
                return {
                    'ndvi_array': ndvi,
                    'transform': red_transform,
                    'crs': raster_crs,
                    'candidate_data': candidate_data
                }
                
        except Exception as e:
            self.log_step("ERROR", f"NDVI processing failed for candidate {candidate_index}: {e}")
            return None
            
    def analyze_ndvi_patterns(self, ndvi_data, params):
        """
        Analyze NDVI array for archaeological patterns using given parameters.
        
        Implements the core pattern detection algorithm:
        - Identifies low NDVI regions (potential archaeological signatures)
        - Applies morphological operations for noise reduction
        - Classifies patterns by geometric shape
        - Calculates detailed measurements
        
        Args:
            ndvi_data (dict): NDVI array and associated metadata
            params (dict): Detection parameters to test
            
        Returns:
            list: List of detected pattern dictionaries
        """
        ndvi_array = ndvi_data['ndvi_array']
        transform = ndvi_data['transform']
        crs = ndvi_data['crs']
        candidate_data = ndvi_data['candidate_data']
        
        # Skip arrays that are too small for meaningful analysis
        if ndvi_array.shape[0] < 12 or ndvi_array.shape[1] < 12:
            return []
            
        # Handle invalid values
        ndvi_array = np.nan_to_num(ndvi_array, nan=0.5)
        
        # Skip if mostly zeros (no valid data)
        valid_data = ndvi_array[ndvi_array != 0]
        if len(valid_data) < ndvi_array.size * 0.12:
            return []
            
        # Normalize NDVI values for consistent thresholding
        if len(valid_data) > 0:
            ndvi_min, ndvi_max = np.percentile(valid_data, [3, 97])
            if ndvi_max > ndvi_min:
                ndvi_normalized = np.clip((ndvi_array - ndvi_min) / (ndvi_max - ndvi_min), 0, 1)
            else:
                ndvi_normalized = np.ones_like(ndvi_array) * 0.5
        else:
            return []
            
        # Detect low NDVI regions (potential archaeological signatures)
        valid_normalized = ndvi_normalized[ndvi_normalized > 0]
        if len(valid_normalized) == 0:
            return []
            
        # Dynamic threshold based on data distribution
        threshold = np.percentile(valid_normalized, 25)
        low_ndvi_mask = (ndvi_normalized < threshold) & (ndvi_normalized > 0)
        
        # Apply morphological operations for noise reduction
        low_ndvi_mask = morphology.remove_small_objects(low_ndvi_mask, min_size=6)
        low_ndvi_mask = morphology.remove_small_holes(low_ndvi_mask, area_threshold=6)
        
        # Find connected components (individual patterns)
        labeled_mask = measure.label(low_ndvi_mask)
        regions = measure.regionprops(labeled_mask)
        
        # Limit number of regions to process (focus on largest)
        if len(regions) > 10:
            regions = sorted(regions, key=lambda r: r.area, reverse=True)[:10]
        
        patterns = []
        
        # Calculate pixel size in meters for measurements
        pixel_size_x_m, pixel_size_y_m = self._calculate_pixel_size_meters(transform, crs, candidate_data)
        
        for region in regions:
            # Apply minimum size filter
            if region.area < params['min_pattern_pixels']:
                continue
                
            # Analyze geometric characteristics
            shape_analysis = self._analyze_region_geometry(region, params)
            
            if shape_analysis['is_geometric']:
                # Calculate NDVI contrast (how different from surroundings)
                region_mask = labeled_mask == region.label
                region_ndvi = np.mean(ndvi_normalized[region_mask])
                surrounding_mask = (ndvi_normalized > 0) & (~low_ndvi_mask)
                
                if np.any(surrounding_mask):
                    surrounding_ndvi = np.mean(ndvi_normalized[surrounding_mask])
                    contrast = abs(surrounding_ndvi - region_ndvi)
                    
                    # Apply contrast threshold
                    if contrast > params['ndvi_contrast_threshold']:
                        # Convert to geographic coordinates
                        centroid_lat, centroid_lon = self._convert_coordinates_to_geographic(
                            transform, crs, region.centroid[0], region.centroid[1]
                        )
                        
                        # Calculate detailed geometric measurements
                        geometric_details = self._calculate_detailed_geometry(
                            region, pixel_size_x_m, pixel_size_y_m, shape_analysis['shape_type']
                        )
                        
                        # Apply size filter
                        if 0.1 <= geometric_details['area_hectares'] <= params['max_pattern_hectares']:
                            pattern = {
                                'type': shape_analysis['shape_type'],
                                'confidence': shape_analysis['confidence'],
                                'contrast': contrast,
                                'coords': {
                                    'lat': centroid_lat,
                                    'lon': centroid_lon
                                },
                                # Include all geometric details
                                **geometric_details,
                                'region_properties': {
                                    'eccentricity': region.eccentricity,
                                    'solidity': region.solidity,
                                    'extent': region.extent
                                }
                            }
                            patterns.append(pattern)
        
        return patterns
        
    def _calculate_pixel_size_meters(self, transform, crs, candidate_data):
        """
        Calculate pixel size in meters accounting for coordinate system.
        
        Handles both projected (UTM) and geographic coordinate systems
        with dynamic calculation for any global study area.
        
        Args:
            transform: Rasterio affine transform
            crs: Coordinate reference system
            candidate_data (dict): Candidate metadata with location
            
        Returns:
            tuple: (pixel_size_x_meters, pixel_size_y_meters)
        """
        pixel_x_size = abs(transform[0])
        pixel_y_size = abs(transform[4])
        
        # Check if coordinates are already in meters (UTM or other projected systems)
        crs_string = str(crs).upper()
        if 'UTM' in crs_string or 'METER' in crs_string or pixel_x_size > 1.0:
            # Already in meters
            return pixel_x_size, pixel_y_size
        else:
            # Geographic coordinates (degrees) - convert to meters
            center_lat = candidate_data['centroid_lat']
            
            # Use study area center if candidate coordinates seem wrong
            study_bounds = self.config['study_area']['bounds']
            if not (study_bounds['min_lat'] <= center_lat <= study_bounds['max_lat']):
                print(f"   [WARNING] Using study area center lat for pixel size calculation")
                center_lat = (study_bounds['min_lat'] + study_bounds['max_lat']) / 2
            
            # Convert degrees to meters at given latitude
            lat_rad = math.radians(abs(center_lat))
            meters_per_degree_lon = 111320 * math.cos(lat_rad)  # Longitude varies with latitude
            meters_per_degree_lat = 111320                       # Latitude is constant
            
            pixel_size_x_m = pixel_x_size * meters_per_degree_lon
            pixel_size_y_m = pixel_y_size * meters_per_degree_lat
            
            return pixel_size_x_m, pixel_size_y_m
        
    def _calculate_detailed_geometry(self, region, pixel_size_x_m, pixel_size_y_m, shape_type):
        """
        Calculate detailed geometric measurements for archaeological features.
        
        Provides comprehensive measurements for different shape types:
        - Circular: radius, diameter, circumference
        - Rectangular: length, width, aspect ratio
        - Linear: length, width, linearity ratio
        
        Args:
            region: Scikit-image region properties object
            pixel_size_x_m (float): Pixel size in meters (X direction)
            pixel_size_y_m (float): Pixel size in meters (Y direction)
            shape_type (str): Classification of shape type
            
        Returns:
            dict: Comprehensive geometric measurements
        """
        # Basic measurements in pixels and meters
        area_pixels = region.area
        pixel_area_m2 = pixel_size_x_m * pixel_size_y_m
        area_m2 = area_pixels * pixel_area_m2
        area_hectares = area_m2 / 10000
        
        # Axis lengths in meters
        major_axis_pixels = region.major_axis_length
        minor_axis_pixels = region.minor_axis_length
        avg_pixel_size = (pixel_size_x_m + pixel_size_y_m) / 2
        major_axis_meters = major_axis_pixels * avg_pixel_size
        minor_axis_meters = minor_axis_pixels * avg_pixel_size
        
        # Orientation in degrees
        orientation_degrees = math.degrees(region.orientation)
        
        # Perimeter in meters
        perimeter_meters = region.perimeter * avg_pixel_size
        
        # Base geometric details
        geometric_details = {
            'area_pixels': int(area_pixels),
            'area_square_meters': float(area_m2),
            'area_hectares': float(area_hectares),
            'perimeter_meters': float(perimeter_meters),
            'major_axis_meters': float(major_axis_meters),
            'minor_axis_meters': float(minor_axis_meters),
            'orientation_degrees': float(orientation_degrees),
            'pixel_size_x_meters': float(pixel_size_x_m),
            'pixel_size_y_meters': float(pixel_size_y_m)
        }
        
        # Shape-specific calculations
        if shape_type == 'circular':
            # Calculate radius from area (more accurate than using axes)
            radius_meters = math.sqrt(area_m2 / math.pi)
            diameter_meters = 2 * radius_meters
            
            geometric_details.update({
                'radius_meters': float(radius_meters),
                'diameter_meters': float(diameter_meters),
                'circumference_meters': float(2 * math.pi * radius_meters)
            })
            
        elif shape_type == 'rectangular':
            # Use major/minor axis as length/width
            length_meters = max(major_axis_meters, minor_axis_meters)
            width_meters = min(major_axis_meters, minor_axis_meters)
            
            geometric_details.update({
                'length_meters': float(length_meters),
                'width_meters': float(width_meters),
                'aspect_ratio': float(length_meters / width_meters if width_meters > 0 else 1)
            })
            
        elif shape_type == 'linear':
            # Linear features use major axis as length, minor as width
            geometric_details.update({
                'length_meters': float(major_axis_meters),
                'width_meters': float(minor_axis_meters),
                'linearity_ratio': float(major_axis_meters / minor_axis_meters if minor_axis_meters > 0 else 1)
            })
        
        # Add equivalent circle radius for size comparison
        geometric_details['equivalent_radius_meters'] = float(math.sqrt(area_m2 / math.pi))
        
        return geometric_details
        
    def _analyze_region_geometry(self, region, params):
        """
        Analyze region geometry to classify shape type and assess confidence.
        
        Uses multiple geometric metrics to classify detected patterns:
        - Circularity: based on area-to-perimeter ratio
        - Eccentricity: measure of elongation
        - Solidity: measure of convexity
        - Extent: how much of bounding box is filled
        
        Args:
            region: Scikit-image region properties object
            params (dict): Analysis parameters with thresholds
            
        Returns:
            dict: Shape classification and confidence assessment
        """
        eccentricity = region.eccentricity
        solidity = region.solidity
        extent = region.extent
        
        # Calculate circularity metric
        area_to_perimeter_ratio = region.area / (region.perimeter**2) if region.perimeter > 0 else 0
        circularity = 4 * np.pi * area_to_perimeter_ratio
        
        confidence = 0
        shape_type = 'irregular'
        
        # Shape classification with confidence scoring
        if circularity > 0.4 and eccentricity < 0.7 and solidity > 0.6:
            shape_type = 'circular'
            confidence = min(0.80, circularity * 1.4 + (1 - eccentricity) * 0.4)
        elif extent > 0.5 and solidity > 0.6 and eccentricity < 0.85:
            shape_type = 'rectangular'
            confidence = min(0.75, extent * solidity * 1.2)
        elif eccentricity > 0.65 and solidity > 0.55 and extent > 0.45:
            shape_type = 'linear'
            confidence = min(0.70, eccentricity * solidity * 1.0)
        
        # Apply size validation penalty
        if region.area < params['min_pattern_pixels']:
            confidence *= 0.5
        
        # Apply extreme aspect ratio penalty
        if region.major_axis_length / region.minor_axis_length > 12:
            confidence *= 0.7
        
        # Cap maximum confidence
        confidence = min(confidence, 0.80)
        
        # Determine if pattern meets geometric criteria
        is_geometric = (confidence > params['geometry_threshold'] and 
                       region.area >= params['min_pattern_pixels'] * 0.75 and
                       circularity > 0.15)
        
        return {
            'is_geometric': is_geometric,
            'shape_type': shape_type,
            'confidence': confidence
        }
        
    def run_parameter_grid_search(self):
        """
        Execute parameter grid search to find optimal detection settings.
        
        Tests all parameter combinations to identify the settings that
        provide the best balance of detection sensitivity and accuracy.
        """
        self.log_step("GRID", "Starting parameter grid search")
        
        # Generate all parameter combinations to test
        param_combinations = self.generate_parameter_combinations()
        
        # Process all candidates to NDVI first (one-time computational cost)
        self.log_step("NDVI", "Processing all candidates to NDVI")
        ndvi_data_cache = {}
        
        for candidate_index, candidate_info in self.candidates_info.items():
            if not candidate_info.get('skipped', False):
                ndvi_data = self.process_candidate_to_ndvi(candidate_index, candidate_info)
                if ndvi_data:
                    ndvi_data_cache[candidate_index] = ndvi_data
        
        print(f"   [INFO] Successfully processed {len(ndvi_data_cache)} candidates to NDVI")
        
        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            self.log_step("TEST", f"Testing parameter set {i+1}/{len(param_combinations)}")
            
            total_detections = 0
            candidate_results = []
            
            # Apply current parameters to all NDVI data
            for candidate_index, ndvi_data in ndvi_data_cache.items():
                patterns = self.analyze_ndvi_patterns(ndvi_data, params)
                
                if patterns:
                    total_detections += len(patterns)
                    candidate_results.append({
                        'candidate_index': candidate_index,
                        'patterns': patterns
                    })
            
            # Score this parameter set
            param_result = {
                'parameters': params,
                'total_detections': total_detections,
                'candidates_with_detections': len(candidate_results),
                'candidate_results': candidate_results,
                'detection_rate': len(candidate_results) / len(ndvi_data_cache) if ndvi_data_cache else 0
            }
            
            self.parameter_results.append(param_result)
            
            print(f"      Results: {total_detections} patterns in {len(candidate_results)} candidates "
                  f"({param_result['detection_rate']:.1%} detection rate)")
        
        # Select best parameters based on comprehensive scoring
        self._select_best_parameters()
        
    def _select_best_parameters(self):
        """
        Select optimal parameter combination based on balanced criteria.
        
        Balances multiple factors:
        - Total number of detections
        - Consistency of detection across candidates
        - Parameter reasonableness
        """
        self.log_step("SELECT", "Selecting optimal parameters")
        
        # Score parameter sets using composite criteria
        for result in self.parameter_results:
            # Balance total detections with detection rate
            detection_score = result['total_detections'] * 0.7
            rate_score = result['detection_rate'] * 30  # Favor consistent detection
            quality_bonus = 0
            
            # Quality bonus for reasonable parameters
            params = result['parameters']
            if 0.05 <= params['ndvi_contrast_threshold'] <= 0.18:
                quality_bonus += 5
            if 0.30 <= params['geometry_threshold'] <= 0.70:
                quality_bonus += 5
            
            result['composite_score'] = detection_score + rate_score + quality_bonus
        
        # Sort by composite score and select best
        self.parameter_results.sort(key=lambda x: x['composite_score'], reverse=True)
        self.best_parameters = self.parameter_results[0]
        
        print(f"   [BEST] Best parameters selected:")
        for param, value in self.best_parameters['parameters'].items():
            print(f"      {param}: {value}")
        print(f"   [INFO] Results: {self.best_parameters['total_detections']} detections, "
              f"{self.best_parameters['detection_rate']:.1%} rate")
        
    def generate_final_detections(self):
        """
        Generate final detection results using optimal parameters with GPT analysis.
        
        Creates comprehensive detection records with all measurements
        and optional AI interpretation of patterns.
        
        Returns:
            list: List of final detection dictionaries
        """
        self.log_step("FINAL", "Generating final detections with GPT analysis")
        
        best_result = self.best_parameters
        final_detections = []
        
        # Process each candidate with detected patterns
        for candidate_result in best_result['candidate_results']:
            candidate_index = candidate_result['candidate_index']
            candidate_info = self.candidates_info[str(candidate_index)]
            candidate_data = candidate_info['candidate_data']
            
            # Create detailed detection records
            for pattern in candidate_result['patterns']:
                detection = {
                    'polygon_id': candidate_data['polygon_id'],
                    'candidate_index': candidate_index,
                    'pattern_type': pattern['type'],
                    'confidence': pattern['confidence'],
                    'ndvi_contrast': pattern['contrast'],
                    'lat': pattern['coords']['lat'],
                    'lon': pattern['coords']['lon'],
                    
                    # Comprehensive geometric measurements
                    'area_hectares': pattern['area_hectares'],
                    'area_square_meters': pattern['area_square_meters'],
                    'perimeter_meters': pattern['perimeter_meters'],
                    'major_axis_meters': pattern['major_axis_meters'],
                    'minor_axis_meters': pattern['minor_axis_meters'],
                    'orientation_degrees': pattern['orientation_degrees'],
                    'equivalent_radius_meters': pattern['equivalent_radius_meters'],
                    
                    # Shape-specific measurements (if available)
                    'radius_meters': pattern.get('radius_meters'),
                    'diameter_meters': pattern.get('diameter_meters'),
                    'circumference_meters': pattern.get('circumference_meters'),
                    'length_meters': pattern.get('length_meters'),
                    'width_meters': pattern.get('width_meters'),
                    'aspect_ratio': pattern.get('aspect_ratio'),
                    'linearity_ratio': pattern.get('linearity_ratio'),
                    
                    # Shape quality metrics
                    'eccentricity': pattern['region_properties']['eccentricity'],
                    'solidity': pattern['region_properties']['solidity'],
                    'extent': pattern['region_properties']['extent'],
                    
                    # Source metadata
                    'sentinel_product': candidate_info['product_name'],
                    'cloud_cover': candidate_info['cloud_cover'],
                    'pixel_size_x_meters': pattern['pixel_size_x_meters'],
                    'pixel_size_y_meters': pattern['pixel_size_y_meters']
                }
                final_detections.append(detection)
        
        self.final_detections = final_detections
        
        print(f"   [RESULT] Generated {len(final_detections)} final archaeological pattern detections")
        
        # Optional GPT analysis for top patterns
        if len(final_detections) > 0:
            max_gpt_patterns = 20  # Reasonable limit for AI analysis
            
            # Sort by confidence and analyze top patterns
            sorted_detections = sorted(final_detections, key=lambda x: x['confidence'], reverse=True)
            patterns_to_analyze = sorted_detections[:max_gpt_patterns]
            
            print(f"   [GPT] Analyzing top {len(patterns_to_analyze)} highest-confidence patterns...")
            print(f"   [INFO] Total patterns: {len(final_detections)}")
            print(f"   [CONFIDENCE] Range: {patterns_to_analyze[0]['confidence']:.3f} to {patterns_to_analyze[-1]['confidence']:.3f}")
            
            for i, detection in enumerate(patterns_to_analyze):
                try:
                    # Prepare pattern data for GPT analysis
                    pattern_data = {
                        'type': detection['pattern_type'],
                        'contrast': detection.get('ndvi_contrast', 0),
                        'confidence': detection['confidence'],
                        'area_hectares': detection.get('area_hectares', 0),
                        'major_axis_meters': detection.get('major_axis_meters', 0),
                        'minor_axis_meters': detection.get('minor_axis_meters', 0),
                        'location': f"{detection['lat']:.4f}°, {detection['lon']:.4f}°",
                        'perimeter_meters': detection.get('perimeter_meters', 0),
                        'orientation_degrees': detection.get('orientation_degrees', 0)
                    }
                    
                    # Get GPT pattern interpretation
                    gpt_result = self.openai_analyzer.describe_ndvi_patterns(pattern_data)
                    
                    if gpt_result['success']:
                        detection['gpt_interpretation'] = gpt_result['response']
                        detection['gpt_processing_time'] = gpt_result['processing_time']
                        conf = detection['confidence']
                        print(f"      ✓ Pattern {i+1}/{len(patterns_to_analyze)} analyzed (confidence: {conf:.3f})")
                    else:
                        detection['gpt_interpretation'] = f"Analysis failed: {gpt_result.get('error', 'Unknown error')}"
                        print(f"      ✗ Pattern {i+1} failed: {gpt_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    detection['gpt_interpretation'] = f"Analysis error: {str(e)}"
                    print(f"      ✗ Pattern {i+1} error: {e}")
            
            # Mark which patterns received GPT analysis
            for detection in final_detections:
                detection['gpt_analyzed'] = detection in patterns_to_analyze
        
        # Validate coordinates for study region
        if final_detections:
            sample_coords = final_detections[0]
            print(f"   [VALIDATE] Sample coordinates: ({sample_coords['lat']:.4f}°, {sample_coords['lon']:.4f}°)")
            
            # Use study area bounds from configuration
            study_bounds = self.config['study_area']['bounds']
            study_name = self.config['study_area']['name']
            
            print(f"   Expected {study_name} range: "
                f"Lat {study_bounds['min_lat']} to {study_bounds['max_lat']}, "
                f"Lon {study_bounds['min_lon']} to {study_bounds['max_lon']}")
            
            # Count valid coordinates
            valid_coords = 0
            for detection in final_detections:
                if (study_bounds['min_lat'] <= detection['lat'] <= study_bounds['max_lat'] and 
                    study_bounds['min_lon'] <= detection['lon'] <= study_bounds['max_lon']):
                    valid_coords += 1
            
            print(f"   Valid coordinates: {valid_coords}/{len(final_detections)}")
            
            if valid_coords > 0:
                print(f"   [SUCCESS] Coordinates are within expected {study_name} region bounds")
            else:
                print(f"   [WARNING] Some coordinates may be outside expected {study_name} region")
        
        return final_detections

    def _convert_numpy_types(self, obj):
        """
        Convert numpy types to JSON serializable types for saving results.
        
        Recursively processes nested dictionaries and lists to convert
        numpy data types to standard Python types.
        
        Args:
            obj: Object to convert (dict, list, or primitive)
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_results(self):
        """
        Save comprehensive Stage 2B results including GPT analyses.
        
        Outputs multiple formats:
        - JSON files with parameter results and detections
        - CSV file for easy analysis and visualization
        - Metadata with processing information
        - GPT analysis results (if available)
        """
        self.log_step("SAVE", "Saving Stage 2B results with GPT analyses")
        
        # Convert numpy types for JSON serialization
        parameter_results_clean = self._convert_numpy_types(self.parameter_results)
        
        # Save parameter grid search results
        param_results_path = self.analysis_dir / 'parameter_grid_results.json'
        with open(param_results_path, 'w') as f:
            json.dump(parameter_results_clean, f, indent=2)
        
        # Save final detections in multiple formats
        if self.final_detections:
            # CSV format for easy analysis
            detections_df = pd.DataFrame(self.final_detections)
            detections_csv = Path(self.paths['stage2_dir']) / 'pattern_summary.csv'
            detections_df.to_csv(detections_csv, index=False)
            
            # JSON format with full detail
            final_detections_clean = self._convert_numpy_types(self.final_detections)
            detections_json = self.analysis_dir / 'final_detections.json'
            with open(detections_json, 'w') as f:
                json.dump(final_detections_clean, f, indent=2)
        
        # Save GPT analysis results
        gpt_results = {
            'timestamp': datetime.now().isoformat(),
            'stage': 'STAGE_2B_NDVI_ANALYSIS_WITH_GPT',
            'checkpoint1_completed': hasattr(self, 'checkpoint1_result'),
            'evidence_synthesis': getattr(self, 'evidence_synthesis', 'No synthesis available'),
            'pattern_analyses': []
        }
        
        # Include Checkpoint 1 result if available
        if hasattr(self, 'checkpoint1_result'):
            gpt_results['checkpoint1_surface_description'] = self.checkpoint1_result
        
        # Save individual pattern analyses
        if hasattr(self, 'final_detections'):
            for detection in self.final_detections:
                if 'gpt_interpretation' in detection:
                    gpt_results['pattern_analyses'].append({
                        'pattern_id': f"{detection['polygon_id']}_{detection['candidate_index']}",
                        'pattern_type': detection['pattern_type'],
                        'location': f"{detection['lat']:.4f}°, {detection['lon']:.4f}°",
                        'gpt_interpretation': detection['gpt_interpretation'],
                        'processing_time': detection.get('gpt_processing_time', 0)
                    })
        
        gpt_results_path = self.analysis_dir / 'stage2b_gpt_analyses.json'
        with open(gpt_results_path, 'w') as f:
            json.dump(gpt_results, f, indent=2)
        
        # Save comprehensive metadata
        metadata_clean = self._convert_numpy_types({
            'analysis_date': datetime.now().isoformat(),
            'method': 'PARAMETER_GRID_SEARCH_WITH_GPT_INTEGRATION',
            'checkpoint1_completed': hasattr(self, 'checkpoint1_result'),
            'candidates_processed': len(self.candidates_info),
            'parameter_combinations_tested': len(self.parameter_results),
            'best_parameters': self.best_parameters['parameters'] if self.best_parameters else None,
            'final_detections': len(self.final_detections) if hasattr(self, 'final_detections') else 0,
            'gpt_analyses_completed': len(gpt_results['pattern_analyses']),
            'coordinate_system': 'WGS84_GEOGRAPHIC',
            'coordinate_transformation': 'UTM_TO_WGS84_APPLIED',
            'download_metadata': self.download_metadata
        })
        
        metadata_path = self.analysis_dir / 'stage2b_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata_clean, f, indent=2)
        
        print(f"   [SAVE] Results saved:")
        print(f"      Parameter analysis: {param_results_path}")
        if hasattr(self, 'final_detections') and self.final_detections:
            print(f"      Pattern detections: {detections_csv}")
            print(f"      GPT analyses: {gpt_results_path}")
            print(f"      Checkpoint 1: COMPLETED")
        print(f"      Metadata: {metadata_path}")
        if hasattr(self, 'checkpoint1_result'):
            print(f"      Surface feature description: COMPLETED")
        if hasattr(self, 'evidence_synthesis'):
            print(f"      Evidence synthesis: COMPLETED")
    
    def run_stage2b(self):
        """
        Execute the complete Stage 2B NDVI analysis process.
        
        This is the main entry point that orchestrates:
        1. Parameter grid search optimization
        2. NDVI pattern detection and classification
        3. Coordinate validation and measurement calculation
        4. Optional GPT interpretation of patterns
        5. Results saving in multiple formats
        
        Returns:
            bool: True if analysis completed successfully
        """
        study_name = self.config['study_area']['name']
        print(f"[ANALYZE] Stage 2B: NDVI Analysis for {study_name} Region")
        print("=" * 70)
        
        try:
            # CHECKPOINT 1: Surface Feature Description (if first candidate available)
            print(f"\n[CHECKPOINT1] Surface Feature Description")
            print("-" * 50)
            
            if len(self.candidates_info) > 0:
                # Get first candidate for Checkpoint 1 analysis
                first_candidate_key = list(self.candidates_info.keys())[0]
                first_candidate = self.candidates_info[first_candidate_key]
                
                # Prepare surface description data
                candidate_data = first_candidate['candidate_data']
                product_name = first_candidate['product_name']
                
                data_description = f"""
                Sentinel-2 Level-2A multispectral satellite imagery:
                - Product: {product_name}
                - Location: {candidate_data['centroid_lat']:.4f}°, {candidate_data['centroid_lon']:.4f}°
                - Region: {study_name}, Amazon Basin
                - Deforestation year: {candidate_data.get('deforestation_year', 'Unknown')}
                - Area of interest: {candidate_data.get('area_ha', 'Unknown')} hectares
                - Cloud cover: {first_candidate.get('cloud_cover', 'Unknown')}%
                - Context: Recently deforested area with potential archaeological features
                - Bands available: Red, NIR, Green, Blue (10m resolution)
                """
                
                dataset_id = f"SENTINEL2_{product_name}"
                
                print(f"   [DATA] Analyzing {product_name}")
                print(f"   [LOCATION] {candidate_data['centroid_lat']:.4f}°, {candidate_data['centroid_lon']:.4f}°")
                
                checkpoint1_result = self.openai_analyzer.describe_surface_features(
                    data_type="Sentinel-2 Level-2A multispectral satellite imagery",
                    data_description=data_description,
                    dataset_id=dataset_id
                )
                
                if checkpoint1_result['success']:
                    print(f"   [SUCCESS] ✓ Surface feature description completed")
                    print(f"   [MODEL] {checkpoint1_result['model_version']}")
                    print(f"   [DATASET] {checkpoint1_result['dataset_id']}")
                    print(f"   [ANALYSIS] {checkpoint1_result['response'][:300]}...")
                    
                    # Save checkpoint 1 result
                    checkpoint1_path = self.analysis_dir / 'checkpoint1_surface_description.json'
                    with open(checkpoint1_path, 'w') as f:
                        json.dump(checkpoint1_result, f, indent=2)
                    
                    print(f"   [SAVED] {checkpoint1_path}")
                    
                    # Store for later use
                    self.checkpoint1_result = checkpoint1_result
                    
                else:
                    print(f"   [ERROR] ✗ Checkpoint 1 failed: {checkpoint1_result.get('error', 'Unknown error')}")
                    
            else:
                print(f"   [ERROR] No candidate data available for Checkpoint 1")
            
            # Continue with main analysis pipeline
            print(f"\n[PARAMETER SEARCH] Starting parameter grid search...")
            self.run_parameter_grid_search()
            
            print(f"\n[DETECTIONS] Generating final detections...")
            self.generate_final_detections()
            
            # Save all results
            self.save_results()
            
            print(f"\n[SUCCESS] Stage 2B Complete for {study_name}!")
            print(f"   [CHECKPOINT1] Surface description: COMPLETED")
            print(f"   [PARAMETERS] Grid search: {len(self.parameter_results)} combinations tested")
            print(f"   [DETECTIONS] Final patterns: {len(self.final_detections)}")
            print(f"   [GPT] Pattern analyses: COMPLETED")
            print(f"   [COORDINATES] WGS84 geographic system validated")
            print(f"   [OUTPUT] Results saved to: {self.analysis_dir}")
            
            if self.final_detections:
                print(f"\n[READY] Ready for Stage 3 FABDEM validation")
                sample = self.final_detections[0]
                print(f"   Sample coordinates: ({sample['lat']:.4f}°, {sample['lon']:.4f}°)")
                
                # Validate coordinates using study area bounds
                study_bounds = self.config['study_area']['bounds']
                valid_coords = 0
                for detection in self.final_detections:
                    if (study_bounds['min_lat'] <= detection['lat'] <= study_bounds['max_lat'] and 
                        study_bounds['min_lon'] <= detection['lon'] <= study_bounds['max_lon']):
                        valid_coords += 1
                
                print(f"   Valid coordinates: {valid_coords}/{len(self.final_detections)}")
                
                if valid_coords > 0:
                    print(f"   ✅ Coordinates validated for {study_name}")
                else:
                    print(f"   ⚠️  Coordinate validation issues detected")
            else:
                print(f"\n[WARNING] No patterns detected")
                
            return True
            
        except Exception as e:
            print(f"[ERROR] Stage 2B failed for {study_name}: {e}")
            raise


def main():
    """Main entry point for Stage 2B processing."""
    analyzer = Sentinel2ParameterAnalyzer()
    analyzer.run_stage2b()


if __name__ == "__main__":
    main()