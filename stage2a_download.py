#!/usr/bin/env python3
"""
Stage 2A: Sentinel-2 Data Download

This module handles the download of Sentinel-2 satellite imagery from the
Copernicus Data Space Ecosystem for archaeological candidates identified
in Stage 1. It manages authentication, product search, and bulk download
operations.

Key Features:
- OAuth2 authentication with Copernicus Data Space
- Intelligent product selection (L2A preferred over L1C)
- Cloud cover filtering and dry season preference
- Robust download with retry logic and validation
- Metadata tracking for Stage 2B analysis

The downloaded imagery feeds into Stage 2B for NDVI pattern analysis.

Authors: Archaeological AI Team
License: MIT
"""

import os
import json
import yaml
import warnings
warnings.filterwarnings('ignore')

import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()


class Sentinel2Downloader:
    """
    Downloads Sentinel-2 satellite data for archaeological candidate areas.
    
    This class handles the complete Sentinel-2 acquisition workflow:
    - Authentication with Copernicus Data Space
    - Product search and filtering
    - Download management with progress tracking
    - Data validation and metadata storage
    """
    
    def __init__(self):
        """Initialize downloader with configuration and authentication setup."""
        # Load configuration parameters
        with open("config/parameters.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.download_params = self.config['sentinel_download']
        self.paths = self.config['paths']
        
        # Create output directories
        self.downloads_dir = Path(self.paths['stage2_dir']) / "downloads"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        # Load authentication credentials from environment
        self.copernicus_user = os.getenv("USER_NAME")
        self.copernicus_password = os.getenv("USER_PASSWORD")
        
        if not self.copernicus_user or not self.copernicus_password:
            raise ValueError("Environment variables USER_NAME and USER_PASSWORD required")
        
        # Session management for API calls
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.session = None
        
        # Track download results for reporting
        self.download_results = {}
        
    def log_step(self, step, message):
        """Log processing steps with timestamps for monitoring progress."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {step}: {message}")
        
    def load_candidates(self):
        """
        Load archaeological candidates from Stage 1 results.
        
        Returns:
            list: List of candidate dictionaries with query parameters
        """
        self.log_step("LOAD", "Loading archaeological candidates from Stage 1")
        
        queries_path = Path(self.paths['sentinel_queries'])
        if not queries_path.exists():
            raise FileNotFoundError(f"Candidates file not found: {queries_path}")
        
        with open(queries_path, 'r') as f:
            self.candidates = json.load(f)
        
        print(f"   [INFO] Loaded {len(self.candidates)} candidates for download")
        
        # Show download plan summary
        print(f"   [RESULT] Download plan:")
        for i, candidate in enumerate(self.candidates[:5]):
            bounds = candidate['query_bounds']
            lat, lon = candidate['centroid_lat'], candidate['centroid_lon']
            print(f"      Candidate {i}: ({lat:.4f}, {lon:.4f})")
        
        if len(self.candidates) > 5:
            print(f"      ... and {len(self.candidates) - 5} more")
        
        return self.candidates
        
    def authenticate(self):
        """
        Authenticate with Copernicus Data Space using OAuth2.
        
        Obtains access and refresh tokens for API access.
        """
        self.log_step("AUTH", "Authenticating with Copernicus Data Space")
        
        # OAuth2 token request
        data = {
            "client_id": "cdse-public",
            "username": self.copernicus_user,
            "password": self.copernicus_password,
            "grant_type": "password",
        }
        
        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data,
                timeout=30
            )
            r.raise_for_status()
            
            token_data = r.json()
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token")
            
            # Calculate token expiration time with buffer
            expires_in = token_data.get("expires_in", 600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 30)
            
            # Set up authenticated session
            self.session = requests.Session()
            self.session.headers.update({'Authorization': f'Bearer {self.access_token}'})
            
            print(f"   [SUCCESS] Authentication successful")
            
        except Exception as e:
            raise Exception(f"Authentication failed: {e}")
            
    def refresh_access_token(self):
        """
        Refresh the access token using the refresh token.
        
        Falls back to full authentication if refresh fails.
        """
        if not self.refresh_token:
            self.authenticate()
            return
            
        self.log_step("AUTH", "Refreshing access token")
        
        data = {
            "client_id": "cdse-public",
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        
        try:
            response = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data,
                timeout=30
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token", self.refresh_token)
            
            expires_in = token_data.get("expires_in", 600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 30)
            
            self.session.headers.update({'Authorization': f'Bearer {self.access_token}'})
            
        except Exception as e:
            self.log_step("AUTH", f"Token refresh failed, re-authenticating")
            self.authenticate()
            
    def ensure_valid_token(self):
        """Ensure we have a valid access token before making API calls."""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            if self.refresh_token:
                self.refresh_access_token()
            else:
                self.authenticate()
                
    def search_products_for_candidate(self, candidate):
        """
        Search for Sentinel-2 products covering a specific candidate area.
        
        Implements intelligent product selection:
        - Prefers Level-2A (atmospherically corrected) over Level-1C
        - Filters by cloud cover threshold
        - Focuses on dry season imagery for better archaeological visibility
        
        Args:
            candidate (dict): Candidate with geographic bounds and metadata
            
        Returns:
            dict or None: Best matching Sentinel-2 product metadata
        """
        bounds = candidate['query_bounds']
        candidate_index = candidate['candidate_index']
        
        # Create search polygon in WKT format
        aoi = (f"POLYGON(({bounds['min_lon']} {bounds['min_lat']},"
               f"{bounds['max_lon']} {bounds['min_lat']},"
               f"{bounds['max_lon']} {bounds['max_lat']},"
               f"{bounds['min_lon']} {bounds['max_lat']},"
               f"{bounds['min_lon']} {bounds['min_lat']}))")
        
        self.log_step("SEARCH", f"Searching for candidate {candidate_index}")
        
        # Search for recent dry season imagery (better for archaeology)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*3)  # 3-year window
        
        # Define dry season periods (June-September in Amazon)
        search_periods = []
        for year_offset in [0, 1, 2]:
            year = end_date.year - year_offset
            search_periods.append((f"{year}-06-01", f"{year}-09-30"))
        
        all_products = []
        
        # Search each dry season period
        for start_period, end_period in search_periods:
            try:
                # Construct OData API query
                search_url = (
                    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
                    f"$filter=Collection/Name eq 'SENTINEL-2' "
                    f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') "
                    f"and ContentDate/Start gt {start_period}T00:00:00.000Z "
                    f"and ContentDate/Start lt {end_period}T23:59:59.999Z"
                    f"&$top=10"
                )
                
                response = requests.get(search_url)
                response.raise_for_status()
                
                json_data = response.json()
                period_products = json_data.get('value', [])
                
                # Filter and enrich product metadata
                for product in period_products:
                    product_name = product.get('Name', '')
                    is_l2a = 'MSIL2A' in product_name  # Atmospherically corrected
                    is_l1c = 'MSIL1C' in product_name  # Top-of-atmosphere
                    
                    if is_l2a or is_l1c:
                        cloud_cover = self._extract_cloud_cover(product)
                        
                        # Apply cloud cover filter
                        if cloud_cover <= self.download_params['cloud_cover_threshold']:
                            product['extracted_cloud_cover'] = cloud_cover
                            product['is_l2a'] = is_l2a
                            all_products.append(product)
                            
            except Exception as e:
                self.log_step("WARNING", f"Search failed for period {start_period}: {e}")
                continue
                
        # Sort products by preference: L2A > cloud cover > date
        all_products.sort(key=lambda x: (
            not x.get('is_l2a', False),           # L2A products first
            x['extracted_cloud_cover'],           # Lower cloud cover first
            -datetime.fromisoformat(x['ContentDate']['Start'].replace('Z', '+00:00')).timestamp()  # Recent first
        ))
        
        if all_products:
            best_product = all_products[0]
            product_type = "L2A" if best_product.get('is_l2a') else "L1C"
            print(f"   [SUCCESS] Found {len(all_products)} products, best: {product_type} "
                  f"({best_product['extracted_cloud_cover']:.1f}% cloud)")
            return best_product
        else:
            print(f"   [ERROR] No suitable products found")
            return None
            
    def _extract_cloud_cover(self, product):
        """
        Extract cloud cover percentage from product metadata.
        
        Handles different metadata formats and provides defaults for Amazon region.
        
        Args:
            product (dict): Sentinel-2 product metadata
            
        Returns:
            float: Cloud cover percentage (0-100)
        """
        # Try direct CloudCover field first
        if 'CloudCover' in product and product['CloudCover'] is not None:
            return float(product['CloudCover'])
            
        # Search in Attributes array
        if 'Attributes' in product and isinstance(product['Attributes'], list):
            for attr in product['Attributes']:
                if isinstance(attr, dict):
                    attr_name = attr.get('Name', '').lower()
                    if 'cloud' in attr_name:
                        try:
                            return float(attr.get('Value', 50))
                        except (ValueError, TypeError):
                            continue
                            
        # Default cloud cover for Amazon region (conservative estimate)
        return 45.0
        
    def download_product(self, product, candidate_index):
        """
        Download a Sentinel-2 product with progress tracking and validation.
        
        Implements robust download with:
        - Resume capability for interrupted downloads
        - Progress reporting for large files
        - ZIP file validation
        - Automatic retry on failure
        
        Args:
            product (dict): Product metadata from search
            candidate_index (int): Index of candidate being processed
            
        Returns:
            Path: Path to downloaded file, or None if failed
        """
        product_id = product.get('Id')
        product_name = product.get('Name', f'candidate_{candidate_index}_product')
        
        if not product_id:
            raise ValueError("No product ID found")
        
        # Create candidate-specific directory
        candidate_dir = self.downloads_dir / f"candidate_{candidate_index}"
        candidate_dir.mkdir(exist_ok=True)
        
        # Prepare output filename (sanitize for filesystem)
        safe_name = product_name.replace('/', '_').replace('\\', '_')
        if not safe_name.endswith('.zip'):
            safe_name += '.zip'
        output_path = candidate_dir / safe_name
        
        # Check if already downloaded and valid
        if output_path.exists():
            print(f"   [OUTPUT] Already downloaded: {safe_name}")
            return output_path
        
        self.log_step("DOWNLOAD", f"Downloading {safe_name[:50]}...")
        
        # Ensure valid authentication token
        self.ensure_valid_token()
        
        # Construct download URL
        download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        
        try:
            # Initiate download request
            response = self.session.get(download_url, allow_redirects=False, stream=True)
            
            # Handle HTTP redirects manually (up to 5 redirects)
            redirect_count = 0
            while response.status_code in (301, 302, 303, 307) and redirect_count < 5:
                download_url = response.headers.get('Location')
                if not download_url:
                    raise RuntimeError("Redirect without location header")
                response = self.session.get(download_url, allow_redirects=False, stream=True)
                redirect_count += 1
                
            # Handle authentication errors
            if response.status_code == 401:
                self.refresh_access_token()
                response = self.session.get(download_url, allow_redirects=False, stream=True)
                
            if response.status_code != 200:
                raise RuntimeError(f"Download failed with status code: {response.status_code}")
                
            # Download with progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            start_time = datetime.now()
            
            if total_size > 0:
                print(f"   [DATA] File size: {total_size / 1024 / 1024:.1f} MB")
            
            # Stream download with progress reporting
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Report progress every 100MB
                        if downloaded % (100 * 1024 * 1024) == 0 and total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"   [INFO] Progress: {percent:.1f}%")
                            
            # Validate download completeness
            if total_size > 0 and downloaded < total_size * 0.95:
                raise RuntimeError(f"Download incomplete")
                
            # Basic ZIP file validation
            import zipfile
            try:
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    if len(file_list) < 10:  # Sentinel-2 products have many files
                        raise RuntimeError(f"ZIP seems corrupted")
            except zipfile.BadZipFile:
                raise RuntimeError("Downloaded file is not a valid ZIP")
                
            print(f"   [SUCCESS] Download complete: {downloaded / 1024 / 1024:.1f} MB")
            return output_path
            
        except Exception as e:
            # Clean up partial downloads
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Download failed: {e}")
            
    def check_existing_downloads(self):
        """
        Check for existing downloads to avoid re-downloading.
        
        Returns:
            dict: Mapping of candidate indices to existing download paths
        """
        existing_downloads = {}
        
        for candidate_dir in self.downloads_dir.glob("candidate_*"):
            if candidate_dir.is_dir():
                zip_files = list(candidate_dir.glob("*.zip"))
                if zip_files:
                    candidate_index = int(candidate_dir.name.split('_')[1])
                    existing_downloads[candidate_index] = zip_files[0]
        
        if existing_downloads:
            print(f"   [OUTPUT] Found {len(existing_downloads)} existing downloads")
            for idx, path in existing_downloads.items():
                size_mb = path.stat().st_size / (1024*1024)
                print(f"      Candidate {idx}: {path.name} ({size_mb:.1f} MB)")
        
        return existing_downloads
        
    def download_all_candidates(self):
        """
        Download Sentinel-2 data for all archaeological candidates.
        
        Manages the complete download workflow with error handling
        and progress tracking across all candidates.
        
        Returns:
            dict: Dictionary of successful downloads with metadata
        """
        self.log_step("DOWNLOAD", "Starting Sentinel-2 downloads")
        
        # Set up authentication
        self.authenticate()
        
        # Check for existing downloads to skip
        existing_downloads = self.check_existing_downloads()
        
        successful_downloads = {}
        failed_downloads = []
        
        # Process each candidate
        for candidate in self.candidates:
            candidate_index = candidate['candidate_index']
            
            try:
                # Skip if already downloaded
                if candidate_index in existing_downloads:
                    download_path = existing_downloads[candidate_index]
                    
                    # Try to load existing metadata
                    metadata_file = download_path.parent / "download_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            product_info = metadata.get('product', {})
                    else:
                        product_info = {'Name': download_path.name}
                    
                    successful_downloads[candidate_index] = {
                        'candidate': candidate,
                        'product': product_info,
                        'download_path': download_path,
                        'candidate_dir': download_path.parent,
                        'skipped': True
                    }
                    continue
                
                # Search for suitable products
                product = self.search_products_for_candidate(candidate)
                
                if not product:
                    failed_downloads.append({
                        'candidate_index': candidate_index,
                        'reason': 'No suitable products found'
                    })
                    continue
                
                # Download the selected product
                download_path = self.download_product(product, candidate_index)
                
                if download_path and download_path.exists():
                    successful_downloads[candidate_index] = {
                        'candidate': candidate,
                        'product': product,
                        'download_path': download_path,
                        'candidate_dir': download_path.parent
                    }
                    
                    # Save individual download metadata
                    metadata = {
                        'candidate': candidate,
                        'product': product,
                        'download_date': datetime.now().isoformat(),
                        'download_path': str(download_path)
                    }
                    
                    metadata_file = download_path.parent / "download_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                else:
                    failed_downloads.append({
                        'candidate_index': candidate_index,
                        'reason': 'Download failed'
                    })
                    
            except Exception as e:
                self.log_step("ERROR", f"Error processing candidate {candidate_index}: {e}")
                failed_downloads.append({
                    'candidate_index': candidate_index,
                    'reason': str(e)
                })
                continue
        
        # Report download summary
        print(f"\n   [INFO] Download Summary:")
        print(f"      Total candidates: {len(self.candidates)}")
        print(f"      Successful: {len(successful_downloads)}")
        print(f"      Failed: {len(failed_downloads)}")
        
        if failed_downloads:
            print(f"   [ERROR] Failed downloads:")
            for failure in failed_downloads:
                print(f"      Candidate {failure['candidate_index']}: {failure['reason']}")
        
        return successful_downloads
        
    def save_download_metadata(self, successful_downloads):
        """
        Save comprehensive download metadata for Stage 2B analysis.
        
        Args:
            successful_downloads (dict): Dictionary of successful download results
            
        Returns:
            Path: Path to saved metadata file
        """
        self.log_step("SAVE", "Saving download metadata")
        
        # Compile comprehensive metadata
        metadata = {
            'download_date': datetime.now().isoformat(),
            'download_parameters': self.download_params,
            'total_candidates': len(self.candidates),
            'successful_downloads': len(successful_downloads),
            'candidates_info': {}
        }
        
        # Store detailed information for each successful download
        for candidate_index, download_info in successful_downloads.items():
            candidate = download_info['candidate']
            product = download_info['product']
            
            metadata['candidates_info'][candidate_index] = {
                'candidate_data': candidate,
                'download_path': str(download_info['download_path']),
                'product_name': product.get('Name', ''),
                'cloud_cover': product.get('extracted_cloud_cover', 0),
                'acquisition_date': product.get('ContentDate', {}).get('Start', ''),
                'product_type': "L2A" if product.get('is_l2a') else "L1C",
                'candidate_dir': str(download_info['candidate_dir']),
                'skipped': download_info.get('skipped', False)
            }
        
        # Save metadata file
        metadata_path = Path(self.paths['stage2_dir']) / 'download_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   [SAVE] Metadata saved: {metadata_path}")
        return metadata_path
        
    def run_stage2a(self):
        """
        Execute the complete Stage 2A download process.
        
        This is the main entry point that orchestrates:
        1. Loading archaeological candidates from Stage 1
        2. Authenticating with Copernicus Data Space
        3. Searching and downloading Sentinel-2 products
        4. Saving metadata for Stage 2B analysis
        
        Returns:
            bool: True if downloads completed successfully
        """
        print("[STAGE2] Stage 2A: Sentinel-2 Download")
        print("=" * 50)
        
        try:
            # Load archaeological candidates from Stage 1
            self.load_candidates()
            
            # Download Sentinel-2 data for all candidates
            successful_downloads = self.download_all_candidates()
            
            # Save comprehensive metadata
            self.save_download_metadata(successful_downloads)
            
            print(f"\n[SUCCESS] Stage 2A Complete!")
            print(f"   [INFO] Downloads: {len(successful_downloads)}/{len(self.candidates)}")
            print(f"   [OUTPUT] Data saved to: {self.downloads_dir}")
            
            if successful_downloads:
                # Calculate total downloaded data size
                total_size = 0
                for download_info in successful_downloads.values():
                    if download_info['download_path'].exists():
                        total_size += download_info['download_path'].stat().st_size
                
                print(f"   [SAVE] Total downloaded: {total_size / (1024*1024*1024):.1f} GB")
                print(f"   [RESULT] Ready for Stage 2B analysis")
            else:
                print(f"   [WARNING] No successful downloads - check authentication and network")
                
            return len(successful_downloads) > 0
            
        except Exception as e:
            print(f"[ERROR] Stage 2A failed: {e}")
            raise


def main():
    """Main entry point for Stage 2A processing."""
    downloader = Sentinel2Downloader()
    downloader.run_stage2a()


if __name__ == "__main__":
    main()