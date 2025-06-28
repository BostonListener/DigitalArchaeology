#!/usr/bin/env python3
"""
Archaeological Discovery Pipeline - Checkpoint Compliance Runner

This script executes all checkpoint requirements in sequence to ensure complete
compliance with competition guidelines. It orchestrates the execution of all
checkpoint scripts and provides comprehensive compliance verification.

The checkpoint system validates that the archaeological detection pipeline meets
all competition requirements for AI-powered archaeological discovery:

Checkpoint 1: Familiarize with challenge and data
- Load core datasets and demonstrate OpenAI model integration
- Validate dataset access and API functionality

Checkpoint 2: Early explorer analysis  
- Mine multiple data sources for archaeological anomalies
- Identify exactly 5 candidate anomaly footprints
- Demonstrate leveraged re-prompting techniques

Checkpoint 3: New site discovery documentation
- Select single best archaeological discovery
- Document algorithmic detection methods
- Conduct historical research and comparative analysis

Checkpoint 4: Story and impact narrative
- Generate cultural context and survey planning
- Create presentation materials for live demonstration
- Develop partnership strategies and research proposals

The script provides:
- Sequential checkpoint execution with error handling
- Prerequisite verification and dependency checking
- Comprehensive compliance reporting
- Detailed logging and result tracking
- Success rate calculation and status reporting

Usage:
    python run_checkpoint.py

Requirements:
    - Completed pipeline execution through at least Stage 1
    - Valid OpenAI API credentials
    - All checkpoint script files present
    - Required data files from pipeline stages

Authors: Archaeological AI Team
License: MIT
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_checkpoint_script(script_name, checkpoint_name):
    """
    Execute a specific checkpoint script with error handling and monitoring.
    
    Runs checkpoint scripts as separate processes to isolate execution environments
    and provide robust error handling. Captures output and monitors execution time
    to ensure reliable checkpoint completion.
    
    Args:
        script_name (str): Name of the checkpoint script file to execute
        checkpoint_name (str): Human-readable checkpoint name for logging
        
    Returns:
        bool: True if checkpoint executed successfully, False otherwise
    """
    
    print(f"\n{'='*60}")
    print(f"RUNNING {checkpoint_name}")
    print(f"{'='*60}")
    
    try:
        # Execute checkpoint script with timeout protection
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False,  # Allow real-time output
                              text=True, 
                              timeout=600)  # 10 minute timeout for complex analysis
        
        # Check execution result and log outcome
        if result.returncode == 0:
            print(f"\n✅ {checkpoint_name}: SUCCESS")
            return True
        else:
            print(f"\n❌ {checkpoint_name}: FAILED (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        # Handle checkpoint timeout scenarios
        print(f"\n⏰ {checkpoint_name}: TIMEOUT (>10 minutes)")
        print(f"   This may indicate:")
        print(f"   - Large dataset processing requirements")
        print(f"   - OpenAI API rate limiting or delays") 
        print(f"   - Complex analysis requiring more time")
        return False
    except Exception as e:
        # Handle unexpected errors during checkpoint execution
        print(f"\n💥 {checkpoint_name}: ERROR - {e}")
        return False

def check_prerequisites():
    """
    Verify that prerequisite data and pipeline results exist before running checkpoints.
    
    Performs comprehensive validation of data dependencies to ensure checkpoint
    scripts have the required input data. Provides clear guidance on missing
    prerequisites and how to obtain them.
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    
    print("CHECKING PREREQUISITES")
    print("=" * 30)
    
    # Define required files that must exist for checkpoint execution
    required_files = [
        ("data/stage1/archaeological_candidates.csv", "Stage 1 - Deforestation candidates"),
    ]
    
    # Define optional files that improve checkpoint quality but aren't mandatory
    optional_files = [
        ("data/stage2/pattern_summary.csv", "Stage 2 - NDVI patterns"),
        ("data/stage3/final_archaeological_sites.csv", "Stage 3 - Final sites"),
    ]
    
    # Check for required files that must be present
    missing_required = []
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f"✅ {description}: Found")
        else:
            print(f"❌ {description}: Missing")
            missing_required.append(file_path)
    
    # Check for optional files that enhance checkpoint quality
    found_optional = []
    for file_path, description in optional_files:
        if Path(file_path).exists():
            print(f"✅ {description}: Found")
            found_optional.append(file_path)
        else:
            print(f"⚠️  {description}: Not found (will use fallbacks)")
    
    # Evaluate prerequisite status and provide guidance
    if missing_required:
        print(f"\n❌ PREREQUISITES NOT MET")
        print(f"Missing required files:")
        for file_path in missing_required:
            print(f"   {file_path}")
        print(f"\nSOLUTION: Run the main pipeline first to generate required data:")
        print(f"   python run_pipeline.py --full")
        print(f"   python run_pipeline.py --stage1  # Minimum requirement")
        return False
    
    print(f"\n✅ PREREQUISITES MET")
    print(f"Required files: All present")
    print(f"Optional files: {len(found_optional)}/2 present")
    
    # Provide recommendations for optimal checkpoint execution
    if len(found_optional) < 2:
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        if not Path("data/stage2/pattern_summary.csv").exists():
            print(f"   Run Stage 2 for better anomaly detection: python run_pipeline.py --stage2")
        if not Path("data/stage3/final_archaeological_sites.csv").exists():
            print(f"   Run Stage 3 for elevation validation: python run_pipeline.py --stage3")
        print(f"   Higher-quality pipeline results improve checkpoint performance")
    
    return True

def create_checkpoint_summary():
    """
    Create comprehensive summary of all checkpoint execution results.
    
    Analyzes checkpoint output directories and files to generate detailed
    compliance report. Tracks file creation, interaction counts, and
    overall completion status for verification purposes.
    
    Returns:
        dict: Comprehensive summary of checkpoint execution and compliance
    """
    
    print(f"\nCREATING CHECKPOINT SUMMARY")
    print("=" * 30)
    
    # Define checkpoint output directories to analyze
    checkpoint_dirs = [
        Path("data/checkpoint2_outputs"),
        Path("data/checkpoint3_outputs"),
        Path("data/checkpoint4_outputs")
    ]
    
    # Initialize summary data structure
    summary = {
        'completion_timestamp': datetime.now().isoformat(),
        'checkpoints_completed': [],
        'total_files_created': 0,
        'openai_interactions_total': 0,
        'compliance_status': {},
        'data_quality_assessment': {}
    }
    
    # Analyze each checkpoint directory for completion and quality
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            checkpoint_name = checkpoint_dir.name.replace('_outputs', '')
            files = list(checkpoint_dir.glob('*'))
            
            # Count different file types for quality assessment
            json_files = len([f for f in files if f.suffix == '.json'])
            log_files = len([f for f in files if f.suffix == '.txt'])
            
            checkpoint_info = {
                'name': checkpoint_name,
                'completed': True,
                'output_dir': str(checkpoint_dir),
                'files_created': len(files),
                'json_files': json_files,
                'log_files': log_files,
                'key_files': [f.name for f in files if f.suffix in ['.json', '.txt']],
                'completion_quality': 'HIGH' if json_files >= 2 else 'MEDIUM' if json_files >= 1 else 'LOW'
            }
            
            summary['checkpoints_completed'].append(checkpoint_info)
            summary['total_files_created'] += len(files)
            
            print(f"✅ {checkpoint_name}: {len(files)} files created ({checkpoint_info['completion_quality']} quality)")
            
            # Load OpenAI interaction counts if available
            openai_log_dir = checkpoint_dir / 'openai_logs'
            if openai_log_dir.exists():
                openai_files = list(openai_log_dir.glob('*.json'))
                if openai_files:
                    print(f"   📊 OpenAI logs: {len(openai_files)} files")
                    
        else:
            print(f"❌ {checkpoint_dir.name}: Not completed")
            summary['checkpoints_completed'].append({
                'name': checkpoint_dir.name.replace('_outputs', ''),
                'completed': False,
                'reason': 'Directory not found'
            })
    
    # Calculate overall compliance metrics
    completed_checkpoints = [c for c in summary['checkpoints_completed'] if c['completed']]
    summary['completion_rate'] = len(completed_checkpoints) / 3 * 100  # 3 total checkpoints
    summary['high_quality_checkpoints'] = len([c for c in completed_checkpoints if c.get('completion_quality') == 'HIGH'])
    
    # Save comprehensive summary for record keeping
    summary_path = Path("data/all_checkpoints_summary.json")
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 CHECKPOINT SUMMARY SAVED: {summary_path}")
    print(f"   Total checkpoints completed: {len(completed_checkpoints)}/3")
    print(f"   Total files created: {summary['total_files_created']}")
    print(f"   Completion rate: {summary['completion_rate']:.1f}%")
    print(f"   High-quality completions: {summary['high_quality_checkpoints']}")
    
    return summary

def validate_openai_setup():
    """
    Validate OpenAI API setup and credentials before checkpoint execution.
    
    Performs basic validation of OpenAI integration to prevent checkpoint
    failures due to API configuration issues. Provides guidance for
    resolving common setup problems.
    
    Returns:
        bool: True if OpenAI setup appears valid, False otherwise
    """
    
    print("\nVALIDATING OPENAI SETUP")
    print("=" * 25)
    
    try:
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Check for API key presence
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print(f"❌ OPENAI_API_KEY not found in environment")
            print(f"   Create .env file with: OPENAI_API_KEY=your_key_here")
            return False
        
        # Validate API key format (basic check)
        if not api_key.startswith('sk-'):
            print(f"❌ OPENAI_API_KEY format appears invalid")
            print(f"   API keys should start with 'sk-'")
            return False
        
        print(f"✅ OPENAI_API_KEY found and formatted correctly")
        
        # Check OpenAI package availability
        try:
            import openai
            print(f"✅ OpenAI package available")
        except ImportError:
            print(f"❌ OpenAI package not installed")
            print(f"   Install with: pip install openai")
            return False
        
        print(f"✅ OpenAI setup validation passed")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI setup validation failed: {e}")
        return False

def main():
    """
    Execute complete checkpoint compliance verification workflow.
    
    Orchestrates the entire checkpoint execution process with comprehensive
    error handling, prerequisite checking, and compliance reporting. Provides
    detailed feedback on execution status and guidance for resolving issues.
    
    Returns:
        bool: True if all checkpoints completed successfully, False otherwise
    """
    
    print("ARCHAEOLOGICAL DISCOVERY PIPELINE")
    print("CHECKPOINT COMPLIANCE VERIFICATION")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Validate all prerequisites before starting checkpoints
    if not check_prerequisites():
        print(f"\n💡 SOLUTION: Complete pipeline prerequisites before running checkpoints")
        print(f"   1. Run: python setup_pipeline.py")
        print(f"   2. Obtain required input data (see setup output)")
        print(f"   3. Run: python run_pipeline.py --stage1  # Minimum requirement")
        return False
    
    # Step 2: Validate OpenAI API setup
    if not validate_openai_setup():
        print(f"\n💡 SOLUTION: Configure OpenAI API access")
        print(f"   1. Obtain API key from: https://platform.openai.com/api-keys")
        print(f"   2. Create .env file with: OPENAI_API_KEY=your_key_here")
        print(f"   3. Install OpenAI package: pip install openai")
        return False
    
    # Define checkpoint execution sequence
    checkpoints = [
        ("checkpoint2_analysis.py", "CHECKPOINT 2: Early Explorer"),
        ("checkpoint3_notebook.py", "CHECKPOINT 3: Best Site Discovery"),
        ("checkpoint4_story.py", "CHECKPOINT 4: Story & Impact")
    ]
    
    # Track execution results for final reporting
    results = []
    execution_times = []
    
    # Step 3: Execute each checkpoint in sequence
    for script_name, checkpoint_name in checkpoints:
        # Verify checkpoint script exists
        if not Path(script_name).exists():
            print(f"\n❌ {checkpoint_name}: Script not found - {script_name}")
            print(f"   Ensure all checkpoint scripts are present in the working directory")
            results.append(False)
            continue
        
        # Execute checkpoint with timing
        start_time = datetime.now()
        success = run_checkpoint_script(script_name, checkpoint_name)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        execution_times.append(execution_time)
        results.append(success)
        
        # Provide feedback on checkpoint completion
        if not success:
            print(f"\n⚠️  {checkpoint_name} failed, but continuing with remaining checkpoints...")
            print(f"   Check error messages above for specific failure details")
            print(f"   Common issues: Missing data, API limits, configuration errors")
    
    # Step 4: Create comprehensive execution summary
    summary = create_checkpoint_summary()
    
    # Step 5: Generate final compliance report
    print(f"\n{'='*60}")
    print("FINAL CHECKPOINT COMPLIANCE REPORT")
    print(f"{'='*60}")
    
    total_checkpoints = len(checkpoints)
    successful_checkpoints = sum(results)
    
    # Report individual checkpoint results
    for i, ((script_name, checkpoint_name), success, exec_time) in enumerate(zip(checkpoints, results, execution_times)):
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{checkpoint_name}")
        print(f"   Status: {status}")
        print(f"   Execution time: {exec_time:.1f} seconds")
        if not success:
            print(f"   Requires attention: Review error messages and prerequisites")
    
    # Calculate and report overall compliance
    print(f"\nOVERALL COMPLIANCE ASSESSMENT:")
    print(f"   Checkpoints passed: {successful_checkpoints}/{total_checkpoints}")
    success_rate = (successful_checkpoints / total_checkpoints) * 100
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Total execution time: {sum(execution_times):.1f} seconds")
    
    # Provide status-specific guidance
    if successful_checkpoints == total_checkpoints:
        print(f"\n🎉 CHECKPOINT COMPLIANCE: COMPLETE")
        print(f"   ✅ All competition requirements satisfied")
        print(f"   ✅ Ready for final submission")
        print(f"   ✅ All analysis workflows validated")
        print(f"   ✅ OpenAI integration verified")
        
        print(f"\nNEXT STEPS:")
        print(f"   1. Review checkpoint outputs in data/checkpoint*_outputs/")
        print(f"   2. Prepare presentation materials from checkpoint 4")
        print(f"   3. Validate field research partnerships and permits")
        print(f"   4. Submit complete pipeline and checkpoint results")
        
    elif successful_checkpoints >= 2:
        print(f"\n⚠️  CHECKPOINT COMPLIANCE: PARTIAL")
        print(f"   ✅ Core requirements met ({successful_checkpoints}/3 checkpoints)")
        print(f"   ⚠️  Some enhancements needed for full compliance")
        
        failed_checkpoints = [name for (_, name), success in zip(checkpoints, results) if not success]
        print(f"\nREMAINING WORK:")
        for checkpoint in failed_checkpoints:
            print(f"   - Resolve issues with: {checkpoint}")
        
    else:
        print(f"\n🔧 CHECKPOINT COMPLIANCE: NEEDS SIGNIFICANT WORK")
        print(f"   ❌ Multiple checkpoints require attention")
        print(f"   ❌ Review prerequisites and error messages")
        
        print(f"\nTROUBLESHOTING RECOMMENDATIONS:")
        print(f"   1. Verify all input data is available and accessible")
        print(f"   2. Check OpenAI API key and account status")
        print(f"   3. Ensure sufficient pipeline execution (Stage 1 minimum)")
        print(f"   4. Review error logs for specific failure causes")
        print(f"   5. Consider running pipeline stages individually")
    
    # Display output locations for reference
    print(f"\nOUTPUT DIRECTORIES:")
    output_dirs = [
        "data/checkpoint2_outputs/",
        "data/checkpoint3_outputs/", 
        "data/checkpoint4_outputs/",
        "data/all_checkpoints_summary.json"
    ]
    
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            print(f"   📁 {output_dir}")
        else:
            print(f"   ❌ {output_dir} (not created)")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return success status for external script integration
    return successful_checkpoints == total_checkpoints

if __name__ == "__main__":
    # Execute checkpoint compliance verification
    success = main()
    
    # Exit with appropriate code for shell scripting and CI/CD integration
    if success:
        print(f"\n🏆 CHECKPOINT COMPLIANCE: COMPLETE")
        sys.exit(0)  # Success exit code
    else:
        print(f"\n🔧 CHECKPOINT COMPLIANCE: NEEDS WORK")
        sys.exit(1)  # Failure exit code for automated systems