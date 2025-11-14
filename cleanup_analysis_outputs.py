"""
Cleanup utility for analysis output files.

Simple script to remove all generated plots and reports before running fresh analyses.
Useful when you want to regenerate everything or free up disk space. It asks for
confirmation before deleting anything.
"""

import os
import glob

def cleanup_analysis_outputs():
    """
    Remove all generated analysis output files.
    
    Finds and deletes PNG plot files and text report files matching the standard
    naming convention. Shows what was deleted and calculates total space freed.
    """
    # File patterns to match our standard output naming
    patterns = [
        "*_analysis_plots.png",
        "*_analysis_report.txt"
    ]
    
    deleted_files = []
    total_size = 0
    
    print("="*80)
    print("CLEANUP SCRIPT - Removing Analysis Output Files")
    print("="*80)
    print()
    
    # Find and delete files matching patterns
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                # Get file size before deletion
                size = os.path.getsize(file)
                total_size += size
                
                # Delete the file
                os.remove(file)
                deleted_files.append(file)
                print(f"  ✓ Deleted: {file} ({size:,} bytes)")
            except Exception as e:
                print(f"  ✗ Error deleting {file}: {str(e)}")
    
    # Summary
    print()
    print("="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    print(f"Files deleted: {len(deleted_files)}")
    print(f"Total space freed: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    
    if deleted_files:
        print("\nDeleted files:")
        for file in deleted_files:
            print(f"  - {file}")
    else:
        print("\nNo files found to delete.")
    
    print()
    print("="*80)
    print("Cleanup complete!")
    print("="*80)


if __name__ == "__main__":
    # Ask for confirmation
    print("\nThis script will delete all analysis output files:")
    print("  - *_analysis_plots.png (visualization files)")
    print("  - *_analysis_report.txt (report files)")
    print()
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        cleanup_analysis_outputs()
    else:
        print("\nCleanup cancelled.")

