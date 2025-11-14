"""
Heater Profile Comparison Tool
Cross-day comparison of HP322 and HP354 sensor performance

I built this to compare how the two heater profiles perform across different test days.
It extracts key metrics from each data file and organizes them into comparison tables
that make it easy to spot trends, differences, and performance variations.

The script automatically identifies the heater profile (322 or 354) and test day from
the filename, then groups the results accordingly. Each profile gets its own table
showing how metrics change across days - useful for identifying which profile is more
stable or if there are day-specific effects.

Outputs two CSV files that can be opened in Excel for further analysis or reporting.
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from analyze_all_sensor_data import SensorStabilityAnalyzer

def extract_date_and_profile(filename):
    """
    Parse filename to extract test day and heater profile.
    
    Our filenames have inconsistent formatting, so I use regex to handle variations
    like "8_oct Hp_322 exp1.b.1.csv" and "14_oct hp_322 dry testing.csv". The regex
    looks for the day number before "oct" and the 3-digit profile number after "hp" or "h_".
    """
    filename_lower = filename.lower()
    
    # Extract day number (e.g., 8, 9, 10, 13, 14)
    date_match = re.search(r'(\d+)[_\s]*oct', filename_lower)
    day = None
    if date_match:
        day = int(date_match.group(1))
    
    # Extract heater profile number (322 or 354)
    profile_match = re.search(r'(?:hp|h_)[_\s]*(\d{3})', filename_lower)
    profile = None
    if profile_match:
        profile = profile_match.group(1)
    
    return day, profile

def analyze_file_for_comparison(filepath):
    """
    Quick analysis of a single file to extract comparison metrics.
    
    This runs a subset of the full analysis - just the parts needed for comparison.
    I skip the visualizations and full report generation to speed things up when
    processing multiple files. The extracted metrics focus on ADC performance and
    stability, which are the key indicators for comparing heater profiles.
    """
    try:
        analyzer = SensorStabilityAnalyzer(filepath)
        analyzer.load_data()
        analyzer.data_quality_analysis()
        analyzer.statistical_characterization()
        analyzer.stability_metrics()
        analyzer.sensor_performance_metrics()
        
        # Extract key metrics for comparison
        summary = {
            'filename': os.path.basename(filepath),
            'total_rows': len(analyzer.df),
        }
        
        # ADC metrics
        if 'ADC' in analyzer.df.columns:
            adc_data = analyzer.df['ADC'].dropna()
            if len(adc_data) > 0:
                summary['adc_mean'] = float(adc_data.mean())
                summary['adc_std'] = float(adc_data.std())
                summary['adc_cv_pct'] = float(adc_data.std() / adc_data.mean() * 100) if adc_data.mean() != 0 else 0
                summary['adc_min'] = float(adc_data.min())
                summary['adc_max'] = float(adc_data.max())
                summary['adc_range'] = float(adc_data.max() - adc_data.min())
        
        # Stability metrics from ADC
        if 'stability_metrics' in analyzer.results and 'ADC' in analyzer.results['stability_metrics']:
            adc_stability = analyzer.results['stability_metrics']['ADC']
            summary['adc_drift_rate'] = adc_stability.get('drift_rate_per_sample', 0)
            summary['adc_stability_index'] = adc_stability.get('stability_index', 0)
            summary['adc_long_term_drift_pct'] = adc_stability.get('long_term_drift_pct', 0)
        
        # Temperature metrics
        if 'Temp_C' in analyzer.df.columns:
            temp_data = analyzer.df['Temp_C'].dropna()
            if len(temp_data) > 0:
                summary['temp_mean'] = float(temp_data.mean())
                summary['temp_std'] = float(temp_data.std())
                summary['temp_min'] = float(temp_data.min())
                summary['temp_max'] = float(temp_data.max())
        
        # Humidity metrics
        if 'Humidity_pct' in analyzer.df.columns:
            hum_data = analyzer.df['Humidity_pct'].dropna()
            if len(hum_data) > 0:
                summary['humidity_mean'] = float(hum_data.mean())
                summary['humidity_std'] = float(hum_data.std())
        
        # Voltage metrics
        if 'Voltage_V' in analyzer.df.columns:
            volt_data = analyzer.df['Voltage_V'].dropna()
            if len(volt_data) > 0:
                summary['voltage_mean'] = float(volt_data.mean())
                summary['voltage_std'] = float(volt_data.std())
                summary['voltage_cv_pct'] = float(volt_data.std() / volt_data.mean() * 100) if volt_data.mean() != 0 else 0
        
        # Sensor performance (SNR)
        if 'sensor_performance' in analyzer.results and 'ADC' in analyzer.results['sensor_performance']:
            adc_perf = analyzer.results['sensor_performance']['ADC']
            summary['adc_snr_db'] = adc_perf.get('snr_db', 0)
        
        # Outliers count
        if 'data_quality' in analyzer.results:
            dq = analyzer.results['data_quality']
            if 'outliers_iqr' in dq and 'ADC' in dq['outliers_iqr']:
                summary['adc_outliers_count'] = dq['outliers_iqr']['ADC'].get('count', 0)
                summary['adc_outliers_pct'] = dq['outliers_iqr']['ADC'].get('percentage', 0)
        
        return summary
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {str(e)}")
        return None

def create_comparison_tables():
    """Create comparison tables for HP322 and HP354"""
    
    print("="*80)
    print("HEATER PROFILE COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    # Find all CSV files
    data_dir = "RawDryRunData"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    csv_files.sort()
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}/")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze...")
    print()
    
    # Analyze all files and collect summaries
    all_summaries = []
    for csv_file in csv_files:
        print(f"Analyzing: {os.path.basename(csv_file)}...")
        summary = analyze_file_for_comparison(csv_file)
        if summary:
            day, profile = extract_date_and_profile(summary['filename'])
            summary['day'] = day
            summary['profile'] = profile
            all_summaries.append(summary)
    
    if not all_summaries:
        print("No valid summaries collected.")
        return
    
    # Create DataFrames for each profile
    df_all = pd.DataFrame(all_summaries)
    
    # Separate by profile
    df_322 = df_all[df_all['profile'] == '322'].copy()
    df_354 = df_all[df_all['profile'] == '354'].copy()
    
    # Sort by day
    df_322 = df_322.sort_values('day').reset_index(drop=True)
    df_354 = df_354.sort_values('day').reset_index(drop=True)
    
    # Create comparison tables
    print("\n" + "="*80)
    print("GENERATING COMPARISON TABLES")
    print("="*80)
    print()
    
    # Table 1: HP322 Comparison
    if len(df_322) > 0:
        print("TABLE 1: HP322 HEATER PROFILE - COMPARISON ACROSS DAYS")
        print("="*80)
        
        # Select key columns for comparison
        comparison_cols_322 = {
            'Day': 'day',
            'Filename': 'filename',
            'Samples': 'total_rows',
            'ADC Mean': 'adc_mean',
            'ADC Std': 'adc_std',
            'ADC CV (%)': 'adc_cv_pct',
            'ADC Range': 'adc_range',
            'Drift Rate': 'adc_drift_rate',
            'Stability Index': 'adc_stability_index',
            'Long-term Drift (%)': 'adc_long_term_drift_pct',
            'SNR (dB)': 'adc_snr_db',
            'Outliers (%)': 'adc_outliers_pct',
            'Temp Mean (°C)': 'temp_mean',
            'Humidity Mean (%)': 'humidity_mean',
            'Voltage Mean (V)': 'voltage_mean'
        }
        
        table_322 = pd.DataFrame()
        for display_name, col_name in comparison_cols_322.items():
            if col_name in df_322.columns:
                table_322[display_name] = df_322[col_name]
        
        # Format numeric columns
        numeric_cols = table_322.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'Day' not in col and 'Samples' not in col:
                table_322[col] = table_322[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) and abs(x) < 1000 else f"{x:.2f}" if pd.notna(x) else "N/A")
        
        print(table_322.to_string(index=False))
        print()
        
        # Save to CSV
        output_file_322 = "HP322_comparison_table.csv"
        table_322.to_csv(output_file_322, index=False)
        print(f"✓ Saved HP322 comparison table to: {output_file_322}")
        print()
    
    # Table 2: HP354 Comparison
    if len(df_354) > 0:
        print("TABLE 2: HP354 HEATER PROFILE - COMPARISON ACROSS DAYS")
        print("="*80)
        
        # Select key columns for comparison
        comparison_cols_354 = {
            'Day': 'day',
            'Filename': 'filename',
            'Samples': 'total_rows',
            'ADC Mean': 'adc_mean',
            'ADC Std': 'adc_std',
            'ADC CV (%)': 'adc_cv_pct',
            'ADC Range': 'adc_range',
            'Drift Rate': 'adc_drift_rate',
            'Stability Index': 'adc_stability_index',
            'Long-term Drift (%)': 'adc_long_term_drift_pct',
            'SNR (dB)': 'adc_snr_db',
            'Outliers (%)': 'adc_outliers_pct',
            'Temp Mean (°C)': 'temp_mean',
            'Humidity Mean (%)': 'humidity_mean',
            'Voltage Mean (V)': 'voltage_mean'
        }
        
        table_354 = pd.DataFrame()
        for display_name, col_name in comparison_cols_354.items():
            if col_name in df_354.columns:
                table_354[display_name] = df_354[col_name]
        
        # Format numeric columns
        numeric_cols = table_354.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'Day' not in col and 'Samples' not in col:
                table_354[col] = table_354[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) and abs(x) < 1000 else f"{x:.2f}" if pd.notna(x) else "N/A")
        
        print(table_354.to_string(index=False))
        print()
        
        # Save to CSV
        output_file_354 = "HP354_comparison_table.csv"
        table_354.to_csv(output_file_354, index=False)
        print(f"✓ Saved HP354 comparison table to: {output_file_354}")
        print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()
    
    if len(df_322) > 0:
        print("HP322 Profile:")
        print(f"  Total files analyzed: {len(df_322)}")
        print(f"  Days covered: {sorted(df_322['day'].unique())}")
        if 'adc_mean' in df_322.columns:
            print(f"  Average ADC Mean: {df_322['adc_mean'].mean():.4f}")
            print(f"  Average Stability Index: {df_322['adc_stability_index'].mean():.4f}")
        print()
    
    if len(df_354) > 0:
        print("HP354 Profile:")
        print(f"  Total files analyzed: {len(df_354)}")
        print(f"  Days covered: {sorted(df_354['day'].unique())}")
        if 'adc_mean' in df_354.columns:
            print(f"  Average ADC Mean: {df_354['adc_mean'].mean():.4f}")
            print(f"  Average Stability Index: {df_354['adc_stability_index'].mean():.4f}")
        print()
    
    print("="*80)
    print("COMPARISON ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    create_comparison_tables()

