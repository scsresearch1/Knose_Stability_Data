"""
BME690 Dual Sensor Stability Analysis Pipeline
Comprehensive analysis script for BME690 sensor data with S1 and S2 sensor sets

This script analyzes the 12_Nov_BME690_HP_354.csv file which contains data from two sensor sets:
- S1: Primary sensor readings (ADC_S1, Humidity_pct_S1, Temp_C_S1, Voltage_V_S1)
- S2: Secondary sensor readings (ADC_S2, Humidity_pct_S2, Temp_C_S2, Voltage_V_S2)

The analysis covers:
- Data quality assessment for both sensor sets
- Statistical characterization and comparison between S1 and S2
- Stability metrics for each sensor
- Time series analysis
- Sensor-to-sensor comparison and correlation analysis
- ML readiness assessment
- Performance metrics

Key features:
- Handles Running_Time column (HH:MM:SS format)
- Analyzes both sensors separately and together
- Generates comparative visualizations
- Comprehensive reporting for dual-sensor systems
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BME690DualSensorAnalyzer:
    """
    Analyzer class for BME690 dual sensor stability data.
    
    Handles analysis of both S1 and S2 sensor sets with comprehensive
    statistical analysis, stability metrics, and comparative evaluation.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.df = None
        self.results = {}
        
        # Define sensor column sets
        self.s1_cols = {
            'ADC': 'ADC_S1',
            'Humidity': 'Humidity_pct_S1',
            'Temp': 'Temp_C_S1',
            'Voltage': 'Voltage_V_S1'
        }
        self.s2_cols = {
            'ADC': 'ADC_S2',
            'Humidity': 'Humidity_pct_S2',
            'Temp': 'Temp_C_S2',
            'Voltage': 'Voltage_V_S2'
        }
        
    def parse_running_time_to_seconds(self, time_str):
        """Parse Running_Time string (HH:MM:SS) to total seconds."""
        try:
            parts = str(time_str).split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0
    
    def load_data(self):
        """Load CSV data and parse Running_Time column."""
        print(f"Loading data from {self.filename}...")
        self.df = pd.read_csv(self.filepath)
        
        # Parse Running_Time to seconds for analysis
        if 'Running_Time' in self.df.columns:
            self.df['TimeElapsed_sec'] = self.df['Running_Time'].apply(self.parse_running_time_to_seconds)
        else:
            # Fallback: use index if Running_Time not available
            self.df['TimeElapsed_sec'] = self.df.index
        
        # Handle datetime parsing if needed
        try:
            if 'Date' in self.df.columns and 'Time' in self.df.columns:
                try:
                    self.df['DateTime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'], 
                                                        format='%d-%m-%Y %H:%M:%S:%f', errors='coerce')
                except:
                    self.df['DateTime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not parse datetime: {e}")
        
        print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Check which sensors have data
        s1_has_data = any(self.df[col].notna().any() for col in self.s1_cols.values() if col in self.df.columns)
        s2_has_data = any(self.df[col].notna().any() for col in self.s2_cols.values() if col in self.df.columns)
        
        print(f"S1 sensor data available: {s1_has_data}")
        print(f"S2 sensor data available: {s2_has_data}")
        
        return self.df
    
    def analyze_sensor_set(self, sensor_name, sensor_cols):
        """
        Analyze a single sensor set (S1 or S2).
        Returns comprehensive analysis results.
        """
        results = {}
        
        # Get available columns
        available_cols = {k: v for k, v in sensor_cols.items() 
                          if v in self.df.columns and self.df[v].notna().any()}
        
        if not available_cols:
            return None
        
        # Data quality analysis
        quality = {}
        outliers_iqr = {}
        for param, col in available_cols.items():
            data = self.df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                    outliers_iqr[param] = {
                        'count': int(outliers),
                        'percentage': float(outliers / len(data) * 100),
                        'bounds': [float(lower_bound), float(upper_bound)]
                    }
        
        quality['outliers_iqr'] = outliers_iqr
        quality['total_rows'] = len(self.df)
        quality['rows_with_data'] = {}
        for param, col in available_cols.items():
            quality['rows_with_data'][param] = int(self.df[col].notna().sum())
        
        results['data_quality'] = quality
        
        # Statistical characterization
        stats_report = {}
        for param, col in available_cols.items():
            data = self.df[col].dropna()
            if len(data) > 0:
                stats_report[param] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'variance': float(data.var()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                    'coefficient_of_variation': float(data.std() / data.mean() * 100) if data.mean() != 0 else 0
                }
                
                # Normality tests
                if len(data) >= 8:
                    try:
                        _, p_dagostino = normaltest(data)
                        stats_report[param]['normality_dagostino_p'] = float(p_dagostino)
                    except:
                        pass
                    
                    if len(data) <= 5000:
                        try:
                            shapiro_stat, p_shapiro = shapiro(data)
                            stats_report[param]['shapiro_wilk'] = {
                                'statistic': float(shapiro_stat),
                                'p_value': float(p_shapiro),
                                'is_normal': p_shapiro > 0.05
                            }
                        except:
                            pass
                    
                    try:
                        jb_stat, p_jb = jarque_bera(data)
                        stats_report[param]['jarque_bera'] = {
                            'statistic': float(jb_stat),
                            'p_value': float(p_jb),
                            'is_normal': p_jb > 0.05
                        }
                    except:
                        pass
        
        results['statistical_characterization'] = stats_report
        
        # Stability metrics
        stability_report = {}
        for param, col in available_cols.items():
            data = self.df[col].dropna().values
            if len(data) < 2:
                continue
            
            # Linear drift analysis
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            drift_rate = slope
            
            cv = (np.std(data) / np.mean(data) * 100) if np.mean(data) != 0 else 0
            stability_index = 100 / (1 + abs(cv) + abs(drift_rate * len(data) / np.mean(data) * 100)) if np.mean(data) != 0 else 0
            
            # Long-term drift
            q1_data = data[:len(data)//4] if len(data) >= 4 else data
            q4_data = data[-len(data)//4:] if len(data) >= 4 else data
            long_term_drift = (np.mean(q4_data) - np.mean(q1_data)) / np.mean(q1_data) * 100 if np.mean(q1_data) != 0 else 0
            
            stability_report[param] = {
                'drift_rate_per_sample': float(drift_rate),
                'drift_p_value': float(p_value),
                'drift_r_squared': float(r_value**2),
                'coefficient_of_variation_pct': float(cv),
                'stability_index': float(stability_index),
                'long_term_drift_pct': float(long_term_drift),
                'mean_value': float(np.mean(data)),
                'std_value': float(np.std(data))
            }
        
        results['stability_metrics'] = stability_report
        
        # Time series analysis
        ts_report = {}
        for param, col in available_cols.items():
            data = self.df[col].dropna().values
            if len(data) < 10:
                continue
            
            # ADF test
            try:
                adf_result = adfuller(data)
                ts_report[param] = {
                    'adfuller': {
                        'statistic': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'is_stationary': adf_result[1] < 0.05
                    }
                }
            except Exception as e:
                ts_report[param] = {'adfuller_error': str(e)}
            
            # KPSS test
            try:
                kpss_result = kpss(data, regression='c')
                ts_report[param]['kpss'] = {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'is_stationary': kpss_result[1] > 0.05
                }
            except Exception as e:
                ts_report[param]['kpss_error'] = str(e)
            
            # Ljung-Box test
            if len(data) > 20:
                try:
                    lb_result = acorr_ljungbox(data, lags=min(10, len(data)//4), return_df=True)
                    ts_report[param]['ljung_box'] = {
                        'statistic': float(lb_result['lb_stat'].iloc[-1]) if len(lb_result) > 0 else None,
                        'p_value': float(lb_result['lb_pvalue'].iloc[-1]) if len(lb_result) > 0 else None,
                        'has_autocorr': float(lb_result['lb_pvalue'].iloc[-1]) < 0.05 if len(lb_result) > 0 else None
                    }
                except Exception as e:
                    ts_report[param]['ljung_box_error'] = str(e)
        
        results['time_series_analysis'] = ts_report
        
        return results
    
    def compare_sensors(self):
        """Compare S1 and S2 sensors - correlation and differences."""
        comparison = {}
        
        # Correlation analysis for overlapping data
        for param in ['ADC', 'Humidity', 'Temp', 'Voltage']:
            s1_col = self.s1_cols.get(param)
            s2_col = self.s2_cols.get(param)
            
            if s1_col in self.df.columns and s2_col in self.df.columns:
                # Get overlapping data (where both sensors have readings)
                mask = self.df[s1_col].notna() & self.df[s2_col].notna()
                if mask.sum() > 10:
                    s1_data = self.df.loc[mask, s1_col]
                    s2_data = self.df.loc[mask, s2_col]
                    
                    correlation = s1_data.corr(s2_data)
                    mean_diff = (s1_data - s2_data).mean()
                    std_diff = (s1_data - s2_data).std()
                    mean_abs_diff = (s1_data - s2_data).abs().mean()
                    mean_pct_diff = ((s1_data - s2_data) / s1_data * 100).mean()
                    
                    comparison[param] = {
                        'correlation': float(correlation),
                        'mean_difference': float(mean_diff),
                        'std_difference': float(std_diff),
                        'mean_absolute_difference': float(mean_abs_diff),
                        'mean_percentage_difference': float(mean_pct_diff),
                        'overlapping_samples': int(mask.sum())
                    }
        
        self.results['sensor_comparison'] = comparison
        return comparison
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations for both sensor sets."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Determine how many subplots we need
        num_params = 4  # ADC, Humidity, Temp, Voltage
        fig = plt.figure(figsize=(20, 24))
        
        # Time series plots - S1 and S2 together
        plot_idx = 1
        for param in ['ADC', 'Humidity', 'Temp', 'Voltage']:
            s1_col = self.s1_cols.get(param)
            s2_col = self.s2_cols.get(param)
            
            ax = plt.subplot(6, 2, plot_idx)
            
            # Plot S1
            if s1_col in self.df.columns:
                s1_mask = self.df[s1_col].notna()
                if s1_mask.any():
                    plt.plot(self.df.loc[s1_mask, 'TimeElapsed_sec'], 
                            self.df.loc[s1_mask, s1_col], 
                            label=f'{param}_S1', alpha=0.7, linewidth=1, color='blue')
            
            # Plot S2
            if s2_col in self.df.columns:
                s2_mask = self.df[s2_col].notna()
                if s2_mask.any():
                    plt.plot(self.df.loc[s2_mask, 'TimeElapsed_sec'], 
                            self.df.loc[s2_mask, s2_col], 
                            label=f'{param}_S2', alpha=0.7, linewidth=1, color='red')
            
            plt.xlabel('Time Elapsed (seconds)')
            plt.ylabel(param)
            plt.title(f'Time Series: {param} (S1 vs S2)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Distribution plots
        for param in ['ADC', 'Humidity', 'Temp', 'Voltage']:
            s1_col = self.s1_cols.get(param)
            s2_col = self.s2_cols.get(param)
            
            ax = plt.subplot(6, 2, plot_idx)
            
            if s1_col in self.df.columns:
                s1_data = self.df[s1_col].dropna()
                if len(s1_data) > 0:
                    plt.hist(s1_data, bins=50, alpha=0.5, label=f'{param}_S1', color='blue', edgecolor='black')
            
            if s2_col in self.df.columns:
                s2_data = self.df[s2_col].dropna()
                if len(s2_data) > 0:
                    plt.hist(s2_data, bins=50, alpha=0.5, label=f'{param}_S2', color='red', edgecolor='black')
            
            plt.xlabel(param)
            plt.ylabel('Frequency')
            plt.title(f'Distribution: {param} (S1 vs S2)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Correlation scatter plots (where both sensors have data)
        for i, param in enumerate(['ADC', 'Temp'], 1):
            s1_col = self.s1_cols.get(param)
            s2_col = self.s2_cols.get(param)
            
            if s1_col in self.df.columns and s2_col in self.df.columns:
                mask = self.df[s1_col].notna() & self.df[s2_col].notna()
                if mask.sum() > 10:
                    ax = plt.subplot(6, 2, plot_idx)
                    s1_data = self.df.loc[mask, s1_col]
                    s2_data = self.df.loc[mask, s2_col]
                    
                    plt.scatter(s1_data, s2_data, alpha=0.5, s=10)
                    plt.xlabel(f'{param}_S1')
                    plt.ylabel(f'{param}_S2')
                    plt.title(f'Correlation: {param}_S1 vs {param}_S2')
                    
                    # Add correlation line
                    if len(s1_data) > 1:
                        z = np.polyfit(s1_data, s2_data, 1)
                        p = np.poly1d(z)
                        plt.plot(s1_data, p(s1_data), "r--", alpha=0.8, linewidth=2)
                    
                    corr = s1_data.corr(s2_data)
                    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    plt.grid(True, alpha=0.3)
                    plot_idx += 1
        
        plt.tight_layout()
        output_file = self.filename.replace('.csv', '_analysis_plots.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {output_file}")
        plt.close()
        
        return output_file
    
    def generate_report(self):
        """Generate comprehensive text report."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report_file = self.filename.replace('.csv', '_analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"BME690 DUAL SENSOR STABILITY ANALYSIS REPORT\n")
            f.write(f"File: {self.filename}\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # S1 Analysis
            if 's1_analysis' in self.results and self.results['s1_analysis']:
                f.write("="*80 + "\n")
                f.write("SENSOR S1 ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                s1 = self.results['s1_analysis']
                
                # Data Quality
                if 'data_quality' in s1:
                    f.write("1. DATA QUALITY\n")
                    f.write("-"*80 + "\n")
                    dq = s1['data_quality']
                    f.write(f"Total Rows: {dq['total_rows']}\n")
                    for param, count in dq.get('rows_with_data', {}).items():
                        f.write(f"  {param}: {count} rows with data\n")
                    f.write("\n")
                
                # Statistical Characterization
                if 'statistical_characterization' in s1:
                    f.write("2. STATISTICAL CHARACTERIZATION\n")
                    f.write("-"*80 + "\n")
                    for param, stats_dict in s1['statistical_characterization'].items():
                        f.write(f"\n{param}:\n")
                        f.write(f"  Mean: {stats_dict['mean']:.4f}\n")
                        f.write(f"  Std: {stats_dict['std']:.4f}\n")
                        f.write(f"  CV: {stats_dict['coefficient_of_variation']:.2f}%\n")
                        f.write("\n")
                
                # Stability Metrics
                if 'stability_metrics' in s1:
                    f.write("3. STABILITY METRICS\n")
                    f.write("-"*80 + "\n")
                    for param, metrics in s1['stability_metrics'].items():
                        f.write(f"\n{param}:\n")
                        f.write(f"  CV: {metrics['coefficient_of_variation_pct']:.4f}%\n")
                        f.write(f"  Stability Index: {metrics['stability_index']:.2f}\n")
                        f.write(f"  Long-term Drift: {metrics['long_term_drift_pct']:.4f}%\n")
                        f.write("\n")
            
            # S2 Analysis
            if 's2_analysis' in self.results and self.results['s2_analysis']:
                f.write("\n" + "="*80 + "\n")
                f.write("SENSOR S2 ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                s2 = self.results['s2_analysis']
                
                # Data Quality
                if 'data_quality' in s2:
                    f.write("1. DATA QUALITY\n")
                    f.write("-"*80 + "\n")
                    dq = s2['data_quality']
                    f.write(f"Total Rows: {dq['total_rows']}\n")
                    for param, count in dq.get('rows_with_data', {}).items():
                        f.write(f"  {param}: {count} rows with data\n")
                    f.write("\n")
                
                # Statistical Characterization
                if 'statistical_characterization' in s2:
                    f.write("2. STATISTICAL CHARACTERIZATION\n")
                    f.write("-"*80 + "\n")
                    for param, stats_dict in s2['statistical_characterization'].items():
                        f.write(f"\n{param}:\n")
                        f.write(f"  Mean: {stats_dict['mean']:.4f}\n")
                        f.write(f"  Std: {stats_dict['std']:.4f}\n")
                        f.write(f"  CV: {stats_dict['coefficient_of_variation']:.2f}%\n")
                        f.write("\n")
                
                # Stability Metrics
                if 'stability_metrics' in s2:
                    f.write("3. STABILITY METRICS\n")
                    f.write("-"*80 + "\n")
                    for param, metrics in s2['stability_metrics'].items():
                        f.write(f"\n{param}:\n")
                        f.write(f"  CV: {metrics['coefficient_of_variation_pct']:.4f}%\n")
                        f.write(f"  Stability Index: {metrics['stability_index']:.2f}\n")
                        f.write(f"  Long-term Drift: {metrics['long_term_drift_pct']:.4f}%\n")
                        f.write("\n")
            
            # Sensor Comparison
            if 'sensor_comparison' in self.results:
                f.write("\n" + "="*80 + "\n")
                f.write("SENSOR COMPARISON (S1 vs S2)\n")
                f.write("="*80 + "\n\n")
                
                for param, comp in self.results['sensor_comparison'].items():
                    f.write(f"{param}:\n")
                    f.write(f"  Correlation: {comp['correlation']:.4f}\n")
                    f.write(f"  Mean Difference: {comp['mean_difference']:.4f}\n")
                    f.write(f"  Mean Absolute Difference: {comp['mean_absolute_difference']:.4f}\n")
                    f.write(f"  Mean Percentage Difference: {comp['mean_percentage_difference']:.2f}%\n")
                    f.write(f"  Overlapping Samples: {comp['overlapping_samples']}\n")
                    f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Report saved to: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Execute all analysis steps."""
        print("\n" + "="*80)
        print(f"BME690 DUAL SENSOR STABILITY ANALYSIS")
        print(f"File: {self.filename}")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Analyze S1
        print("\n" + "="*60)
        print("ANALYZING SENSOR S1")
        print("="*60)
        self.results['s1_analysis'] = self.analyze_sensor_set('S1', self.s1_cols)
        
        # Analyze S2
        print("\n" + "="*60)
        print("ANALYZING SENSOR S2")
        print("="*60)
        self.results['s2_analysis'] = self.analyze_sensor_set('S2', self.s2_cols)
        
        # Compare sensors
        print("\n" + "="*60)
        print("COMPARING SENSORS S1 AND S2")
        print("="*60)
        self.compare_sensors()
        
        # Generate outputs
        self.generate_visualizations()
        self.generate_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        return self.results


if __name__ == "__main__":
    # Analyze the BME690 file
    filepath = r'RawDryRunData\12_Nov_BME690_HP_354.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        exit(1)
    
    print("="*80)
    print("BME690 DUAL SENSOR DATA ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {os.path.basename(filepath)}")
    print("\n" + "="*80)
    
    try:
        analyzer = BME690DualSensorAnalyzer(filepath)
        results = analyzer.run_complete_analysis()
        print("\n✓ Analysis completed successfully!")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

