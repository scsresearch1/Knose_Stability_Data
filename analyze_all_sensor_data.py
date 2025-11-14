"""
Sensor Stability Analysis Pipeline
Batch processing script for analyzing multiple sensor data files

I built this to handle our dry run testing data systematically. It processes all CSV files
in the RawDryRunData folder and generates the standard analysis outputs we need for each
test run. The analysis covers data quality checks, statistical characterization, stability
metrics, and sensor performance evaluation.

The script automatically discovers all CSV files and processes them sequentially, which
saves a lot of time compared to running individual analyses. Each file gets its own set
of plots and a detailed text report.

Key analyses performed:
- Data quality assessment (outlier detection, temporal consistency)
- Statistical characterization (distribution properties, normality tests)
- Stability metrics (drift analysis, coefficient of variation, Allan variance)
- Time series analysis (stationarity, autocorrelation)
- ML readiness assessment (feature engineering, normalization needs)
- Sensor performance (SNR, resolution, environmental correlations)


STATISTICAL TESTS AND METHODS - DEFINITIONS
===========================================

Isolation Forest (Anomaly Detection):
--------------------------------------
An unsupervised anomaly detection algorithm that works by randomly selecting features
and split values to isolate observations. The idea is that anomalies are easier to
isolate (require fewer splits) than normal points. It's particularly useful for sensor
data because it can detect multivariate anomalies - cases where individual sensor
channels might look normal, but their combination is unusual. For example, a reading
where temperature and humidity are both high might be normal, but if ADC is also
unusually high at the same time, that could be an anomaly. The contamination parameter
(0.1 = 10%) sets the expected proportion of anomalies in the data. Lower values are
more conservative (fewer anomalies detected), higher values are more aggressive.

Interpretation:
- Anomalies detected: Number of data points flagged as unusual
- Anomaly percentage: Proportion of data identified as anomalous
- Higher values indicate more unusual patterns in the data


Shapiro-Wilk Test (Normality Test):
-----------------------------------
One of the most powerful normality tests, especially for small to medium sample sizes
(up to 5000 points). It compares the sample data to a normal distribution with the same
mean and variance. The test statistic (W) ranges from 0 to 1, with values closer to 1
indicating better fit to normality. A p-value > 0.05 suggests the data is normally
distributed. This test is sensitive to both skewness and kurtosis deviations. For sensor
data, normal distributions are ideal because many statistical methods assume normality,
but real sensor data often shows some deviation due to environmental effects or sensor
characteristics.

Null hypothesis: Data follows a normal distribution
Interpretation:
- p-value > 0.05: Accept null hypothesis (data is normal)
- p-value <= 0.05: Reject null hypothesis (data is not normal)
- W statistic close to 1: More normal distribution


Jarque-Bera Test (Normality Test):
----------------------------------
Tests whether sample data has the skewness and kurtosis matching a normal distribution.
It's based on the fact that normal distributions have zero skewness and zero excess
kurtosis. The test statistic combines both measures - higher values indicate greater
deviation from normality. This test works well for larger samples and is particularly
useful when you want to know specifically if skewness or kurtosis is the problem. For
sensor stability analysis, non-normal distributions might indicate systematic biases
(skewness) or heavy tails from outliers (kurtosis).

Null hypothesis: Data has normal skewness and kurtosis (i.e., is normally distributed)
Interpretation:
- p-value > 0.05: Accept null hypothesis (data is normal)
- p-value <= 0.05: Reject null hypothesis (data is not normal)
- Higher test statistic: Greater deviation from normality


Augmented Dickey-Fuller (ADF) Test (Stationarity Test):
-------------------------------------------------------
Tests for unit roots in a time series - essentially checking if the series has a
stochastic trend (random walk behavior). The null hypothesis is that the series is
non-stationary (has a unit root). If p < 0.05, we reject the null and conclude the
series is stationary. For sensor data, stationarity means the mean and variance
don't change over time - this is what we want for stable sensors. Non-stationary
behavior indicates drift, trends, or other systematic changes that could affect
sensor reliability. The test includes lagged differences to account for autocorrelation.

Null hypothesis: Series has a unit root (is non-stationary)
Interpretation:
- p-value < 0.05: Reject null hypothesis (series is stationary) - GOOD for sensors
- p-value >= 0.05: Accept null hypothesis (series is non-stationary) - indicates drift/trend
- Negative test statistic with large magnitude: Suggests stationarity


KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin) (Stationarity Test):
------------------------------------------------------------------
The complement to ADF - tests the null hypothesis that the series is stationary
(trend-stationary or level-stationary). If p > 0.05, we accept stationarity. This
test is useful because ADF and KPSS can sometimes give conflicting results - if both
agree, we have strong evidence. KPSS is particularly good at detecting trend-stationary
processes (where removing a trend makes it stationary). For sensors, this helps
distinguish between true drift (non-stationary) and a constant offset (stationary
around a different mean).

Null hypothesis: Series is stationary (around a level or trend)
Interpretation:
- p-value > 0.05: Accept null hypothesis (series is stationary) - GOOD for sensors
- p-value <= 0.05: Reject null hypothesis (series is non-stationary) - indicates drift
- Lower test statistic: Suggests stationarity


Ljung-Box Test (Autocorrelation Test):
--------------------------------------
Tests whether autocorrelations of the time series are significantly different from zero.
The null hypothesis is that the data are independently distributed (no autocorrelation).
If p < 0.05, we reject the null and conclude there is significant autocorrelation.
This is important for sensor data because high autocorrelation suggests the sensor
has memory - each reading depends on previous readings. This could indicate slow
response times, filtering effects, or systematic dependencies. For stability analysis,
we generally want low autocorrelation (white noise) because it means the sensor
responds independently to each measurement. The test examines multiple lags to detect
autocorrelation at different time scales.

Null hypothesis: No autocorrelation (data are independently distributed)
Interpretation:
- p-value < 0.05: Reject null hypothesis (significant autocorrelation exists)
- p-value >= 0.05: Accept null hypothesis (no significant autocorrelation) - GOOD for sensors
- Higher test statistic: Indicates stronger autocorrelation
- For stable sensors, we want low autocorrelation (white noise behavior)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
import glob
# Set style for better plots
# Use seaborn style if available, otherwise fall back to default style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except Exception:
    plt.style.use('default')

if sns is not None:
    try:
        sns.set_palette("husl")
    except Exception:
        pass

class SensorStabilityAnalyzer:
    """
    Main analyzer class for sensor stability data.
    
    This handles the full analysis pipeline for a single sensor data file. I've structured
    it so each analysis step is independent - makes it easier to debug and modify specific
    parts without breaking the whole workflow.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split('\\')[-1].split('/')[-1]
        self.df = None
        self.results = {}
        
    def load_data(self):
        """
        Load CSV data and handle datetime parsing.
        
        Our data files have inconsistent date formats depending on when they were exported,
        so I've added fallback parsing to handle the common variations. The time elapsed
        calculation is critical for stability analysis - we need proper temporal ordering.
        """
        print(f"Loading data from {self.filename}...")
        self.df = pd.read_csv(self.filepath)
        
        # Handle date/time parsing
        try:
            if 'Date' in self.df.columns and 'Time' in self.df.columns:
                # Try different date formats
                try:
                    self.df['DateTime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'], 
                                                        format='%d-%m-%Y %H:%M:%S:%f', errors='coerce')
                except:
                    try:
                        self.df['DateTime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'], 
                                                            format='%d/%m/%Y %H:%M:%S:%f', errors='coerce')
                    except:
                        self.df['DateTime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'], errors='coerce')
                
                self.df = self.df.sort_values('DateTime').reset_index(drop=True)
                self.df['TimeElapsed_sec'] = (self.df['DateTime'] - self.df['DateTime'].iloc[0]).dt.total_seconds()
        except Exception as e:
            print(f"Warning: Could not parse datetime: {e}")
            self.df['TimeElapsed_sec'] = self.df.index
        
        print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def data_quality_analysis(self):
        """
        Data quality and integrity checks.
        
        I use multiple outlier detection methods because each has different strengths:
        - IQR method: Good for detecting values outside the expected distribution range
        - Z-score: Catches extreme statistical outliers (3 sigma rule)
        - Isolation Forest: Finds anomalies that don't fit the multivariate pattern
          (see module docstring for detailed definition)
        
        For sensor data, outliers can indicate measurement errors, environmental disturbances,
        or actual sensor issues. The IQR bounds are particularly useful for understanding
        the normal operating range of each sensor channel.
        """
        print("\n" + "="*60)
        print("1. DATA QUALITY & INTEGRITY ANALYSIS")
        print("="*60)
        
        quality_report = {}
        
        quality_report['total_rows'] = len(self.df)
        quality_report['total_columns'] = len(self.df.columns)
        
        # IQR-based outlier detection - standard approach for sensor data
        numeric_cols = ['ADC', 'Humidity_pct', 'Temp_C', 'Voltage_V']
        outliers_iqr = {}
        for col in numeric_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outliers_iqr[col] = {
                    'count': int(outliers),
                    'percentage': float(outliers / len(self.df) * 100),
                    'bounds': [float(lower_bound), float(upper_bound)]
                }
        quality_report['outliers_iqr'] = outliers_iqr
        
        # Z-score method - catches statistical outliers (3-sigma rule)
        # This is more sensitive than IQR and helps identify extreme deviations
        outliers_zscore = {}
        for col in numeric_cols:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = (z_scores > 3).sum()
                outliers_zscore[col] = {
                    'count': int(outliers),
                    'percentage': float(outliers / len(self.df) * 100)
                }
        quality_report['outliers_zscore'] = outliers_zscore
        
        # Isolation Forest - see module docstring for definition
        # Detects multivariate anomalies by isolating unusual data point combinations
        isolation_forest_results = {}
        if len(self.df) > 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            numeric_data = self.df[numeric_cols].dropna()
            if len(numeric_data) > 0:
                iso_forest.fit(numeric_data)
                anomalies = iso_forest.predict(numeric_data)
                anomaly_count = int((anomalies == -1).sum())
                anomaly_pct = float(anomaly_count / len(numeric_data) * 100)
                isolation_forest_results = {
                    'anomalies_detected': anomaly_count,
                    'anomaly_percentage': anomaly_pct,
                    'total_samples_analyzed': len(numeric_data),
                    'contamination_parameter': 0.1
                }
                quality_report['isolation_forest_anomalies'] = anomaly_count
        quality_report['isolation_forest_results'] = isolation_forest_results
        
        # Temporal consistency check
        # Irregular sampling intervals can affect stability calculations, so we track this
        if 'TimeElapsed_sec' in self.df.columns:
            time_diffs = self.df['TimeElapsed_sec'].diff().dropna()
            quality_report['sampling_rate_stats'] = {
                'mean_interval_sec': float(time_diffs.mean()),
                'std_interval_sec': float(time_diffs.std()),
                'min_interval_sec': float(time_diffs.min()),
                'max_interval_sec': float(time_diffs.max()),
                'median_interval_sec': float(time_diffs.median())
            }
            # Large gaps might indicate data logging interruptions or system resets
            large_gaps = (time_diffs > time_diffs.median() * 3).sum()
            quality_report['large_temporal_gaps'] = int(large_gaps)
        
        self.results['data_quality'] = quality_report
        
        # Print summary
        print(f"Total Rows: {quality_report['total_rows']}")
        print(f"Outliers (IQR method): {sum([v['count'] for v in outliers_iqr.values()])}")
        
        # Isolation Forest results
        if isolation_forest_results:
            anomaly_pct = isolation_forest_results['anomaly_percentage']
            if anomaly_pct < 5:
                interpretation = "Very few anomalies detected - data quality is excellent"
            elif anomaly_pct < 10:
                interpretation = "Low anomaly rate - data quality is good (within expected contamination parameter)"
            elif anomaly_pct < 20:
                interpretation = "Moderate anomaly rate - some unusual patterns detected (may need investigation)"
            else:
                interpretation = "High anomaly rate - significant unusual patterns detected (data quality concern)"
            print(f"\nIsolation Forest Anomaly Detection:")
            print(f"  Anomalies detected: {isolation_forest_results['anomalies_detected']} ({anomaly_pct:.2f}%)")
            print(f"  Samples analyzed: {isolation_forest_results['total_samples_analyzed']}")
            print(f"  Interpretation: {interpretation}")
        
        return quality_report
    
    def statistical_characterization(self):
        """
        Statistical characterization of sensor channels.
        
        I calculate the standard descriptive statistics plus distribution shape metrics.
        Skewness and kurtosis tell us about the distribution shape - important for understanding
        if the sensor response is symmetric or if there are systematic biases.
        
        The coefficient of variation (CV) is particularly useful for sensor data - it's the
        relative standard deviation normalized by the mean. Lower CV generally means better
        stability, but the acceptable threshold depends on the sensor type and application.
        
        Normality tests (Shapiro-Wilk and Jarque-Bera) help determine if parametric statistical
        methods are appropriate. See module docstring for detailed definitions of these tests.
        I run multiple tests because they have different sensitivities and assumptions.
        """
        print("\n" + "="*60)
        print("2. STATISTICAL CHARACTERIZATION")
        print("="*60)
        
        numeric_cols = ['ADC', 'Humidity_pct', 'Temp_C', 'Voltage_V']
        stats_report = {}
        
        for col in numeric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    stats_report[col] = {
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
                    
                    # Normality testing - useful for deciding on statistical methods
                    # D'Agostino works well for larger samples, Shapiro-Wilk is more powerful
                    # for smaller samples but has a 5000 point limit
                    if len(data) >= 8:
                        try:
                            _, p_dagostino = normaltest(data)
                            stats_report[col]['normality_dagostino_p'] = float(p_dagostino)
                        except:
                            pass
                        
                        # Shapiro-Wilk test - see module docstring for definition
                        # Most powerful for small to medium samples (up to 5000 points)
                        if len(data) <= 5000:
                            try:
                                shapiro_stat, p_shapiro = shapiro(data)
                                stats_report[col]['shapiro_wilk'] = {
                                    'statistic': float(shapiro_stat),
                                    'p_value': float(p_shapiro),
                                    'is_normal': p_shapiro > 0.05
                                }
                            except:
                                pass
                        
                        # Jarque-Bera test - see module docstring for definition
                        # Tests normality via skewness and kurtosis
                        try:
                            jb_stat, p_jb = jarque_bera(data)
                            stats_report[col]['jarque_bera'] = {
                                'statistic': float(jb_stat),
                                'p_value': float(p_jb),
                                'is_normal': p_jb > 0.05
                            }
                        except:
                            pass
        
        self.results['statistical_characterization'] = stats_report
        
        # Print summary
        print("\nKey Statistics:")
        for col, stats_dict in stats_report.items():
            if isinstance(stats_dict, dict) and 'mean' in stats_dict:
                print(f"\n{col}:")
                print(f"  Mean: {stats_dict['mean']:.4f}, Std: {stats_dict['std']:.4f}")
                print(f"  CV: {stats_dict['coefficient_of_variation']:.2f}%")
                print(f"  Skewness: {stats_dict['skewness']:.4f}, Kurtosis: {stats_dict['kurtosis']:.4f}")
                
                # Shapiro-Wilk test results
                if 'shapiro_wilk' in stats_dict:
                    sw = stats_dict['shapiro_wilk']
                    interpretation = "Data appears normally distributed" if sw['is_normal'] else "Data does NOT follow normal distribution (may have skewness/kurtosis issues)"
                    print(f"  Shapiro-Wilk: statistic={sw['statistic']:.4f}, p={sw['p_value']:.4f}")
                    print(f"    Interpretation: {interpretation}")
                
                # Jarque-Bera test results
                if 'jarque_bera' in stats_dict:
                    jb = stats_dict['jarque_bera']
                    interpretation = "Data appears normally distributed" if jb['is_normal'] else "Data does NOT follow normal distribution (skewness/kurtosis mismatch)"
                    print(f"  Jarque-Bera: statistic={jb['statistic']:.4f}, p={jb['p_value']:.4f}")
                    print(f"    Interpretation: {interpretation}")
        
        return stats_report
    
    def stability_metrics(self):
        """
        Stability analysis - the core of sensor performance evaluation.
        
        Drift analysis uses linear regression to detect systematic trends over time. A
        significant drift (low p-value) suggests the sensor is changing its baseline,
        which could indicate aging, temperature effects, or other systematic issues.
        
        The stability index is a composite metric I developed that combines CV and drift.
        Higher values indicate better stability. It's normalized so we can compare across
        different sensor channels and test runs.
        
        Allan variance is borrowed from precision measurement (atomic clocks, etc.). It
        helps characterize the noise characteristics at different time scales. For sensors,
        this tells us about short-term noise vs. long-term drift behavior.
        
        Long-term drift compares the first and last quartiles - gives a practical measure
        of how much the sensor baseline shifted over the entire test period.
        """
        print("\n" + "="*60)
        print("3. STABILITY METRICS ANALYSIS")
        print("="*60)
        
        numeric_cols = ['ADC', 'Humidity_pct', 'Temp_C', 'Voltage_V']
        stability_report = {}
        
        for col in numeric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna().values
                if len(data) < 2:
                    continue
                
                # Linear drift analysis - detects systematic trends
                x = np.arange(len(data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
                drift_rate = slope  # units per sample
                
                # Coefficient of Variation - relative variability
                cv = (np.std(data) / np.mean(data) * 100) if np.mean(data) != 0 else 0
                
                # Custom stability index - combines CV and drift into single metric
                # Higher is better (inverse relationship with variability and drift)
                stability_index = 100 / (1 + abs(cv) + abs(drift_rate * len(data) / np.mean(data) * 100)) if np.mean(data) != 0 else 0
                
                # Allan variance - noise characterization at different time scales
                # Useful for understanding sensor noise characteristics
                if len(data) >= 10:
                    tau = 1
                    allan_vars = []
                    for m in range(1, min(10, len(data)//2)):
                        if m * tau < len(data):
                            clusters = len(data) // m
                            if clusters > 1:
                                cluster_means = [np.mean(data[i*m:(i+1)*m]) for i in range(clusters)]
                                if len(cluster_means) > 1:
                                    allan_var = np.var(np.diff(cluster_means)) / 2
                                    allan_vars.append(allan_var)
                    avg_allan_var = np.mean(allan_vars) if allan_vars else np.nan
                else:
                    avg_allan_var = np.nan
                
                # Long-term drift - practical measure of baseline shift
                q1_data = data[:len(data)//4]
                q4_data = data[-len(data)//4:]
                long_term_drift = (np.mean(q4_data) - np.mean(q1_data)) / np.mean(q1_data) * 100 if np.mean(q1_data) != 0 else 0
                
                stability_report[col] = {
                    'drift_rate_per_sample': float(drift_rate),
                    'drift_p_value': float(p_value),
                    'drift_r_squared': float(r_value**2),
                    'coefficient_of_variation_pct': float(cv),
                    'stability_index': float(stability_index),
                    'allan_variance': float(avg_allan_var) if not np.isnan(avg_allan_var) else None,
                    'long_term_drift_pct': float(long_term_drift),
                    'mean_value': float(np.mean(data)),
                    'std_value': float(np.std(data))
                }
        
        self.results['stability_metrics'] = stability_report
        
        # Print summary
        print("\nStability Metrics:")
        for col, metrics in stability_report.items():
            print(f"\n{col}:")
            print(f"  Drift Rate: {metrics['drift_rate_per_sample']:.6f} per sample")
            print(f"  CV: {metrics['coefficient_of_variation_pct']:.4f}%")
            print(f"  Stability Index: {metrics['stability_index']:.2f}")
            print(f"  Long-term Drift: {metrics['long_term_drift_pct']:.4f}%")
        
        return stability_report
    
    def time_series_analysis(self):
        """
        Time series analysis for temporal pattern detection.
        
        Stationarity tests (ADF and KPSS) check if the statistical properties of the signal
        are constant over time. For sensor stability, we generally want stationary signals -
        non-stationary behavior suggests drift or systematic changes. See module docstring
        for detailed definitions of ADF and KPSS tests.
        
        Autocorrelation measures how much each sample depends on previous samples. High
        autocorrelation means the sensor has memory effects or slow response times. Low
        autocorrelation suggests the noise is more random/white. The Ljung-Box test
        (see module docstring) formally tests for autocorrelation.
        
        Rate of change statistics help identify sudden jumps or rapid transitions, which
        could indicate measurement artifacts or environmental disturbances.
        """
        print("\n" + "="*60)
        print("4. TIME SERIES ANALYSIS")
        print("="*60)
        
        numeric_cols = ['ADC', 'Humidity_pct', 'Temp_C', 'Voltage_V']
        ts_report = {}
        
        for col in numeric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna().values
                if len(data) < 10:
                    continue
                
                # ADF test - see module docstring for definition
                # Tests for unit root (non-stationarity)
                try:
                    adf_result = adfuller(data)
                    ts_report[col] = {
                        'adfuller': {
                            'statistic': float(adf_result[0]),
                            'p_value': float(adf_result[1]),
                            'is_stationary': adf_result[1] < 0.05,
                            'critical_values': {f'level_{k}': float(v) for k, v in adf_result[4].items()}
                        }
                    }
                except Exception as e:
                    ts_report[col] = {'adfuller_error': str(e)}
                
                # KPSS test - see module docstring for definition
                # Tests for stationarity (complement to ADF)
                try:
                    kpss_result = kpss(data, regression='c')
                    ts_report[col]['kpss'] = {
                        'statistic': float(kpss_result[0]),
                        'p_value': float(kpss_result[1]),
                        'is_stationary': kpss_result[1] > 0.05,
                        'critical_values': {f'level_{k}': float(v) for k, v in kpss_result[3].items()}
                    }
                except Exception as e:
                    ts_report[col]['kpss_error'] = str(e)
                
                # Ljung-Box test - see module docstring for definition
                # Tests for autocorrelation at multiple lags
                if len(data) > 20:
                    try:
                        lb_result = acorr_ljungbox(data, lags=min(10, len(data)//4), return_df=True)
                        ts_report[col]['ljung_box'] = {
                            'statistic': float(lb_result['lb_stat'].iloc[-1]) if len(lb_result) > 0 else None,
                            'p_value': float(lb_result['lb_pvalue'].iloc[-1]) if len(lb_result) > 0 else None,
                            'has_autocorr': float(lb_result['lb_pvalue'].iloc[-1]) < 0.05 if len(lb_result) > 0 else None,
                            'lags_tested': len(lb_result)
                        }
                    except Exception as e:
                        ts_report[col]['ljung_box_error'] = str(e)
                
                # Autocorrelation - measures temporal dependence
                # Lag-1 autocorrelation is most important for sensor data
                if len(data) > 20:
                    try:
                        autocorr = [np.corrcoef(data[:-i], data[i:])[0,1] for i in range(1, min(20, len(data)//4))]
                        ts_report[col]['autocorrelation_lag1'] = float(autocorr[0]) if len(autocorr) > 0 else None
                        ts_report[col]['autocorrelation_mean'] = float(np.mean(autocorr)) if len(autocorr) > 0 else None
                    except:
                        pass
                
                # Rate of change - identifies rapid transitions
                if len(data) > 1:
                    rate_of_change = np.diff(data)
                    ts_report[col]['mean_rate_of_change'] = float(np.mean(rate_of_change))
                    ts_report[col]['std_rate_of_change'] = float(np.std(rate_of_change))
                    ts_report[col]['max_rate_of_change'] = float(np.max(np.abs(rate_of_change)))
        
        self.results['time_series_analysis'] = ts_report
        
        # Print summary
        print("\nTime Series Analysis:")
        for col, analysis in ts_report.items():
            print(f"\n{col}:")
            
            # ADF test results
            if 'adfuller' in analysis:
                adf = analysis['adfuller']
                if adf['is_stationary']:
                    interpretation = "Series is STATIONARY - good for sensor stability (no significant drift detected)"
                else:
                    interpretation = "Series is NON-STATIONARY - indicates drift, trend, or systematic changes (potential stability concern)"
                print(f"  ADF Test (Augmented Dickey-Fuller):")
                print(f"    Statistic: {adf['statistic']:.4f}, p-value: {adf['p_value']:.4f}")
                print(f"    Interpretation: {interpretation}")
            
            # KPSS test results
            if 'kpss' in analysis:
                kpss_test = analysis['kpss']
                if kpss_test['is_stationary']:
                    interpretation = "Series is STATIONARY - good for sensor stability (statistical properties constant over time)"
                else:
                    interpretation = "Series is NON-STATIONARY - indicates drift or trend (potential stability concern)"
                print(f"  KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin):")
                print(f"    Statistic: {kpss_test['statistic']:.4f}, p-value: {kpss_test['p_value']:.4f}")
                print(f"    Interpretation: {interpretation}")
            
            # Ljung-Box test results
            if 'ljung_box' in analysis:
                lb = analysis['ljung_box']
                if lb['statistic'] is not None:
                    if not lb['has_autocorr']:
                        interpretation = "No significant autocorrelation - good (sensor responds independently to each measurement)"
                    else:
                        interpretation = "Significant autocorrelation detected - sensor has memory effects (may indicate slow response or filtering)"
                    print(f"  Ljung-Box Test:")
                    print(f"    Statistic: {lb['statistic']:.4f}, p-value: {lb['p_value']:.4f}")
                    print(f"    Interpretation: {interpretation}")
                    print(f"    Lags tested: {lb['lags_tested']}")
            
            # Autocorrelation
            if 'autocorrelation_lag1' in analysis and analysis['autocorrelation_lag1'] is not None:
                print(f"  Autocorrelation (lag 1): {analysis['autocorrelation_lag1']:.4f}")
        
        return ts_report
    
    def ml_readiness_assessment(self):
        """
        ML readiness evaluation for potential model training.
        
        This section assesses whether the data is suitable for machine learning applications.
        I check feature counts, sample sizes, data balance, and normalization needs.
        
        The feature importance metric uses normalized variance as a proxy - features with
        higher relative variance tend to be more informative, though this is just a
        rough indicator. For sensor data, we often need domain knowledge to determine
        true feature importance.
        
        Normalization requirements are based on coefficient of variation and dynamic range.
        Sensor data often spans multiple orders of magnitude (e.g., ADC counts vs. temperature),
        so normalization is usually necessary for ML algorithms.
        """
        print("\n" + "="*60)
        print("5. ML READINESS ASSESSMENT")
        print("="*60)
        
        numeric_cols = ['ADC', 'Humidity_pct', 'Temp_C', 'Voltage_V']
        ml_report = {}
        
        ml_report['feature_count'] = len(numeric_cols)
        ml_report['sample_count'] = len(self.df)
        ml_report['samples_per_feature'] = len(self.df) / len(numeric_cols) if len(numeric_cols) > 0 else 0
        
        # Check if we have phase/condition labels for supervised learning
        if 'phase' in self.df.columns or 'PHASE' in self.df.columns:
            phase_col = 'phase' if 'phase' in self.df.columns else 'PHASE'
            phase_counts = self.df[phase_col].value_counts()
            ml_report['phase_distribution'] = phase_counts.to_dict()
            ml_report['phase_balance_ratio'] = float(phase_counts.min() / phase_counts.max()) if len(phase_counts) > 0 else 0
        
        # Feature importance proxy - normalized variance
        # Higher values suggest more informative features (rough estimate)
        feature_importance = {}
        for col in numeric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    feature_importance[col] = float(data.var() / data.mean()**2) if data.mean() != 0 else 0
        
        ml_report['feature_importance_variance'] = feature_importance
        
        # Normalization needs - check CV and dynamic range
        # High CV or large range differences suggest normalization is needed
        normalization_needs = {}
        for col in numeric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    cv = data.std() / data.mean() if data.mean() != 0 else 0
                    needs_normalization = cv > 0.1 or data.max() / data.min() > 10 if data.min() > 0 else True
                    normalization_needs[col] = {
                        'needs_normalization': needs_normalization,
                        'recommended_method': 'standardization' if needs_normalization else 'none',
                        'current_range': [float(data.min()), float(data.max())]
                    }
        
        ml_report['normalization_requirements'] = normalization_needs
        
        # Standard train/val/test split recommendation
        ml_report['recommended_split'] = {
            'train': 0.7,
            'validation': 0.15,
            'test': 0.15,
            'min_samples_for_ml': 100
        }
        self.results['ml_readiness'] = ml_report
        
        # Print summary
        print(f"Sample Count: {ml_report['sample_count']}")
        print(f"Samples per Feature: {ml_report['samples_per_feature']:.1f}")
        print(f"Recommended Train/Val/Test Split: 70%/15%/15%")
        
        return ml_report
    
    def sensor_performance_metrics(self):
        """
        Sensor-specific performance evaluation.
        
        SNR (Signal-to-Noise Ratio) is calculated in dB using the standard formula.
        For sensor data, higher SNR means the signal is more distinguishable from noise.
        Typical good sensors have SNR > 20 dB, excellent ones > 40 dB.
        
        Resolution is the minimum detectable change - essentially the quantization step
        or smallest meaningful difference the sensor can measure. For ADC data, this is
        often related to the bit depth of the analog-to-digital converter.
        
        Voltage stability is critical - power supply variations can directly affect
        sensor readings. We track this separately because it's a system-level issue
        rather than a sensor characteristic.
        
        Environmental correlations help identify cross-sensitivities. For example, if
        ADC readings correlate strongly with temperature, we might need temperature
        compensation in the final application.
        """
        print("\n" + "="*60)
        print("6. SENSOR PERFORMANCE METRICS")
        print("="*60)
        
        performance_report = {}
        
        # Signal-to-Noise Ratio (SNR) in dB
        # Standard formula: 20*log10(signal/noise), where noise is std dev
        for col in ['ADC', 'Humidity_pct', 'Temp_C']:
            if col in self.df.columns:
                data = self.df[col].dropna().values
                if len(data) > 1:
                    signal = np.mean(data)
                    noise = np.std(data)
                    snr = 20 * np.log10(signal / noise) if noise > 0 else np.inf
                    performance_report[col] = {
                        'snr_db': float(snr),
                        'signal_mean': float(signal),
                        'noise_std': float(noise)
                    }
        
        # Resolution - minimum detectable change
        # For ADC, this is the smallest step size in the digital output
        if 'ADC' in self.df.columns:
            adc_data = self.df['ADC'].dropna().values
            if len(adc_data) > 1:
                resolution = np.min(np.abs(np.diff(np.unique(adc_data))))
                performance_report['ADC_resolution'] = float(resolution)
        
        # Power supply stability - affects all measurements
        if 'Voltage_V' in self.df.columns:
            voltage_data = self.df['Voltage_V'].dropna()
            performance_report['voltage_stability'] = {
                'mean_voltage': float(voltage_data.mean()),
                'std_voltage': float(voltage_data.std()),
                'voltage_cv_pct': float(voltage_data.std() / voltage_data.mean() * 100) if voltage_data.mean() != 0 else 0,
                'voltage_range': [float(voltage_data.min()), float(voltage_data.max())]
            }
        
        # Cross-sensitivity analysis - environmental effects on sensor output
        if all(col in self.df.columns for col in ['Temp_C', 'Humidity_pct', 'ADC']):
            temp_humidity_corr = self.df['Temp_C'].corr(self.df['Humidity_pct'])
            temp_adc_corr = self.df['Temp_C'].corr(self.df['ADC'])
            humidity_adc_corr = self.df['Humidity_pct'].corr(self.df['ADC'])
            
            performance_report['environmental_correlations'] = {
                'temp_humidity': float(temp_humidity_corr),
                'temp_adc': float(temp_adc_corr),
                'humidity_adc': float(humidity_adc_corr)
            }
        
        self.results['sensor_performance'] = performance_report
        
        # Print summary
        print("\nSensor Performance:")
        for metric, value in performance_report.items():
            if isinstance(value, dict) and 'snr_db' in value:
                print(f"{metric}: SNR = {value['snr_db']:.2f} dB")
        
        return performance_report
    
    def generate_visualizations(self):
        """
        Generate visualization plots for sensor data analysis.
        
        I create a 5x2 grid of plots covering the essential visualizations:
        - Time series: Shows temporal behavior and any obvious trends or anomalies
        - Distributions: Reveals the data shape and helps identify outliers visually
        - Box plots: Quick comparison of variability across sensor channels
        - Stability trend: Visual comparison of first vs. last quartile (drift indicator)
        
        The plots are saved as high-resolution PNG files suitable for reports and presentations.
        """
        print("\n" + "="*60)
        print("7. GENERATING VISUALIZATIONS")
        print("="*60)
        
        fig = plt.figure(figsize=(20, 20))
        
        numeric_cols = ['ADC', 'Humidity_pct', 'Temp_C', 'Voltage_V']
        
        # Time series plots - essential for spotting trends and anomalies
        for i, col in enumerate(numeric_cols, 1):
            if col in self.df.columns:
                ax = plt.subplot(5, 2, i)
                if 'TimeElapsed_sec' in self.df.columns:
                    plt.plot(self.df['TimeElapsed_sec'], self.df[col], alpha=0.7, linewidth=1)
                    plt.xlabel('Time Elapsed (seconds)')
                    # Set minor grid lines at 50-second intervals
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
                    # For y-axis, calculate appropriate minor tick spacing based on data range
                    y_data = self.df[col].dropna()
                    if len(y_data) > 0:
                        y_range = y_data.max() - y_data.min()
                        # Use 50 as minor spacing, but adjust if range is very small
                        y_minor_spacing = 50 if y_range > 100 else max(1, y_range / 20)
                        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor_spacing))
                else:
                    plt.plot(self.df.index, self.df[col], alpha=0.7, linewidth=1)
                    plt.xlabel('Sample Index')
                    # Set minor grid lines at 50-sample intervals
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
                    # For y-axis, calculate appropriate minor tick spacing based on data range
                    y_data = self.df[col].dropna()
                    if len(y_data) > 0:
                        y_range = y_data.max() - y_data.min()
                        # Use 50 as minor spacing, but adjust if range is very small
                        y_minor_spacing = 50 if y_range > 100 else max(1, y_range / 20)
                        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor_spacing))
                plt.ylabel(col)
                plt.title(f'Time Series: {col}')
                # Enable both major and minor grid lines
                ax.grid(True, which='major', alpha=0.5, linewidth=1)
                ax.grid(True, which='minor', alpha=0.2, linewidth=0.5, linestyle='--')
        
        # Distribution histograms - shows data shape and outliers
        for i, col in enumerate(numeric_cols, 5):
            if col in self.df.columns:
                ax = plt.subplot(5, 2, i)
                self.df[col].hist(bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Distribution: {col}')
                plt.grid(True, alpha=0.3)
        
        # Box plots - compare variability across channels
        ax = plt.subplot(5, 2, 9)
        box_data = [self.df[col].dropna() for col in numeric_cols if col in self.df.columns]
        plt.boxplot(box_data, labels=[col for col in numeric_cols if col in self.df.columns])
        plt.ylabel('Value')
        plt.title('Box Plot Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Stability trend visualization - first vs last quartile comparison
        ax = plt.subplot(5, 2, 10)
        if 'ADC' in self.df.columns:
            q1_mean = self.df['ADC'].iloc[:len(self.df)//4].mean()
            q4_mean = self.df['ADC'].iloc[-len(self.df)//4:].mean()
            plt.bar(['First Quartile', 'Last Quartile'], [q1_mean, q4_mean], 
                   color=['blue', 'red'], alpha=0.7)
            plt.ylabel('Mean ADC Value')
            plt.title('Stability Trend: First vs Last Quartile')
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.filename.replace('.csv', '_analysis_plots.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {output_file}")
        plt.close()
        
        return output_file
    
    def generate_report(self):
        """Generate comprehensive text report"""
        print("\n" + "="*60)
        print("8. GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report_file = self.filename.replace('.csv', '_analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"SENSOR STABILITY ANALYSIS REPORT\n")
            f.write(f"File: {self.filename}\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Data Quality
            f.write("1. DATA QUALITY & INTEGRITY\n")
            f.write("-"*80 + "\n")
            if 'data_quality' in self.results:
                dq = self.results['data_quality']
                f.write(f"Total Rows: {dq['total_rows']}\n")
                f.write(f"Outliers (IQR): {sum([v['count'] for v in dq['outliers_iqr'].values()])}\n")
                
                # Isolation Forest results
                if 'isolation_forest_results' in dq and dq['isolation_forest_results']:
                    iso = dq['isolation_forest_results']
                    anomaly_pct = iso['anomaly_percentage']
                    if anomaly_pct < 5:
                        interpretation = "Very few anomalies detected - data quality is excellent"
                    elif anomaly_pct < 10:
                        interpretation = "Low anomaly rate - data quality is good (within expected contamination parameter)"
                    elif anomaly_pct < 20:
                        interpretation = "Moderate anomaly rate - some unusual patterns detected (may need investigation)"
                    else:
                        interpretation = "High anomaly rate - significant unusual patterns detected (data quality concern)"
                    f.write(f"\nIsolation Forest Anomaly Detection:\n")
                    f.write(f"  Anomalies detected: {iso['anomalies_detected']} ({anomaly_pct:.2f}%)\n")
                    f.write(f"  Samples analyzed: {iso['total_samples_analyzed']}\n")
                    f.write(f"  Contamination parameter: {iso['contamination_parameter']}\n")
                    f.write(f"  Interpretation: {interpretation}\n")
                f.write("\n")
            
            # Statistical Characterization
            f.write("2. STATISTICAL CHARACTERIZATION\n")
            f.write("-"*80 + "\n")
            if 'statistical_characterization' in self.results:
                for col, stats_dict in self.results['statistical_characterization'].items():
                    if isinstance(stats_dict, dict) and 'mean' in stats_dict:
                        f.write(f"\n{col}:\n")
                        f.write(f"  Mean: {stats_dict['mean']:.4f}\n")
                        f.write(f"  Std: {stats_dict['std']:.4f}\n")
                        f.write(f"  CV: {stats_dict['coefficient_of_variation']:.2f}%\n")
                        f.write(f"  Skewness: {stats_dict['skewness']:.4f}\n")
                        f.write(f"  Kurtosis: {stats_dict['kurtosis']:.4f}\n")
                        
                        # Shapiro-Wilk test results
                        if 'shapiro_wilk' in stats_dict:
                            sw = stats_dict['shapiro_wilk']
                            interpretation = "Data appears normally distributed" if sw['is_normal'] else "Data does NOT follow normal distribution (may have skewness/kurtosis issues)"
                            f.write(f"\n  Shapiro-Wilk Test:\n")
                            f.write(f"    Statistic: {sw['statistic']:.4f}\n")
                            f.write(f"    p-value: {sw['p_value']:.4f}\n")
                            f.write(f"    Interpretation: {interpretation}\n")
                        
                        # Jarque-Bera test results
                        if 'jarque_bera' in stats_dict:
                            jb = stats_dict['jarque_bera']
                            interpretation = "Data appears normally distributed" if jb['is_normal'] else "Data does NOT follow normal distribution (skewness/kurtosis mismatch)"
                            f.write(f"\n  Jarque-Bera Test:\n")
                            f.write(f"    Statistic: {jb['statistic']:.4f}\n")
                            f.write(f"    p-value: {jb['p_value']:.4f}\n")
                            f.write(f"    Interpretation: {interpretation}\n")
                f.write("\n")
            
            # Stability Metrics
            f.write("3. STABILITY METRICS\n")
            f.write("-"*80 + "\n")
            if 'stability_metrics' in self.results:
                for col, metrics in self.results['stability_metrics'].items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  Drift Rate: {metrics['drift_rate_per_sample']:.6f} per sample\n")
                    f.write(f"  CV: {metrics['coefficient_of_variation_pct']:.4f}%\n")
                    f.write(f"  Stability Index: {metrics['stability_index']:.2f}\n")
                    f.write(f"  Long-term Drift: {metrics['long_term_drift_pct']:.4f}%\n")
                f.write("\n")
            
            # Time Series Analysis
            f.write("4. TIME SERIES ANALYSIS\n")
            f.write("-"*80 + "\n")
            if 'time_series_analysis' in self.results:
                for col, analysis in self.results['time_series_analysis'].items():
                    f.write(f"\n{col}:\n")
                    
                    # ADF test results
                    if 'adfuller' in analysis:
                        adf = analysis['adfuller']
                        if adf['is_stationary']:
                            interpretation = "Series is STATIONARY - good for sensor stability (no significant drift detected)"
                        else:
                            interpretation = "Series is NON-STATIONARY - indicates drift, trend, or systematic changes (potential stability concern)"
                        f.write(f"  ADF Test (Augmented Dickey-Fuller):\n")
                        f.write(f"    Statistic: {adf['statistic']:.4f}\n")
                        f.write(f"    p-value: {adf['p_value']:.4f}\n")
                        f.write(f"    Interpretation: {interpretation}\n")
                    
                    # KPSS test results
                    if 'kpss' in analysis:
                        kpss_test = analysis['kpss']
                        if kpss_test['is_stationary']:
                            interpretation = "Series is STATIONARY - good for sensor stability (statistical properties constant over time)"
                        else:
                            interpretation = "Series is NON-STATIONARY - indicates drift or trend (potential stability concern)"
                        f.write(f"  KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin):\n")
                        f.write(f"    Statistic: {kpss_test['statistic']:.4f}\n")
                        f.write(f"    p-value: {kpss_test['p_value']:.4f}\n")
                        f.write(f"    Interpretation: {interpretation}\n")
                    
                    # Ljung-Box test results
                    if 'ljung_box' in analysis:
                        lb = analysis['ljung_box']
                        if lb['statistic'] is not None:
                            if not lb['has_autocorr']:
                                interpretation = "No significant autocorrelation - good (sensor responds independently to each measurement)"
                            else:
                                interpretation = "Significant autocorrelation detected - sensor has memory effects (may indicate slow response or filtering)"
                            f.write(f"  Ljung-Box Test:\n")
                            f.write(f"    Statistic: {lb['statistic']:.4f}\n")
                            f.write(f"    p-value: {lb['p_value']:.4f}\n")
                            f.write(f"    Interpretation: {interpretation}\n")
                            f.write(f"    Lags tested: {lb['lags_tested']}\n")
                f.write("\n")
            
            # ML Readiness
            f.write("5. ML READINESS ASSESSMENT\n")
            f.write("-"*80 + "\n")
            if 'ml_readiness' in self.results:
                ml = self.results['ml_readiness']
                f.write(f"Sample Count: {ml['sample_count']}\n")
                f.write(f"Samples per Feature: {ml['samples_per_feature']:.1f}\n")
                f.write("\n")
            
            # Sensor Performance
            f.write("6. SENSOR PERFORMANCE\n")
            f.write("-"*80 + "\n")
            if 'sensor_performance' in self.results:
                for metric, value in self.results['sensor_performance'].items():
                    if isinstance(value, dict) and 'snr_db' in value:
                        f.write(f"{metric}: SNR = {value['snr_db']:.2f} dB\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Report saved to: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Execute all analysis steps"""
        print("\n" + "="*80)
        print(f"COMPREHENSIVE SENSOR STABILITY ANALYSIS")
        print(f"File: {self.filename}")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.data_quality_analysis()
        self.statistical_characterization()
        self.stability_metrics()
        self.time_series_analysis()
        self.ml_readiness_assessment()
        self.sensor_performance_metrics()
        
        # Generate outputs
        self.generate_visualizations()
        self.generate_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        return self.results


if __name__ == "__main__":
    # Find all CSV files in RawDryRunData directory
    data_dir = "RawDryRunData"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}/")
        exit(1)
    
    # Sort files for consistent processing order
    csv_files.sort()
    
    print("="*80)
    print("MASTER SENSOR DATA ANALYSIS")
    print("="*80)
    print(f"\nFound {len(csv_files)} CSV file(s) to analyze:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    print("\n" + "="*80)
    
    # Process each file
    results_summary = []
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"\n\n{'='*80}")
        print(f"PROCESSING FILE {idx}/{len(csv_files)}: {os.path.basename(csv_file)}")
        print(f"{'='*80}\n")
        
        try:
            analyzer = SensorStabilityAnalyzer(csv_file)
            results = analyzer.run_complete_analysis()
            results_summary.append({
                'file': os.path.basename(csv_file),
                'status': 'SUCCESS',
                'rows': len(analyzer.df) if analyzer.df is not None else 0
            })
        except Exception as e:
            print(f"\nERROR processing {os.path.basename(csv_file)}: {str(e)}")
            results_summary.append({
                'file': os.path.basename(csv_file),
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Print final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nTotal files processed: {len(csv_files)}")
    successful = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    failed = len(results_summary) - successful
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print("\nSuccessfully analyzed files:")
        for r in results_summary:
            if r['status'] == 'SUCCESS':
                print(f"  ✓ {r['file']} ({r['rows']} rows)")
    
    if failed > 0:
        print("\nFailed files:")
        for r in results_summary:
            if r['status'] == 'FAILED':
                print(f"  ✗ {r['file']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

