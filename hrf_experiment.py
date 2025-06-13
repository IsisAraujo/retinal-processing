"""
Main experimental pipeline for HRF illumination correction study
Implements the complete workflow for academic paper generation
"""

import os
import cv2
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from hrf_core import ProcessingResult, OphthalmicMetrics
from hrf_methods import CLAHEMethod, SingleScaleRetinex, MultiScaleRetinex, MultiScaleRetinexColorRestoration
from hrf_metrics import MetricsCalculator
from hrf_analysis import StatisticalAnalyzer, AcademicVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HRFExperiment:
    """
    Main experimental pipeline for comparative analysis of illumination correction methods
    """

    def __init__(self, dataset_path: str, output_dir: str = "results"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.methods = {
            'CLAHE': CLAHEMethod(),
            'SSR': SingleScaleRetinex(),
            'MSR': MultiScaleRetinex(),
            'MSRCR': MultiScaleRetinexColorRestoration()
        }
        self.results = []

        # Create output directories
        self._create_output_structure()

    def _create_output_structure(self):
        """Create directory structure for results"""
        directories = ['figures', 'tables', 'data', 'logs']
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def load_dataset(self, sample_size: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """Load HRF dataset images"""
        supported_formats = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG")
        image_files = [f for f in self.dataset_path.iterdir()
                      if f.suffix.lower() in supported_formats]

        if sample_size:
            image_files = image_files[:sample_size]

        images = []
        for image_path in sorted(image_files):
            img = cv2.imread(str(image_path))
            if img is not None:
                # Validate HRF resolution
                if img.shape[0] >= 2000 and img.shape[1] >= 3000:
                    images.append((image_path.stem, img))
                    logger.info(f"Loaded: {image_path.name} - Shape: {img.shape}")
                else:
                    logger.warning(f"Skipped (low resolution): {image_path.name}")

        logger.info(f"Loaded {len(images)} valid HRF images")
        return images

    def process_single_image(self, image_id: str, image: np.ndarray) -> List[Dict]:
        """Process single image with all methods"""
        results = []

        # Store original for comparison
        results.append({
            'image_id': image_id,
            'method': 'original',
            'image': image,
            'metrics': {},
            'processing_time_ms': 0
        })

        # Process with each method
        for method_name, method in self.methods.items():
            logger.info(f"Processing {image_id} with {method_name}")

            # Time the processing
            start_time = time.time()
            corrected_image = method.process(image.copy())
            processing_time_ms = (time.time() - start_time) * 1000

            # Calculate metrics
            metrics = MetricsCalculator.calculate_all_metrics(image, corrected_image)

            result = {
                'image_id': image_id,
                'method': method_name,
                'image': corrected_image,
                'metrics': metrics.to_dict(),
                'processing_time_ms': processing_time_ms
            }

            results.append(result)

            logger.info(f"  {method_name}: Contrast={metrics.contrast_ratio:.3f}, "
                       f"Vessels={metrics.vessel_clarity_index:.5f}, "  # Alterado para 5 casas decimais
                       f"Time={processing_time_ms:.1f}ms")

        return results

    def run_experiment(self, sample_size: Optional[int] = None):
        """Run complete experimental pipeline"""
        logger.info("Starting HRF illumination correction experiment")

        # Load dataset
        images = self.load_dataset(sample_size)
        if not images:
            raise ValueError("No valid images found in dataset")

        # Process all images
        total_start_time = time.time()

        for image_id, image in images:
            image_results = self.process_single_image(image_id, image)
            self.results.extend(image_results)

        total_time = time.time() - total_start_time
        logger.info(f"Processing completed in {total_time:.2f} seconds")

        # Save raw results
        self._save_raw_results()

        # Perform statistical analysis
        self._perform_analysis()

        # Generate visualizations
        self._generate_visualizations()

        # Create summary report
        self._generate_report()

    def _save_raw_results(self):
        """Save processing results to disk"""
        # Save metrics data (without images)
        metrics_data = []
        for result in self.results:
            if result['method'] != 'original':
                metrics_data.append({
                    'image_id': result['image_id'],
                    'method': result['method'],
                    'metrics': result['metrics'],
                    'processing_time_ms': result['processing_time_ms']
                })

        json_path = self.output_dir / 'data' / 'metrics_results.json'
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Saved metrics data to {json_path}")

    def _perform_analysis(self):
        """Perform statistical analysis"""
        analyzer = StatisticalAnalyzer()

        # Create dataframe
        df = analyzer.results_dataframe([r for r in self.results if r['method'] != 'original'])
        df.to_csv(self.output_dir / 'data' / 'results_dataframe.csv', index=False)

        # Get metric names
        metrics = ['contrast_ratio', 'vessel_clarity_index', 'illumination_uniformity',
                  'edge_preservation_index', 'microaneurysm_visibility']

        # Normality tests
        normality_results = analyzer.perform_normality_tests(df, metrics)

        # Homoscedasticity tests
        homoscedasticity_results = analyzer.perform_homoscedasticity_test(df, metrics)

        # Comparison tests
        comparison_results = analyzer.perform_comparison_tests(
            df, metrics, normality_results, homoscedasticity_results
        )

        # Save statistical results
        stats_results = {
            'normality': normality_results,
            'homoscedasticity': homoscedasticity_results,
            'comparisons': comparison_results
        }

        stats_path = self.output_dir / 'data' / 'statistical_analysis.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_results, f, indent=2)

        logger.info("Statistical analysis completed")

        # Generate statistical summary table
        visualizer = AcademicVisualizer()
        visualizer.create_statistical_summary_table(comparison_results, self.output_dir / 'tables')

    def _generate_visualizations(self):
        """Generate all visualizations"""
        analyzer = StatisticalAnalyzer()
        visualizer = AcademicVisualizer()

        # Create dataframe
        df = analyzer.results_dataframe([r for r in self.results if r['method'] != 'original'])

        # Generate plots
        logger.info("Generating visualizations...")

        # 1. Visual comparison grid
        visualizer.create_comparison_grid(self.results, self.output_dir / 'figures')

        # 2. Metrics boxplots
        visualizer.create_metrics_boxplots(df, self.output_dir / 'figures')

        # 3. Performance scatter plot
        analyzer.create_performance_scatter(df, self.output_dir / 'figures')

        # 4. Correlation heatmap
        analyzer.create_heatmap(df, self.output_dir / 'figures')

        logger.info("Visualizations completed")

    def _generate_report(self):
        """Generate final report summary"""
        # Load statistical results
        with open(self.output_dir / 'data' / 'statistical_analysis.json', 'r') as f:
            stats_results = json.load(f)

        # Create summary report
        report = f"""# HRF Illumination Correction Analysis Report

## Experimental Setup
- Dataset: HRF (High-Resolution Fundus) Image Database
- Images processed: {len(set(r['image_id'] for r in self.results if r['method'] != 'original'))}
- Methods evaluated: CLAHE, SSR, MSR, MSRCR
- Metrics: 5 ophthalmology-specific quality measures

## Key Findings

### Statistical Significance
"""

        # Add significant findings
        for metric, results in stats_results['comparisons'].items():
            if results['significant']:
                report += f"- **{metric}**: Significant differences found ({results['test']}, p={results['p_value']:.4f})\n"

        report += """
## Outputs Generated
- Statistical analysis: /data/statistical_analysis.json
- Metrics dataframe: /data/results_dataframe.csv
- Visualizations: /figures/
- Tables: /tables/

## Next Steps
1. Review statistical significance of findings
2. Select best-performing method based on clinical relevance
3. Prepare manuscript following journal guidelines
"""

        report_path = self.output_dir / 'ANALYSIS_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report generated: {report_path}")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="HRF Illumination Correction Comparative Study"
    )
    parser.add_argument('dataset_path', help='Path to HRF dataset directory')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--sample_size', type=int, help='Limit to N images for testing')

    args = parser.parse_args()

    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset path not found: {args.dataset_path}")
        return

    # Run experiment
    experiment = HRFExperiment(args.dataset_path, args.output_dir)
    experiment.run_experiment(args.sample_size)

    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()


