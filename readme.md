# SAS Autofocus Tools

This repository contains methods to autofocus single-look complex (SLC) imagery from Synthetic Aperture Sonar (SAS). It includes two main categories of autofocus algorithms and a visualization tool for comparing their results.

## Algorithms

### 1. Contrast Optimization
The first algorithm works by optimizing a contrast metric of the resulting output image over the phase correction space (1D azimuth FFT of SLC).

### 2. Phase Gradient Autofocus (PGA)
The second algorithm is Phase Gradient Autofocus (PGA), adapted from the RITSAR Python toolbox. This implementation has been modified to use the Maximum Likelihood (ML) kernel described in:

> Jakowatz, Charles V., and Daniel E. Wahl. "Eigenvector method for maximum-likelihood estimation of phase errors in synthetic-aperture-radar imagery." *JOSA A* 10.12 (1993): 2539-2546.

Additionally, the PGA implementation includes support for "Shadow PGA" as described in:

> J. Prater, D. Bryner, and S. Synnes. "SHADOW BASED PHASE GRADIENT AUTOFOCUS FOR SYNTHETIC APERTURE SONAR." *5th annual Institute of Acoustics SAS/SAR Conference*. Lerici, Italy. 2023.

## Visualization Tool (`pga_shadow_pga_demo.py`)

This script provides visualization tools for comparing standard PGA and Shadow PGA results.

### Features
- Processes SAS images using both standard PGA and Shadow PGA
- Generates side-by-side comparisons of:
  - Original image
  - PGA-processed result
  - Phase error plots
  - RMS convergence plots
- Creates detailed comparisons of the brightest regions (64x64 pixel patches)
- Supports batch processing of multiple images

### Requirements
- Python 3.x
- NumPy
- SciPy
- OpenCV (cv2)
- Matplotlib
- PIL (Pillow)
- RITSAR
- GDAL

### Usage

```python
# Example usage
import sas_tools

# Load and process a single SLC image
input_file = "path/to/your/test.npz"
output_dir = "processed_images"

# Run the demo script
python pga_shadow_pga_demo.py
```

## Comparison of Standard and Shadow PGA Performance

Standard PGA and Shadow PGA demonstrate distinct characteristics and performance differences across various aspects of autofocus processing. Here are the key observations from comparing their performance:

### Convergence Behavior

Standard PGA shows smoother and more consistent convergence in RMS plots, typically converging within 5-10 iterations. The RMS error generally decreases monotonically, and final RMS values tend to be lower than Shadow PGA.

In contrast, Shadow PGA exhibits more oscillatory convergence behavior and often requires more iterations (20-30). The RMS plots show higher variability between iterations. While final RMS values are generally higher, this method may capture different features that standard PGA misses.

### Phase Error Characteristics

Standard PGA produces smoother phase error curves with generally smaller magnitudes and shows consistent behavior across different types of scenes. 

Shadow PGA generates more variable phase error profiles and often shows larger magnitude phase corrections. The phase error profiles appear more sensitive to scene content, which can be both an advantage and disadvantage depending on the application.

### Image Quality Impact

Standard PGA works particularly well on scenes with strong reflectors and maintains overall image contrast, though it can sometimes blur shadow regions. 

Shadow PGA better preserves shadow region details and is particularly effective on rippled seabed scenes. While it may introduce more noise in bright regions, it often provides better detail preservation in areas with significant shadow content.

### Scene-Dependent Performance

On ripple fields, both methods effectively focus ripple patterns, but Shadow PGA better preserves ripple contrast in shadow regions. Standard PGA provides more consistent focusing across the full scene.

For mixed terrain, Shadow PGA shows advantages in areas with significant shadow content, while Standard PGA performs better on uniformly bright regions. Their complementary strengths suggest potential value in combining approaches.

### Computational Considerations

Shadow PGA typically requires more iterations and has a higher computational cost due to slower convergence. This suggests it may benefit from adaptive iteration limits in practical applications.

### Recommendations for Use

Standard PGA is preferred for scenes with strong reflectors, applications requiring faster processing, and cases where consistent focusing across the scene is a priority.

Shadow PGA is better suited for scenes with significant shadow content, applications where shadow region detail is critical, and cases where maximum shadow contrast is desired.

### Future Research Directions

Potential areas for future research include hybrid approaches that combine standard and shadow PGA results, development of adaptive methods that switch between approaches based on local scene content, and exploration of weighted combinations of both methods.

Additionally, there are opportunities to improve Shadow PGA convergence behavior, develop better termination criteria, and investigate scene-dependent parameter selection.

This comparison suggests that while both methods have their strengths, the choice between them should be guided by the specific requirements of the application and the characteristics of the scene being processed.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Include your license information here]

## Citation
If you use this code in your research, please cite:
```bibtex
@inproceedings{prater2023shadow,
  title={Shadow Based Phase Gradient Autofocus for Synthetic Aperture Sonar},
  author={Prater, J and Bryner, D and Synnes, S},
  booktitle={5th annual Institute of Acoustics SAS/SAR Conference},
  year={2023},
  address={Lerici, Italy}
}
```

## Acknowledgments
- Original RITSAR Python toolbox contributors