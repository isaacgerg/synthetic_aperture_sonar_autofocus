# Simple Autofocus Techniques for Synthetic Aperture Sonar (SAS)
This repository contains two categories of methods to autofocus single-look complex (SLC) imagery from a SAS sonar.  The first algorithm works by optimizing a contrast metric of the resulting output image over the phase correction space (1D azimuth FFT of SLC).  The second algorithm is phase gradient autofocus (PGA) adapted from the RITSAR Python toolbox and modified to use the maximimum likelyhood (ML) kernel of:

 Jakowatz, Charles V., and Daniel E. Wahl. "Eigenvector method for maximum-likelihood 
   estimation of phase errors in synthetic-aperture-radar imagery." 
   JOSA A 10.12 (1993): 2539-2546.

The PGA autofocus has a flag to also compute the "shadow PGA" method of Prater, et al.

J Prater, D Bryner, and S Synnes. "SHADOW BASED PHASE GRADIENT AUTOFOCUS FOR SYNTHETIC APERTURE SONAR." 5th annual Institute of Acoustics SAS/SAR Conference. Lerici, Italy. 2023.

For the first class of algorithms, three objectives functions you can focus the imagery with are:
1. Maximization of mean-normalized variance of the output magnitude image.  
2. Minimization of the entropy of the output magnitude image.
3. Minimization of the -ln(x+b) of the output magnitude image.  See "Optimal Sharpness Function for SAR Autofocus"

The optimization can be carried in one of two ways:
1. Deriviatives are estimated from finite differences (for methods: BFGS, Simplex). You can use Jax or Tensorflow with scipy.optimize also to use the analytic deriviatives.
2. Derivatives are automatically computed using autodiff via chain rule (for method Tensorflow SGD)

## Example Output (Meteric Based Methods)
Before and after applying autofocus

![Before and After Autofocus](https://raw.githubusercontent.com/isaacgerg/synthetic_aperture_sonar_autofocus/master/output_autofocus_before_and_after_44.png)

The phase correction applied to autofocus the image.
![Phase Correction](https://raw.githubusercontent.com/isaacgerg/synthetic_aperture_sonar_autofocus/master/output_phase_correction_44.png)

## Example Output (Phase Gradient Autofocus)
Canonical PGA
![image_26_shadow_pga_False](https://github.com/isaacgerg/synthetic_aperture_sonar_autofocus/assets/11971499/ac7d0896-9d6d-4cb3-abda-46131c201ffd)

Shadow PGA
![image_26_shadow_pga_True](https://github.com/isaacgerg/synthetic_aperture_sonar_autofocus/assets/11971499/cb98090f-3b2a-4cef-93d5-d4713709e9fc)

Sept 21, 2023 - I believe shadow PGA "works" because you are modeling the signal as -1 * direct delta function + DC offset.  This is essentially the same model, mathematically speaking, as original PGA which models the signal as a dirac delta function.  In both cases, convolving the PSF with either of these function behaves as such to characterize the PSF with the shadow PGA inducing a -1 multiplier in the result which is exactly what we see with shadow PGA. The signal model with shadow PGA can also be seen when you examine the incoherent integration of the signal after center shifting the min of each range bin.  The resulting sum will look very similar, but with a -1 multiplier, to that when you incoherently sum the signal after center shifting the max of each range bin.


## Data Attribution
--- Approved for Public Release; distribution is unlimited. ---

Dataset title: Synthetic Aperture Sonar Seabed Environment Dataset (SASSED)

Date: June 2018

Description:
This dataset contains 129 complex-valued, single (high frequency) channel, 1001x1001 pixel, synthetic aperture sonar snippets of various seafloor texture types. Each snippet contains one or more seabed environments, e.g., hardpack sand, mud, sea grass, rock, and sand ripple. 

For each snippet there is a corresponding hand-segmented and -labeled "mask" image. The labels should not be interpreted as the ground truth for specific seafloor types. The labels were not verified by visual inspection of the actual seafloor environments or by any other method. Instead, interpret the labels as groupings of similar seafloor textures. 
Example code for preprocessing the data is included.

The data is stored in hdf5 format. The SAS data is stored under the hdf5 dataset 'snippets', and the hand-segmented labels are stored under 'labels'. For information on how to read hdf5 data, please visit one of the following websites: 
(general) https://support.hdfgroup.org/HDF5/
(python)  https://www.h5py.org 


Acknowledgements: 
Thanks go to J. Tory Cobb for curating this dataset. Please credit NSWC Panama City Division in any publication using this data.

Past Usage:
Cobb, J. T., & Zare, A. (2014). Boundary detection and superpixel formation in synthetic aperture sonar imagery. Proceedings of the Institute of Acoustics, 36(Pt 1).

--- Approved for Public Release; distribution is unlimited. --- 

## Data Download
Data available for download

Readme.txt: https://gergltd.com/data/sassed/readme.txt

Data: https://gergltd.com/data/sassed/sassed.h5

## References

T J Schulz. "Optimal Sharpness Function for SAR Autofocus." https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4035715

J Fienup and J Miller. "Aberration correction by maximizing generalizedsharpness metrics." https://labsites.rochester.edu/fienup/wp-content/uploads/2019/07/JOSAA03_GenSharpness.pdf
