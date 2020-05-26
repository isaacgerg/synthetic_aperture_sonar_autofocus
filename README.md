# Simple autofocus technique for synthetic aperture sonar (SAS)
This repository contains code to autofocus single-look complex (SLC) imagery from a SAS sonar.  The algorithm works by optimizing an attribute of the resulting output image over the phase correction space (1D azimuth FFT of SLC).  

Three optimization functions are here:
1. Maximization of mean-normalized variance of the output magnitude image.  
2. Minimization of the entropy of the output magnitude image.
3. Minimization of the -ln(x+b) of the output magnitude image.  See "Optimal Sharpness Function for SAR Autofocus"

The optimization is carried in one of two ways:1
1. Deriviatives are estimated from finite differences (BFGS, Simplex)
2. Derivatives are automatically computed using autodiff via chain rule (Tensorflow SGD)

## Data attribution
This work uses the Synthetic Aperture Sonar Seabed Environment Dataset (SASSED) dataset. Thanks go to J. Tory Cobb for curating this dataset. Please credit NSWC Panama City Division in any publication using this data.

Approved for Public Release; distribution is unlimited.

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

Approved for Public Release; distribution is unlimited.

## References

T J Schulz. "Optimal Sharpness Function for SAR Autofocus." https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4035715

J Fienup and J Miller. "Aberration correction by maximizing generalizedsharpness metrics." https://labsites.rochester.edu/fienup/wp-content/uploads/2019/07/JOSAA03_GenSharpness.pdf
