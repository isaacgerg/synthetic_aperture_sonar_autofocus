# Simple autofocus technique for syntehtic aperture sonar (SAS)

This repository contains code to autofocus single-look complex (SLC) imagery from a SAS sonar.  The algorithm works by optimizing an attribute of the resulting output image over the phase correction space (1D azimuth FFT of SLC).  

Three optimization functions are here:
1. Maximization of mean-normalized variance of the output magnitude image.  
2. Minimization of the entropy of the output magnitude image.
3. Minimization of the -ln(x+b) of the output magnitude image.  See "Optimal Sharpness Function for SAR Autofocus"

## Data attribution

Thanks go to J. Tory Cobb for curating this dataset. Please credit NSWC Panama City Division in any publication using this data.

## References

T J Schulz. "Optimal Sharpness Function for SAR Autofocus." https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4035715

J Fienup and J Miller. "Aberration correction by maximizing generalizedsharpness metrics." https://labsites.rochester.edu/fienup/wp-content/uploads/2019/07/JOSAA03_GenSharpness.pdf
