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
