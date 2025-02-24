# STACOM2024
This is the repo for the STACOM 2024 paper "Improved 3D Whole Heart Geometry from Sparse CMR Slices".

For the toy case, we generated synthetic DICOM files (SAXs + LAXs) based on the dense segmentation, where the header information of each DICOM file uses a template from the Sunnybrook Cardiac Data(https://www.cardiacatlas.org/sunnybrook-cardiac-data/).

For running the slice shifting algorithm:

1. Execute SSA.py with the toy example
2. It will automatically calculate the in-plane shift based on the DICOM files from input data
3. We have also included 3D sparse volume results in nifti format that show before and after SSA comparisons
