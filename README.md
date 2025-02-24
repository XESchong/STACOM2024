# STACOM2024
This is the repo for the STACOM 2024 paper "Improved 3D Whole Heart Geometry from Sparse CMR Slices".

For the toy case, we generated synthetic DICOM files (SAXs + LAXs) based on the dense segmentation, where the header information of each DICOM file uses a template from the Sunnybrook Cardiac Data(https://www.cardiacatlas.org/sunnybrook-cardiac-data/).

For running the slice shifting algorithm:
1. Execute SSA.py with the toy example
2. It will automatically calculate the in-plane shift based on the DICOM files from input data
3. We have also included 3D sparse volume results in nifti format that show before and after SSA comparisons
4. When applying SSA to realistic data, we have discovered a practical limitation: if the moving image (particularly near the apex area) contains insufficient information, it may cause unexpected behavior in SSA performance. In such cases, we recommend simply removing that slice from the SSA running
5. SSA is also compatible with nifti file inputs

For running the spatial transformer network:
1. You will need to update the paths in both the "pred" section and main part of STN.py
2. Make sure your input is one-hot encoded
3. Given the input, We offer two different versions for the STN, i.e., Sparse STN (SSTN) and Dense STN (DSTN). 

For running the label transformer network:
1. Modify all paths in the "eval_model" section of LTN.py
2. Create a new empty folder to store LTN's output
3. Ensure your input is one-hot encoded
4. Pretrained model: https://drive.google.com/file/d/10t4xQAuO0I69Czrt89-4ekJrP1POoVJ4/view?usp=drive_link

