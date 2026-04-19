# Red Giant and Red Clump Classification Machine Learning using Stellar Spectra from APOGEE

### INTRODUCTION
The goal of this project is to see if we can use machine learning to accurately differentiate between Red Giant and Red Clump stars using infrared spectra data from APOGEE (https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/). 

### RESOURCES
- Data from APOGEE: https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/
- Bitmasks for data cleaning: https://www.sdss4.org/dr17/irspec/apogee-bitmasks/
- Filtering values and reference for classification: https://ui.adsabs.harvard.edu/abs/2014ApJ...790..127B/abstract

### FILE KEY
- Data Cleaning: data_cleaning.ipynb
  Files used:
  - Raw Data: allField-dr17.fits
  - Initial Processing: allstar_processed_fixed.csv
  - Final Datafile: cleaned_data.csv
- Metadata Exploratory Data Analysis: metadata_eda.ipynb
  Files used:
  - Data: cleaned_data.csv
  Graphs made:
  - Effective_temperature_graph.png
  - log_surface_gravity_graph.png
  - Signal_to_noise_graph.png
  - HR_density_heatmap_all
  - HR_density_heatmap_RBG
  - HR_rc_vs_rbg
- Single Spectra EDA: single_spectra_analysis.ipynb
  Files used:
  - Data: apStar-dr17-2M00000002+7417074.fits
  Graphs made:
  - single_spectra.png
- Getting Data: getting_data.ipynb
  Files used:
  - Data: cleaned_data.csv, cleaned_data_shuffled.csv
  - Output files: processed_spectra.npy, processed_labels.npy
Supervised Machine Learning Model: supervised_model.ipynb
- Data: processed_spectra.npy, processed_labels.npy
- Graphs made:
  - supervised_model_accuracy.png
  - supervised_confusion_matrix.png
  - supervised_hr_diagram
  - supervised_spectra_comparison
 Unsupervised Machine Learning Model:

### DATA CLEANING

