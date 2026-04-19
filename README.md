# Red Giant and Red Clump Classification Machine Learning using Stellar Spectra from APOGEE

### INTRODUCTION
The goal of this project is to see if we can use machine learning to accurately differentiate between Red Giant and Red Clump stars using infrared spectra data from APOGEE (https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/). The scheme from the following paper, Bovy etl (https://ui.adsabs.harvard.edu/abs/2014ApJ...790..127B/abstract) to extract the red gaint and red clump star spectra from the data. After cleaning and exploring the data, Convolutional Neural Networks and K-Means were used to preform both supervised and unsupervised machine learning training to achieve this task.

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
The following data cleaning scheme was used.

Flags: https://www.sdss4.org/dr17/irspec/apogee-bitmasks/

Filter out STAR_BAD_BIT, and TELLURIC_BIT from columns STARFLAG, APOGEE_TARGET2, and APOGEE2_TARGET2. COMMISS column should be 0, and RV_TEFF and RV_LOGG columns cannot be invalid values.

After these cuts, over 200,000 sample stars were left. It was added Signal to Noise Ratio needed to be greater than 500. This left just over 22,000 samples.

To isolate the Red Giant and Red Clump stars, the following was used:

Bovy Paper Reference: https://ui.adsabs.harvard.edu/abs/2014ApJ...790..127B/abstract

- 1.0 < RV_LOGG < 3.5
- 3500 < RV_TEFF < 5200

The parallax Signal to Noise Ratio was calculated using the columns GAIAEDR3_PARALLAX divided by GAIAEDR3_PARALLAX_ERROR.

- Parallax Signal to Noise Ratio > 10
- 0 < GAIAEDR3_PARALLAX < 1.5

### METADATA EXPLORATORY DATA ANALYSIS
This notebook's role was to explore the spectra data and distribution before using it for training. Distributions of the data are as followed.
<img width="580" height="455" alt="Signal_to_Noise_graph" src="https://github.com/user-attachments/assets/5e7f3063-df16-46c5-ae6c-abddbf7b465f" />

<img width="579" height="455" alt="Effective_temperature_graph" src="https://github.com/user-attachments/assets/8a1eada8-a07d-4ce7-bf05-c7ef4ef160d1" />

<img width="571" height="455" alt="log_surface_gravity_graph" src="https://github.com/user-attachments/assets/2ed6198f-8fe3-4984-bad4-807072558959" />

HR Diagrams using Log Effective Gravity and Log Effective Temperature
<img width="558" height="455" alt="HR_density_heatmap_all" src="https://github.com/user-attachments/assets/27e5874e-e1c1-4098-b8b4-24bbf057662b" />

<img width="571" height="455" alt="HR_density_heatmap_RBG" src="https://github.com/user-attachments/assets/47473694-1612-427d-a8fa-eb6ed0ccfbee" />

Using the scheme to filter Red Clump stars from the Bovy paper:

t_ref = -382.5 * df_shuffled['RV_FEH'] + 4607

delta_t = df_shuffled['RV_TEFF'] - t_ref

logg_upper_bound = (0.0018 * delta_t) + 2.5

rc_mask = df_shuffled["RV_LOGG"].between(1.8, logg_upper_bound)

df_shuffled["Stellar_type"] = np.where(rc_mask, "Red Clump", "Red Giant"),

we get the following distribution of Red Giant and Red Clump stars in the HR Diagram.

<img width="567" height="455" alt="HR_rc_vs_rbg" src="https://github.com/user-attachments/assets/e140e31b-d9a6-4f44-b38e-edfef5f3d039" />
