# Red Giant and Red Clump Classification Machine Learning using Stellar Spectra from APOGEE

### INTRODUCTION
The goal of this project is to see if we can use machine learning to accurately differentiate between Red Giant and Red Clump stars using infrared spectra data from APOGEE (https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/). The scheme from the following paper, Bovy etl (https://ui.adsabs.harvard.edu/abs/2014ApJ...790..127B/abstract) to extract the red gaint and red clump star spectra from the data. After cleaning and exploring the data, Convolutional Neural Networks and K-Means were used to preform both supervised and unsupervised machine learning training to achieve this task.

### RESOURCES
- Data from APOGEE: https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/
- Bitmasks for data cleaning: https://www.sdss4.org/dr17/irspec/apogee-bitmasks/
- Filtering values and reference for classification: https://ui.adsabs.harvard.edu/abs/2014ApJ...790..127B/abstract

### CONTENTS AND FILE KEY
Data Cleaning: data_cleaning.ipynb
- Purpose: Filtering out unwanted data for training
- Raw Data: allField-dr17.fits
- Initial Processing: allstar_processed_fixed.csv
- Final Datafile: cleaned_data.csv
 
Metadata Exploratory Data Analysis: metadata_eda.ipynb
- Purpose: Exploring the distribution of the data before training
- Data: cleaned_data.csv
- Graphs made:
  - Effective_temperature_graph.png
  - log_surface_gravity_graph.png
  - Signal_to_noise_graph.png
  - HR_density_heatmap_all
  - HR_density_heatmap_RBG
  - HR_rc_vs_rbg
  
Single Spectra Analysis: single_spectra_analysis.ipynb
- Purpose: To understand the structure of each spectra file so it can be properly formated for training
- Data: apStar-dr17-2M00000002+7417074.fits
- Graphs made:
  - single_spectra.png
    
Getting Data: getting_data.ipynb
- Purpose: program to batch download data and format it for training
- Data: cleaned_data.csv, cleaned_data_shuffled.csv
- Output files: processed_spectra.npy, processed_labels.npy
    
Supervised Machine Learning Model: supervised_model.ipynb
- Purpose: Construction, training, and results of supervised machine learning model to seperate red clump and red giant stars
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
The goal of this section of the project was to explore the spectra data and distribution before using it for training. Distributions of the data are as followed.

<img width="580" height="455" alt="Signal_to_Noise_graph" src="https://github.com/user-attachments/assets/5e7f3063-df16-46c5-ae6c-abddbf7b465f" /><br>
<img width="579" height="455" alt="Effective_temperature_graph" src="https://github.com/user-attachments/assets/8a1eada8-a07d-4ce7-bf05-c7ef4ef160d1" /><br>
<img width="571" height="455" alt="log_surface_gravity_graph" src="https://github.com/user-attachments/assets/2ed6198f-8fe3-4984-bad4-807072558959" /><br>

HR Diagrams using Log Effective Gravity and Log Effective Temperature

<img width="558" height="455" alt="HR_density_heatmap_all" src="https://github.com/user-attachments/assets/27e5874e-e1c1-4098-b8b4-24bbf057662b" /><br>
<img width="571" height="455" alt="HR_density_heatmap_RBG" src="https://github.com/user-attachments/assets/47473694-1612-427d-a8fa-eb6ed0ccfbee" /><br>

Using the scheme to filter Red Clump stars from the Bovy paper:

- t_ref = -382.5 * df_shuffled['RV_FEH'] + 4607
- delta_t = df_shuffled['RV_TEFF'] - t_ref
- logg_upper_bound = (0.0018 * delta_t) + 2.5
- rc_mask = df_shuffled["RV_LOGG"].between(1.8, logg_upper_bound)
- df_shuffled["Stellar_type"] = np.where(rc_mask, "Red Clump", "Red Giant"),

we get the following distribution of Red Giant and Red Clump stars in the HR Diagram.

<img width="567" height="455" alt="HR_rc_vs_rbg" src="https://github.com/user-attachments/assets/e140e31b-d9a6-4f44-b38e-edfef5f3d039" /><br>

### SINGLE SPECTRA ANALYSIS
The goal here was to explore how the spectra file is organized and what the spectra looks like.

Spectra File Structure:

APOGEE Reduction Pipeline Version: 0.17.22
- HDU0 : header
- HDU1 : flux
- HDU2 : flux uncertainty
- HDU3 : pixel bitmask
- HDU4 : sky
- HDU5 : sky uncertainty
- HDU6 : telluric
- HDU7 : telluric uncertainty
- HDU8 : LSF table
- HDU9 : RV table
- HDU10 : RV table for combined spectrum

Spectra Sample:
<img width="1023" height="547" alt="single_spectra" src="https://github.com/user-attachments/assets/ea5fb887-29a9-464d-ad1d-2eb577e3b804" /><br>

### SUPERVISED MACHINE LEARNING MODEL
This model was trained with about 13,000 samples. Both weighted classes and balancing classes were attempted, and balancing classes gave better accuracy.

Data Splits:
- Testing: 20%
- Leftover 80%:
  - Validation: 20%
  - Training: 80%
 
Training Scheme: 3 Channels, balanced classes
- Normalized Flux (using clipping)
- Flux Gradient
- Normalized Flux Error (using clipping)

Model Structure: 
- Kernal Sizes: 7, 11, 11
- Activation Function: relu
- Learning Rate: 0.0001
- Loss: binary cross-entropy
- Metric: accuracy

<img width="743" height="566" alt="supervised_model_structure" src="https://github.com/user-attachments/assets/bfa1f015-90bf-4cea-a451-95c949cd6aa3" /> <br>


#### Results

<img width="576" height="455" alt="supervised_model_accuracy" src="https://github.com/user-attachments/assets/0c5c68d2-6fea-40a7-97d2-c810232fc5cb" /><br>
<img width="539" height="455" alt="supervised_confusion_matrix" src="https://github.com/user-attachments/assets/6ee34032-3d29-4450-833d-7aa5e33f12b0" /><br>
<img width="567" height="455" alt="supervised_hr_diagram" src="https://github.com/user-attachments/assets/b23565f1-a555-4c49-b676-8d5a2798c2f1" /><br>
<img width="1590" height="990" alt="supervised_spectra_comparison" src="https://github.com/user-attachments/assets/7d4b3f8d-f97f-43d5-a59c-7e2076a5f6f9" /><br>



