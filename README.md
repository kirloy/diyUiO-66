# diyUiO-66
CatBoost models for UiO-66 properties prediction

Input variables of the machine learning algorithm are the synthesis conditions of the Zr-based metal-organic framework UiO-66, namely, 
concentrations of reagents (Zr-salt, terephtalic acid), H2O:Zr ratio, modulator:Zr ratio, modulator type, Zr-source, synthesis temperature, 
synthesis time, aging time, number of DMF washes, total number of washes, last solvent in pores, activation temperature and activation time.

Output variables are resulting UiO-66 properties such as BET specific surface area, defects concentration in terms of Zr-cluster:linker ratio and average particle size.

Separate datasets for three properties of UiO-66 are provided (Area.csv, Defects.csv, Size.csv) and full dataset with literature sources (UiO_dataset.xls). 
There is a file that works simply in the terminal without JupiterNotebook (diyUiO-66_terminal_only.py). 
There is also a polished version of the program that runs in JupiterNotebook to easily vary conditions and make predictions about UiO-66 properties (Jupyter terminal.ipynb). 
The models are trained every time the code is run. To train models on the new data, it is enough to replace the data in the dataset files.

To reproduce 2D PDP plots and figures with SHAP-values, there is also JupiterNotebook file (Example of 2D PDP curve and SHAP.ipynb).

Required packages: notebook, ipywidgets, pandas, matplotlib, pdpbox (version 0.2.1), seaborn, scikit-learn, catboost
