# diyUiO-66
CatBoost models for UiO-66 properties prediction

Input variables of the machine learning algorithm are the synthesis conditions of the Zr-based metal-organic framework UiO-66, namely, 
concentrations of reagents (Zr-salt, terephtalic acid), H2O:Zr ratio, modulator:Zr ratio, modulator type, Zr-source, synthesis temperature, 
synthesis time, aging time, number of DMF washes, total number of washes, last solvent in pores, activation temperature and activation time.

Output variables are resulting UiO-66 properties such as BET specific surface area, defects concentration in terms of Zr-cluster:linker ratio and average particle size.

Separate datasets for three properties of UiO-66 are provided (Area.csv, Defects.csv, Size.csv) and full dataset with literature sources (UiO_dataset.xls). 

First, you need to install Python (https://www.python.org/).
Then open a terminal (cmd or powershell if Windows) and prepare your environment using the package installer for Python pip to install 
the following packages: notebook, ipywidgets, pandas, matplotlib, pdpbox (version 0.2.1), seaborn, scikit-learn, catboost.
Example: pip install notebook, pip install ipywidgets, etc.

Then download and unpack the zip file containing the repository or clone the repository with the command in terminal: git clone https://github.com/kirloy/diyUiO-66.git
Change the directory in the terminal to the directory containing the repository files (or just open a terminal there).
The program can then be run using the terminal command: python diyUiO-66_terminal_only.py 

There is also a polished version of the program that runs in Jupiter Notebook to easily vary conditions and make predictions about UiO-66 properties (Jupyter terminal.ipynb).
To run it, first start Jupyter Notebook with the command in the terminal: jupyter notebook
After executing the command you will be automatically redirected to the browser with Jupyter Notebook, if this does not happen you can copy one of the URLs specified in the terminal to the browser.
Upload the repository files to Jupyter Notebook or open the repository folder in Jupyter Notebook. You can then open and run the Jupyter terminal.ipynb file.

The models are trained each time the code is run. To train the models on the new data, it is sufficient to replace the data in the dataset files (Area.csv, Defects.csv, Size.csv).

To reproduce 2D PDP plots and figures with SHAP values, there is also a JupiterNotebook file (Example of 2D PDP curve and SHAP.ipynb). It only works with version 0.2.1 of the pdpbox package.

Folder "Optimized_synthesis" contains the implementation of optimal synthesis conditions search for a desired set of UiO-66 properties. No additional packages are needed to run it.
Terminal_only.py in Optimized_synthesis can be used to run search in the terminal and Jupyter terminal.ipynb in Optimized_synthesis can be used to run search in the Jupyter Notebook.
