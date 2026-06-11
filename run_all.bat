@echo off
setlocal
cd /d %~dp0
python code\00_prepare_data.py
python code\01_log_logistic_aft.py
python code\02_random_survival_forest.py
python code\03_shap_and_pdp.py
python code\04_descriptive_plots.py
endlocal
