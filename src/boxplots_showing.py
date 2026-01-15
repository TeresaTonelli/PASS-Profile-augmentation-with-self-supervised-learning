from errors_evaluation import error_evaluation_boxplots



RMSE_txt = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/src/results_trial_1Unet/NITRATE/2026-01-15/error_evaluation/RMSE_file.txt"
RMSE_linear_txt = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/src/results_trial_1Unet/NITRATE/2026-01-15/error_evaluation/RMSE_float_linear_file.txt"
save_dir = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/src/results_trial_1Unet/NITRATE/2026-01-15/error_evaluation"

error_evaluation_boxplots(RMSE_txt, RMSE_linear_txt, save_dir)