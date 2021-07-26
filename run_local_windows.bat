@echo off

call scripts/settings_windows.bat

call conda activate %CONDA_ENV_NAME%

call python afy/test_locally.py --config %CONFIG% --checkpoint %CKPT% %*
