@echo off

set VIDEO_1=with_glasses.mp4
set VIDEO_2=without_glasses.mp4
set VIDEO_3=crystal.mp4
set VIDEO_4=ellbat.mp4
set VIDEO_5=healthy_gamer.mp4
set VIDEO_6=javier.mp4

:: call run_experiment_windows.bat %VIDEO_1% with_glasses 0 256
:: call run_experiment_windows.bat %VIDEO_1% with_glasses 1 256
:: call run_experiment_windows.bat %VIDEO_1% with_glasses 0 512
:: call run_experiment_windows.bat %VIDEO_1% with_glasses 1 512

:: call run_experiment_windows.bat %VIDEO_2% without_glasses 0 256
:: call run_experiment_windows.bat %VIDEO_2% without_glasses 1 256
:: call run_experiment_windows.bat %VIDEO_2% without_glasses 0 512
:: call run_experiment_windows.bat %VIDEO_2% without_glasses 1 512

call run_experiment_windows.bat %VIDEO_3% crystal 0 256
call run_experiment_windows.bat %VIDEO_3% crystal 1 256
call run_experiment_windows.bat %VIDEO_3% crystal 0 512
call run_experiment_windows.bat %VIDEO_3% crystal 1 512

call run_experiment_windows.bat %VIDEO_4% ellbat 0 256
call run_experiment_windows.bat %VIDEO_4% ellbat 1 256
call run_experiment_windows.bat %VIDEO_4% ellbat 0 512
call run_experiment_windows.bat %VIDEO_4% ellbat 1 512

call run_experiment_windows.bat %VIDEO_5% healthy_gamer 0 256
call run_experiment_windows.bat %VIDEO_5% healthy_gamer 1 256
call run_experiment_windows.bat %VIDEO_5% healthy_gamer 0 512
call run_experiment_windows.bat %VIDEO_5% healthy_gamer 1 512

call run_experiment_windows.bat %VIDEO_6% javier 0 256
call run_experiment_windows.bat %VIDEO_6% javier 1 256
call run_experiment_windows.bat %VIDEO_6% javier 0 512
call run_experiment_windows.bat %VIDEO_6% javier 1 512
