@echo off

set VIDEO_1=with_glasses.mp4
set VIDEO_2=without_glasses.mp4
set VIDEO_3=crystal.mp4
set VIDEO_4=ellbat.mp4
set VIDEO_5=healthy_gamer.mp4
set VIDEO_6=javier.mp4

call run_experiment_windows.bat %VIDEO_1% with_glasses 0
call run_experiment_windows.bat %VIDEO_1% with_glasses 1

call run_experiment_windows.bat %VIDEO_2% without_glasses 0
call run_experiment_windows.bat %VIDEO_2% without_glasses 1

call run_experiment_windows.bat %VIDEO_3% crystal 0
call run_experiment_windows.bat %VIDEO_3% crystal 1

call run_experiment_windows.bat %VIDEO_4% ellbat 0
call run_experiment_windows.bat %VIDEO_4% ellbat 1

call run_experiment_windows.bat %VIDEO_5% healthy_gamer 0
call run_experiment_windows.bat %VIDEO_5% healthy_gamer 1

call run_experiment_windows.bat %VIDEO_6% javier 0
call run_experiment_windows.bat %VIDEO_6% javier 1
