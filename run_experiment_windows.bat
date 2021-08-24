@echo off

echo %1
echo %2
echo %3
echo %4

set BASE_PATH=output/%4/%2/%3

echo Articulated (No face-swap)
call run_local_windows.bat --desired-avatar %3 --resolution %4 --input-video videos/%1 --output %BASE_PATH%/articulated
echo Articulated (Poisson)
:: call run_local_windows.bat --desired-avatar %3 --resolution %4 --input-video videos/%1 --output %BASE_PATH%/poisson --swap-face --swapper poisson
echo Articulated (EDS)
:: call run_local_windows.bat --desired-avatar %3 --resolution %4 --input-video videos/%1 --output %BASE_PATH%/eds --swap-face --swapper eds
echo Articulated (Triangulation)
:: call run_local_windows.bat --desired-avatar %3 --resolution %4 --input-video videos/%1 --output %BASE_PATH%/triangulation --swap-face --swapper triangulation
echo FSGAN
call run_local_windows.bat --desired-avatar %3 --resolution %4 --input-video videos/%1 --output %BASE_PATH%/fsgan --fsgan
