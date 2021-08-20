@echo off

echo %1
echo %2
echo %3
echo %4

echo Articulated (No face-swap)
call run_local_windows.bat --resolution %4 --input-video videos/%1 --output output/%4/%2/%3/articulated
echo Articulated (Poisson)
call run_local_windows.bat --resolution %4 --input-video videos/%1 --output output/%4/%2/%3/poisson --swap-face --swapper poisson
echo Articulated (EDS)
call run_local_windows.bat --resolution %4 --input-video videos/%1 --output output/%4/%2/%3/eds --swap-face --swapper eds
echo Articulated (Triangulation)
call run_local_windows.bat --resolution %4 --input-video videos/%1 --output output/%4/%2/%3/triangulation --swap-face --swapper triangulation
echo FSGAN
call run_local_windows.bat --resolution %4 --input-video videos/%1 --output output/%4/%2/%3/fsgan --fsgan
