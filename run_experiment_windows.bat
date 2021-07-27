@echo off

echo %1
echo %2

echo Articulated (No face-swap)
call run_local_windows.bat --input-video videos/%1 --output output/%2/articulated
echo Articulated (Poisson)
call run_local_windows.bat --input-video videos/%1 --output output/%2/poisson --swap-face --swapper poisson
echo Articulated (EDS)
call run_local_windows.bat --input-video videos/%1 --output output/%2/eds --swap-face --swapper eds
echo Articulated (Triangulation)
call run_local_windows.bat --input-video videos/%1 --output output/%2/triangulation --swap-face --swapper triangulation
echo FSGAN
call run_local_windows.bat --input-video videos/%1 --output output/%2/fsgan --fsgan
