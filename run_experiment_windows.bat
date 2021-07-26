@echo off

md output
md output/%2

REM Articulated (No face-swap)
call run_local_windows.bat --input-video videos/%1 --output output/%2/articulated.mp4
REM Articulated (Poisson)
call run_local_windows.bat --input-video videos/%1 --output output/%2/poisson.mp4 --swap-face --swapper poisson
REM Articulated (EDS)
call run_local_windows.bat --input-video videos/%1 --output output/%2/eds.mp4 --swap-face --swapper eds
REM Articulated (Triangulation)
call run_local_windows.bat --input-video videos/%1 --output output/%2/triangulation.mp4 --swap-face --swapper triangulation
REM FSGAN
call run_local_windows.bat --input-video videos/%1 --output output/%2/fsgan.mp4 --fsgan
