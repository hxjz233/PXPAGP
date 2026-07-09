@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode z %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: z
echo Parameters: --l-values --z-min --z-max --z-count --z-output --boundary --inv-sector --backend --force
echo Example: %~nx0 --l-values 14 16 18 20 --z-min 0.0 --z-max 0.25 --z-count 9 --z-output pxp_z_scaling.png --backend cpu --inv-sector 0
echo.
exit /b 0