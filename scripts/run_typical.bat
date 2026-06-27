@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode typical %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: typical
echo Parameters: --l-values --hxz-min --hxz-max --hxz-count --chi-typ-output --boundary --inv-sector --backend --force
echo Example: %~nx0 --l-values 12 14 16 18 --hxz-min -0.03 --hxz-max -0.01 --hxz-count 21 --chi-typ-output pxp_agp_scaling_zoom_vedika.png
echo.
exit /b 0
