@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode hxz %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: hxz
echo Parameters: --l-values --hxz-min --hxz-max --hxz-count --output --boundary --inv-sector --backend --force
echo Example: %~nx0 --l-values 14 16 18 20 --hxz-min 0.0 --hxz-max 0.25 --hxz-count 9 --output pxp_agp_scaling.png --backend cpu --inv-sector 0
echo.
exit /b 0
