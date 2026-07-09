@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode ss %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: ss
echo Parameters: --l-values --ss-min --ss-max --ss-count --ss-output --boundary --inv-sector --backend --force
echo Example: %~nx0 --l-values 14 16 18 20 --ss-min 0.0 --ss-max 0.25 --ss-count 9 --ss-output pxp_ss_scaling.png --backend cpu --inv-sector 0
echo.
exit /b 0