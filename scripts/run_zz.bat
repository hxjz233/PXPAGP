@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode zz %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: zz
echo Parameters: --l-values --zz-min --zz-max --zz-count --zz-output --boundary --inv-sector --backend --force
echo Example: %~nx0 --l-values 14 16 18 20 --zz-min 0.0 --zz-max 0.25 --zz-count 9 --zz-output pxp_zz_scaling.png --backend cpu --inv-sector 0
echo.
exit /b 0