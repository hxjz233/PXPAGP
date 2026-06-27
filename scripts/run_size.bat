@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode size %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: size
echo Parameters: --l-values --hxz-fixed --output --boundary --inv-sector --backend --force
echo Example: %~nx0 --l-values 10 12 14 16 18 --hxz-fixed 0.0
echo.
exit /b 0
