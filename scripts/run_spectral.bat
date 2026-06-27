@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode spectral %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: spectral
echo Parameters: --l-values --spectral-hxz --spectral-bins --spectral-output --boundary --inv-sector --backend --force
echo Example: %~nx0 --l-values 14 16 18 20 --spectral-hxz 2 --spectral-bins 100 --spectral-output pxpz_spectral_2.png --backend cpu --inv-sector 0
echo.
exit /b 0
