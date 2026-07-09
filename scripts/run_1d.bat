@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode 1d %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: 1d
echo Parameters: --operator --calc --coupling-min --coupling-max --coupling-count --output --boundary --inv-sector --backend --force
echo Example: %~nx0 --operator ss --calc spacing --l-values 14 16 18 20 --coupling-min 0.0 --coupling-max 0.25 --coupling-count 9 --output pxp_ss_spacing.png --backend cpu --inv-sector 0
echo.
exit /b 0