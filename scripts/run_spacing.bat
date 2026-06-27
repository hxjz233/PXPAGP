@echo off
setlocal
if "%~1"=="" goto :help
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python src\pxp_agp_scaling.py --mode spacing %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:help
echo Usage: %~nx0 [options]
echo.
echo Mode: spacing
echo Parameters: --l-values --hxz-min --hxz-max --hxz-count --spacing-output --boundary --backend --force
echo Example: %~nx0 --l-values 18 20 22 --hxz-min -0.05 --hxz-max 0.02 --hxz-count 71 --spacing-output pxp_spacing_ratio.png
echo.
exit /b 0
