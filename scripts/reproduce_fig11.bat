@echo off
setlocal
call "D:\Miniconda\Scripts\activate.bat" quspin
pushd "%~dp0.."
python scripts\reproduce_fig11.py %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
