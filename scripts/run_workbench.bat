@echo off
setlocal
set "ROOT=%~dp0.."
set "PY_CMD=python"
if exist "%ROOT%\.venv\Scripts\python.exe" set "PY_CMD=%ROOT%\.venv\Scripts\python.exe"
if defined PYTHON_EXE set "PY_CMD=%PYTHON_EXE%"
set "PYTHONPATH=%ROOT%;%ROOT%\python;%PYTHONPATH%"
pushd "%ROOT%"
"%PY_CMD%" -m workbench.main %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
