@echo off
rem double-click (Windows) to start the exo dashboard in the browser; closing this window stops it
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul || powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv run --group dev start
pause
