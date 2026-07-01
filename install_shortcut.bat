@echo off
cd /d "%~dp0"
echo Converting logo.png to logo.ico...
py -3.11 assets\make_icon.py
if %errorlevel% neq 0 pause & exit /b

echo Creating desktop shortcut...
set "SCRIPT_DIR=%~dp0"
powershell -Command "$wss = New-Object -ComObject WScript.Shell; $s = $wss.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Subtitle Extractor.lnk'); $s.TargetPath = '%SCRIPT_DIR%gui.bat'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.IconLocation = '%SCRIPT_DIR%assets\logo.ico, 0'; $s.Description = 'Subtitle Extractor - OCR + Dictionary + Translation'; $s.Save(); Write-Host 'OK: Desktop shortcut created'"
if %errorlevel% neq 0 echo FAILED & pause & exit /b
pause
