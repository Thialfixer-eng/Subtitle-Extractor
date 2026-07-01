@echo off
cd /d "%~dp0"
echo Setting folder icon...
set "FOLDER=%CD%"
set "ICO=%FOLDER%\assets\logo.ico"

(
echo [.ShellClassInfo]
echo IconResource=%ICO%,0
echo IconFile=%ICO%
echo IconIndex=0
echo InfoTip=Subtitle Extractor
) > "%FOLDER%\desktop.ini"

attrib +h +s "%FOLDER%\desktop.ini"
attrib +r "%FOLDER%"

echo OK: Folder icon set.
echo Press F5 in Explorer to see it.
pause
