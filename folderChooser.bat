:: fchooser.bat
:: launches a folder chooser and outputs choice to the console
:: https://stackoverflow.com/a/15885133/1683264

@echo off
setlocal

set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please choose folder with video data',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "videoFolder=%%I"

set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please choose folder with background image',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "bgImageFolder=%%I"

set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please choose folder to save MRCNN model',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "mrcnnFolder=%%I"

set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please choose folder to save DLC model',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "dlcFolder=%%I"

setlocal enabledelayedexpansion
streamlit run app_markerless_mice_tracking.py -- --video="!videoFolder!"  --background="!bgImageFolder!" --mrcnn_model="!mrcnnFolder!" --dlc_project="!dlcFolder!"
endlocal
