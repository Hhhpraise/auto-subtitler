@echo off
REM Script to create a GitHub release and upload assets (Windows version)

REM Set variables
set VERSION=2.0
set RELEASE_DIR=github_release
set RELEASE_NAME=Auto Subtitler v%VERSION%
set RELEASE_NOTES=Latest release of Auto Subtitler with performance improvements and bug fixes.

REM Create a new release
echo Creating GitHub release...
gh release create "v%VERSION%" --title "%RELEASE_NAME%" --notes "%RELEASE_NOTES%"

REM Upload assets
echo Uploading assets...
gh release upload "v%VERSION%" "%RELEASE_DIR%/AutoSubtitler_Setup.exe#Installer"
gh release upload "v%VERSION%" "%RELEASE_DIR%/AutoSubtitler.exe#Portable Version"
gh release upload "v%VERSION%" "%RELEASE_DIR%/README.md#Release Notes"

echo Release created successfully!
pause
