#!/bin/bash
# Script to create a GitHub release and upload assets

# Set variables
VERSION="2.0"
RELEASE_DIR="github_release"
RELEASE_NAME="Auto Subtitler v$VERSION"
RELEASE_NOTES="Latest release of Auto Subtitler with performance improvements and bug fixes."

# Create a new release
echo "Creating GitHub release..."
gh release create "v$VERSION" --title "$RELEASE_NAME" --notes "$RELEASE_NOTES"

# Upload assets
echo "Uploading assets..."
gh release upload "v$VERSION" "$RELEASE_DIR/AutoSubtitler_Setup.exe#Installer"
gh release upload "v$VERSION" "$RELEASE_DIR/AutoSubtitler.exe#Portable Version"
gh release upload "v$VERSION" "$RELEASE_DIR/README.md#Release Notes"

echo "Release created successfully!"
