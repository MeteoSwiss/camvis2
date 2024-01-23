#!/bin/bash
# remove files except .gitkeeps
find outputs/ -type f ! -name ".gitkeep" ! -name "*.py" -exec rm -f {} +
# remove empty directories
find outputs/ -mindepth 1 -type d -empty -exec rmdir {} +