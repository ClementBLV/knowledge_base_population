#!/bin/bash

# Description: Change all .sh files in the current directory (and subdirectories) to executable.

# Find all .sh files and update permissions
find . -type f -name "*.sh" -exec chmod +x {} \;

# Print the result
echo "All .sh files have been made executable!"
