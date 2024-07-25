#!/bin/bash

# Get the list of installed packages
packages=$(pip list --format=freeze)

# Loop through each package
for package in $packages; do
  # Extract the package name (part before the '=' sign)
  package_name=$(echo $package | cut -d '=' -f 1)
  # Get the package location using pip show
  package_location=$(pip show $package_name | grep Location | awk '{print $2}')
  # Print the package name and its location
  echo "$package_name: $package_location"
done