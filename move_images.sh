#!/bin/bash

# File containing the mapping
MAPPING_FILE="labels.txt"

# Check if the mapping file exists
if [[ ! -f "$MAPPING_FILE" ]]; then
  echo "Mapping file '$MAPPING_FILE' not found!"
  exit 1
fi

# Process each line of the mapping file
while IFS= read -r line || [[ -n "$line" ]]; do
  # Parse the line to extract name and folder (comma-separated)
  NAME=$(echo "$line" | cut -d',' -f1 | xargs)
  FOLDER=$(echo "$line" | cut -d',' -f2 | xargs)
  
  # Find the file matching the name (case-insensitive, assuming single match)
  IMAGE=$(find . -maxdepth 1 -type f -iname "$NAME.*" | head -n 1)
  
  if [[ -z "$IMAGE" ]]; then
    echo "File for name '$NAME' not found, skipping..."
    continue
  fi
  
  # Create the folder if it doesn't exist
  if [[ ! -d "$FOLDER" ]]; then
    mkdir -p "$FOLDER"
  fi
  
  # Move the image to the destination folder
  mv "$IMAGE" "$FOLDER/"
  echo "Moved '$IMAGE' to '$FOLDER/'"
done < "$MAPPING_FILE"

echo "Processing complete!"
