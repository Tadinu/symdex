#!/bin/bash

SERVER="than@denver.informatik.uni-bremen.de:/raid/home/than/EQUIVARIANT/symdex"

# Get modified / new / deleted file paths (porcelain gives status + path)
changed=$(git status --short)

# Optionally bail out if no changes
if [ -z "$changed" ]; then
  echo "No changes"
  exit 0
fi

# Process each line
# In porcelain mode, lines look like: "XY path/to/file" or "XY path1 -> path2" for renames
# Often, you strip the first 3 characters ("XY " prefix) to get the file path(s).
paths=()
while IFS= read -r line; do
  # Strip first 3 chars (status + space)
  # Caution: renames may include " -> " in the path part
  path="${line:3}"
  paths+=("$path")
done <<< "$changed"

# Now you have an array `paths` with the modified file paths
printf 'Changed files:\n'
for p in "${paths[@]}"; do
  echo "  $p"
done

# If you want to join into a single variable (space-separated):
allpaths="${paths[*]}"
echo "All in one: $allpaths"

for c in $allpaths
do
	rsync -avuzP $c "$SERVER"/$c
	echo $c
done
