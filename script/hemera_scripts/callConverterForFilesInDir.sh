#check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

DIRECTORY="$1"

# Check if the provided argument is a valid directory
if [ ! -d "$DIRECTORY" ]; then
  echo "Error: $DIRECTORY is not a valid directory."
  exit 1
fi

# Iterate over files starting with "babelstream-gpu" or "babelstream-cpu"
for file in "$DIRECTORY"/babelstream-{gpu,cpu}*; do
  if [ -f "$file" ]; then
    python3 convertTxtToJson.py "$file"
  fi
done

