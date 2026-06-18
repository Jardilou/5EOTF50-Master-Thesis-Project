import os

def ignore_large_files(directory=".", size_limit_mb=50):
    gitignore_path = os.path.join(directory, ".gitignore")
    size_limit_bytes = size_limit_mb * 1024 * 1024
    
    # Read existing ignored files to avoid duplicates
    existing_ignores = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            existing_ignores = set(line.strip() for line in f if line.strip() and not line.startswith('#'))

    large_files = []
    print(f"Scanning for files larger than {size_limit_mb}MB...")
    
    # Define the exact folder names you want to completely skip searching
    folders_to_skip = { 'DATASETS', 'ultralytics'}
    
    for dirpath, dirnames, filenames in os.walk(directory):
        
        # Modify dirnames in-place to prevent os.walk from descending into them
        dirnames[:] = [d for d in dirnames if d not in folders_to_skip]
            
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                if os.path.getsize(filepath) > size_limit_bytes:
                    # Get relative path and ensure forward slashes for Git
                    rel_path = os.path.relpath(filepath, directory).replace(os.sep, '/')
                    
                    if rel_path not in existing_ignores:
                        large_files.append(rel_path)
            except OSError:
                pass # Skip files we don't have permission to read

    # Append to .gitignore
    if large_files:
        with open(gitignore_path, 'a') as f:
            f.write(f"\n# Auto-generated: Ignore files larger than {size_limit_mb}MB\n")
            for file in large_files:
                f.write(f"{file}\n")
                print(f"Ignored: {file}")
        print(f"\nSuccess! Added {len(large_files)} large files to .gitignore.")
    else:
        print("\nNo unignored large files found.")

if __name__ == "__main__":
    ignore_large_files()