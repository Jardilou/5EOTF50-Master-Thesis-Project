import os
import ast
import sys

def extract_exact_imports(filepath):
    """Safely parses a Python file and extracts top-level import names."""
    imports = set()
    try:
        # errors='ignore' ensures it will NEVER crash on weird characters
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read(), filename=filepath)
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get the top-level module (e.g., 'scipy.spatial' -> 'scipy')
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except SyntaxError:
        # Skips files with invalid Python syntax
        pass
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        
    return imports

def main():
    directory = "."
    all_imports = set()
    
    print("Scanning project for exact import statements...\n")
    
    for root, _, files in os.walk(directory):
        # Skip virtual environments, git, and caches
        if any(skip in root for skip in ['venv', 'env', '.git', '__pycache__']):
            continue
            
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                filepath = os.path.join(root, file)
                # If it's a Jupyter notebook, we won't parse the JSON structure perfectly here,
                # but if you exported your notebook to a .py file, it will scan perfectly.
                if file.endswith('.py'):
                    all_imports.update(extract_exact_imports(filepath))

    # Identify Standard Library modules (Python 3.10+)
    stdlib = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set()
    
    builtins = []
    external = []

    for imp in sorted(all_imports):
        if imp in stdlib or imp in sys.builtin_module_names:
            builtins.append(imp)
        else:
            external.append(imp)

    print("=== EXACT MODULES IMPORTED IN YOUR CODE ===")
    print("\n--- EXTERNAL / THIRD-PARTY MODULES ---")
    for module in external:
        print(f"- {module}")

    print("\n--- PYTHON BUILT-IN LIBRARIES ---")
    for module in builtins:
        print(f"- {module}")

if __name__ == "__main__":
    main()