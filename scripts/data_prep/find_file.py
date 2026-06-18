import os

search_dir = r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE"

print("Searching for your trained weights...")
for root, dirs, files in os.walk(search_dir):
    if "loss.py" in files:
        print(f"\n✅ FOUND IT! Your weights are located at:")
        print(os.path.join(root, "loss.py"))