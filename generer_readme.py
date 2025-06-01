import os

def build_tree(dir_path, prefix=""):
    items = sorted([f for f in os.listdir(dir_path) if not f.startswith(".")])
    result = ""
    for index, name in enumerate(items):
        path = os.path.join(dir_path, name)
        connector = "└── " if index == len(items) - 1 else "├── "
        result += prefix + connector + name + "\n"
        if os.path.isdir(path):
            extension = "    " if index == len(items) - 1 else "│   "
            result += build_tree(path, prefix + extension)
    return result

if __name__ == "__main__":
    root = "."  # racine du projet
    tree_output = build_tree(root)
    with open("TREE.txt", "w", encoding="utf-8") as f:
        f.write(tree_output)
    print("✅ Arborescence générée dans TREE.txt")
