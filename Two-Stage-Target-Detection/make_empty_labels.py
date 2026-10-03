import os

# Paths
train_images = "dataset/images/train"
train_labels = "dataset/labels/train"

# Ensure labels folder exists
os.makedirs(train_labels, exist_ok=True)

# Supported image formats
image_exts = (".jpg", ".jpeg", ".png")

for filename in os.listdir(train_images):
    if filename.lower().endswith(image_exts):
        base_name = os.path.splitext(filename)[0]

        # Skip any file with 'target' in its name
        if "target" in base_name.lower():
            continue  

        # Label file path
        label_file = os.path.join(train_labels, base_name + ".txt")

        # Create only if it doesn't already exist
        if not os.path.exists(label_file):
            with open(label_file, "w") as f:
                pass
            print(f"Created empty label: {label_file}")
        else:
            print(f"Skipped (already exists): {label_file}")
