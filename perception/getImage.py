from pathlib import Path
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = Path.cwd()


def get_image():
    while True:
        image_path = input("\nEnter the path to an image: ").strip().strip('"')

        if not image_path:
            raise ValueError("No path entered")

        input_path = Path(image_path)
        candidate_paths = []

        if input_path.is_absolute():
            candidate_paths.append(input_path)
        else:
            candidate_paths.extend([
                input_path,
                WORKDIR / input_path,
                SCRIPT_DIR / input_path,
            ])

        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                img = cv2.imread(str(candidate))
                if img is not None:
                    resolved_path = str(candidate.resolve())
                    print(f"Loaded image: {resolved_path}")
                    return img, resolved_path

        raise FileNotFoundError(f"Could not load image at {image_path}")
