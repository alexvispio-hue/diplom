import json
from pathlib import Path

import cv2
import numpy as np
import requests


def main() -> None:
    image = np.full((300, 800, 3), 255, dtype=np.uint8)
    for index, text in enumerate(("first line", "second line", "third line")):
        cv2.putText(image, text, (40, 70 + index * 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    page_path = Path("/tmp/diplom_page_smoke.png")
    cv2.imwrite(str(page_path), image)
    with page_path.open("rb") as page_file:
        response = requests.post(
            "http://localhost:8000/api/recognize-page",
            files={"file": ("page.png", page_file, "image/png")},
            timeout=300,
        )

    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
