from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from layout_spatial_reasoning.cli import main


if __name__ == "__main__":
    main()
