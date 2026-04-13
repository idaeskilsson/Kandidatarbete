import sys
sys.stdout.reconfigure(encoding="utf-8")
from src.preprocess import run_preprocessing
from src.build_features import run_build_features
from src.sleeping_giant_model import run_sleeping_giant
from src.rising_star_model import run_rising_star


def main(fetch_data: bool = False):
    if fetch_data:
        from src.collect_data import run_collection
        run_collection()

    run_preprocessing()
    run_build_features()
    run_rising_star()
    run_sleeping_giant()


if __name__ == "__main__":
    fetch = "--fetch" in sys.argv
    main(fetch_data=fetch)
