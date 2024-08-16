import argparse
from pathlib import Path


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=Path, help="Path to the reference containing the 'gold' preferences.")
    parser.add_argument("--annotation_path", type=Path, help="Path to the annotations file.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    # TODO: report annotator agreement with gold labels (accuracy?)

    # TODO: report annotator agreement across different subsets (and weight them based on the frequency of the subsets.)
    pass


if __name__ == "__main__":
    main()
