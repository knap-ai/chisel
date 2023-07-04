from typing import List, Dict

from pathlib import Path

import json

import os
from os import walk, open


class Models:
    def get_model_directory(self) -> Path:
        return Path("./model-specs")  # TODO config here, eventually API/CMS

    def list_model_specs(self) -> List[Path]:
        directory = self.get_model_directory()
        filenames = []

        filenames = sorted(directory.glob("*.json"))

        #        for directory, directory_names, file_names in walk(directory):
        #           filenames.extend(file_names)

        return filenames

    def get_model_spec(self, filepath: Path) -> Dict:
        spec = {}

        with filepath.open() as read_file:
            spec = json.load(read_file)

        return spec


if __name__ == "__main__":
    models = Models()
    print(os.getcwd())

    specs = models.list_model_specs()
    print(specs)

    for spec_path in specs:
        print("===" + spec_path.name + "===")

        model_spec = models.get_model_spec(spec_path)

        pretty_spec = json.dumps(model_spec, indent=4)
        print(pretty_spec)
