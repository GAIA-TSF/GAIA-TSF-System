#!/usr/bin/python

import argparse
from ruamel.yaml import YAML

def main(password: str, config_file: str = "config.yaml"):
    yaml = YAML()
    yaml.preserve_quotes = True  # keep quotes as in the source
    yaml.indent(mapping=2, sequence=4, offset=2)

    # read
    with open(config_file, "r") as cfg:
        data = yaml.load(cfg)

    # TODO: to be parameterized if needed for more than now
    data["eou"]["eodag"]["cop_dataspace"]["auth"]["credentials"]["password"] = password

    # write
    with open(config_file, "w") as cfg:
        yaml.dump(data, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--password",
        type=str,
        required=True,
        help="New password to store in the configuration file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )

    args = parser.parse_args()
    main(args.password, args.config)
