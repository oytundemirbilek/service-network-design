"""Entrypoint for the CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from airnet import __version__
from airnet.experiments import Experiment, SensitivityAnalysis


def parse_args() -> Namespace:
    """Parse command line arguments and return as dictionary."""
    parser = ArgumentParser(
        prog="airnet",
        description="",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    main_args = parser.add_argument_group("main options")
    main_args.add_argument(
        "-o",
        "--optimize",
        type=str,
        default=None,
        help="The optimization model to run.",
        choices=["service", "hubs"],
    )

    main_args.add_argument(
        "-s",
        "--sensitivity",
        type=str,
        default=None,
        help="The optimization model to run.",
        choices=["cost", "price", "capacity", "all"],
    )

    return parser.parse_args()


def main() -> None:
    """Run main function from CLI."""
    args = parse_args()

    exp = Experiment()
    if args.optimize == "service":
        exp.run_wo_hubs()

    if args.optimize == "hubs":
        exp.run_hub_location()

    sens = SensitivityAnalysis()
    if args.sensitivity == "all":
        sens.run_and_save()

    if args.sensitivity == "price":
        sens.run_price_sensitivity()

    if args.sensitivity == "cost":
        sens.run_cost_sensitivity()

    if args.sensitivity == "capacity":
        sens.run_capacity_sensitivity()


if __name__ == "__main__":
    main()
