"""Command line entry point for ``python -m diann_speclib_explorer``."""

from __future__ import annotations

import sys

from . import __version__

COMMANDS = {
    "parse": "diann_speclib_explorer.parse_diann_speclib",
    "extract-rt": "diann_speclib_explorer.extract_diann_rt",
    "plot-rt": "diann_speclib_explorer.plot_diann_rt_comparison",
    "plot-modifications": "diann_speclib_explorer.plot_diann_modification_pairs",
}


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] in {"--version", "-V"}:
        print(__version__)
        return 0
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(f"DIA-NN Speclib Explorer {__version__}\n")
        print("Usage: diann-speclib <command> [options]\n")
        print("Commands:")
        for command, module in COMMANDS.items():
            print(f"  {command:22} {module.rsplit('.', 1)[-1].replace('_', ' ')}")
        print("\nRun 'diann-speclib <command> --help' for command-specific options.")
        return 0 if len(sys.argv) >= 2 else 2

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print("Available commands: " + ", ".join(COMMANDS), file=sys.stderr)
        return 2

    from importlib import import_module

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    return import_module(module_name).main()


if __name__ == "__main__":
    raise SystemExit(main())
