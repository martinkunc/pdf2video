"""Entry point: the GUI by default, the CLI for sub-commands.

pdf2video                 → GUI
pdf2video book.pdf        → GUI with the document loaded
pdf2video audio book.pdf  → CLI (same as `pdf2video-cli audio book.pdf`)
"""

from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--cli":
        args = args[1:]
    elif not args or args[0] not in (*_commands(), "-h", "--help"):
        from .ui.app import run

        return run(sys.argv)
    from .cli import main as cli_main

    return cli_main(args)


def cli() -> int:
    from .cli import main as cli_main

    return cli_main()


def _commands() -> tuple[str, ...]:
    from .cli import COMMANDS

    return COMMANDS


if __name__ == "__main__":
    sys.exit(main())
