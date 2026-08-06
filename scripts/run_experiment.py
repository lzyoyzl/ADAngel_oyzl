#!/usr/bin/env python3
from adangel.cli import main

if __name__ == "__main__":
    main(["run", *__import__("sys").argv[1:]])
