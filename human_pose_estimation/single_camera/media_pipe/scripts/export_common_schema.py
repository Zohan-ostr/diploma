#!/usr/bin/env python3
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    print(f'Nothing to export. Current schema is already standardised under: {project_root / "runs"}')

if __name__ == '__main__':
    main()
