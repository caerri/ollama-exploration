#!/bin/bash
# Run the relay using the virtual environment's Python
# Usage: ./run.sh
cd "$(dirname "$0")"
.venv/bin/python main.py
