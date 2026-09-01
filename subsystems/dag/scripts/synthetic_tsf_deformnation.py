"""Compatibility entry point for the formerly misspelled filename."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(Path(__file__).with_name('synthetic_tsf_deformation.py'), run_name='__main__')
