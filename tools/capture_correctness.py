#!/usr/bin/env python3
"""Run real correctness checks in a new Terminal window and capture that window."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'csr_spmv_lab.cpp')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/correctness_capture')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory')
    output.mkdir(parents=True)
    command = 'clear; cd ' + shlex.quote(str(ROOT)) + '; ' + shlex.join([
        sys.executable, str(ROOT / 'tools/verify_correctness.py'),
        '--source', str(args.source.resolve()), '--output', str(output / 'logs')])
    script = '''on run argv
tell application "Terminal"
    activate
    do script (item 1 of argv)
    set w to front window
    try
        set font size of selected tab of w to 14
    end try
    set bounds of w to {60, 60, 1060, 610}
    return id of w
end tell
end run'''
    window_id = subprocess.check_output(['osascript', '-e', script, command], text=True).strip()
    for _ in range(120):
        if (output / 'logs/results.json').exists():
            break
        time.sleep(0.25)
    else:
        raise RuntimeError('Terminal verification did not finish; inspect the new window')
    time.sleep(1)
    screenshot = output / 'correctness.png'
    subprocess.run(['/usr/sbin/screencapture', '-x', '-o', '-l', window_id, str(screenshot)], check=True)
    (output / 'window_id.txt').write_text(window_id + '\n')
    print(screenshot)


if __name__ == '__main__':
    main()
