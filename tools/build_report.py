#!/usr/bin/env python3
"""Render the project's small Markdown report subset to a paginated Chinese PDF."""
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def escape(value):
    substitutions = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                     '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(substitutions.get(c, c) for c in value)


def inline(value):
    pattern = r'(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))'
    parts = re.split(pattern, value)
    output = []
    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            body = part[1:-1]
            if '/' in body and ' ' not in body:
                output.append(r'\nolinkurl{' + body + '}')
            else:
                output.append(r'\texttt{' + escape(body).replace(' ', r'\allowbreak\ ') + '}')
        elif part.startswith('**') and part.endswith('**'):
            output.append(r'\textbf{' + escape(part[2:-2]) + '}')
        elif re.fullmatch(r'\[[^\]]+\]\([^)]+\)', part):
            match = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', part)
            label, target = match.groups()
            output.append(r'\href{' + target + '}{' + escape(label) + '}' if target.startswith('https://') else escape(label))
        else:
            output.append(escape(part))
    return ''.join(output)


def render(markdown):
    lines = markdown.splitlines()
    output = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            output.append(r'\par')
        elif line == '<!-- pagebreak -->':
            output.append(r'\newpage')
        elif line == '<!-- appendix -->':
            output.append(r'\newpage\appendix')
        elif line.startswith('# '):
            output.append(r'{\centering\LARGE\bfseries ' + inline(line[2:]) + r'\par}\vspace{5pt}')
        elif line.startswith('## '):
            output.append(r'\section*{' + inline(line[3:]) + '}')
        elif line.startswith('### '):
            output.append(r'\subsection*{' + inline(line[4:]) + '}')
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            path = (ROOT / match[2]).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            output.append(r'\begin{center}\includegraphics[width=\linewidth,height=0.66\textheight,keepaspectratio]{'
                          + str(path) + r'}\par\small ' + escape(match[1]) + r'\end{center}')
        elif line.startswith('|'):
            table = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                cells = [c.strip() for c in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    table.append(cells)
                index += 1
            columns = len(table[0])
            formats = ['X']
            for col in range(1, columns):
                long_text = any(len(re.findall(r'[\u4e00-\u9fff]', row[col])) > 7 for row in table[1:])
                formats.append('X' if long_text else 'r')
            output.append(r'\begin{center}\small\begin{tabularx}{\linewidth}{@{}' + ''.join(formats) + r'@{}}\toprule')
            for row_index, row in enumerate(table):
                output.append(' & '.join(inline(c) for c in row) + r' \\')
                if row_index == 0:
                    output.append(r'\midrule')
            output.append(r'\bottomrule\end{tabularx}\end{center}')
            continue
        elif line.startswith('- '):
            output.append(r'\noindent\textbullet\quad ' + inline(line[2:]) + r'\par')
        else:
            output.append(inline(line))
        index += 1
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'report.md')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/final_report/pdf')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preamble = r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec,xeCJK}
\setCJKmainfont{Songti SC}\setCJKsansfont{Heiti SC}\setCJKmonofont{Songti SC}
\usepackage[margin=19mm,top=17mm,bottom=18mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,hyperref,url}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={OpenMP CSR SpMV 实验报告}}
\makeatletter
\renewcommand\section{\@startsection{section}{1}{0pt}{7pt}{4pt}{\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{0pt}{5pt}{3pt}{\normalsize\bfseries}}
\makeatother
\setlength{\parskip}{3pt}\setlength{\parindent}{1.5em}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}
\linespread{1.12}\pagestyle{plain}\raggedbottom
\begin{document}\fontsize{10.5}{13}\selectfont
'''
    tex = preamble + render(args.source.read_text()) + '\n' + r'\end{document}' + '\n'
    (output / 'report.tex').write_text(tex)
    for run in range(2):
        result = subprocess.run(['/Library/TeX/texbin/xelatex', '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                                cwd=output, capture_output=True, text=True)
        (output / f'build_{run + 1}.txt').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout[-4500:] + result.stderr)
    print(output / 'report.pdf')


if __name__ == '__main__':
    main()
