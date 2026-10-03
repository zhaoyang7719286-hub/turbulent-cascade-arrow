#!/usr/bin/env python3
"""Rebuild all ten paper figures and exported panel data from shipped tables."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_repository import validate

FIGURES = {
    'fig1': ('main', 'Fig1_Intervention', 295.0),
    'fig2': ('main', 'Fig2_Static_Erasure', 340.0),
    'fig3': ('main', 'Fig3_Dynamic_Recovery', 293.0),
    'fig4': ('main', 'Fig4_Geometry_Association', 357.0),
    'figs1': ('supplement', 'FigS1_Flux_Decomposition', 202.0),
    'figs2': ('supplement', 'FigS2_Numerical_Admissibility', 321.0),
    'figs3': ('supplement', 'FigS3_Window_Robustness', 326.0),
    'figs4': ('supplement', 'FigS4_Extended_Static', 338.0),
    'figs5': ('supplement', 'FigS5_Local_Gradient', 332.0),
    'figs6': ('supplement', 'FigS6_Normalization_Test', 327.0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/paper_figures')
    args = parser.parse_args()
    validate(ROOT)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    for kind, files in [('main', ['build_figures.py', 'redesign.py']),
                        ('supplement', ['build_supplement.py', 'normalization_diagnostic.py'])]:
        stage = out / kind
        stage.mkdir()
        shutil.copytree(ROOT / 'data/paper_sources', stage / 'sources')
        for name in files:
            shutil.copyfile(ROOT / 'code/figures' / name, stage / name)
        entries = [dict(path=str(p.relative_to(stage)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted((stage / 'sources').rglob('*')) if p.is_file()]
        (stage / 'source_hashes.json').write_text(json.dumps(entries, indent=2) + '\n')
        (stage / 'extra_source_hashes.json').write_text('[]\n')
        subprocess.run([sys.executable, str(stage / files[0])], cwd=stage, check=True)
    final = out / 'figures'
    final.mkdir()
    for name, (kind, source, height) in FIGURES.items():
        pdf = out / kind / 'figures' / (source + '.pdf')
        svg = pdf.with_suffix('.svg')
        with fitz.open(pdf) as d:
            page = d[0]
            page.set_cropbox(fitz.Rect(0, 0, page.rect.width, height))
            d.save(final / (name + '.pdf'), garbage=4, deflate=True)
        tree = ET.parse(svg)
        element = tree.getroot()
        width = float(element.attrib['viewBox'].split()[2])
        element.set('height', f'{height:g}pt')
        element.set('viewBox', f'0 0 {width:g} {height:g}')
        tree.write(final / (name + '.svg'), encoding='utf-8', xml_declaration=True)
    print(json.dumps({'status': 'PASS', 'paper_figures': 10, 'output': str(out),
                      'scope': 'Figures rebuilt from supplied tables; raw static fields were not recomputed.'}))


if __name__ == '__main__':
    main()
