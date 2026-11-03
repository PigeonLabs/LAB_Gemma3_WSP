"""Presentation only: render stored calibration results, never rerun inference."""
import html
import json
from pathlib import Path


def render(root, result):
    rows=[('Development: '+name,acc) for name,acc in result['development_accuracies'].items()]
    rows.append(('Independent calibration',result['calibration_accuracy']))
    passed=result['status']=='passed'
    elements=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="310" viewBox="0 0 920 310" role="img" aria-labelledby="title description">',
              '<title id="title">Preregistered observer calibration</title>',
              '<desc id="description">Accuracy by development format and the separately held-out calibration set. The gate requires 80 percent on both development and calibration.</desc>',
              '<rect width="920" height="310" rx="14" fill="#111827"/>',
              '<g font-family="sans-serif" fill="#f9fafb">',
              '<text x="26" y="37" font-size="22">Observer calibration: '+('PASS' if passed else 'STOP')+'</text>',
              '<text x="26" y="62" font-size="14" fill="#cbd5e1">Format chosen on development data; independent gate checked once</text>']
    for i,(label,accuracy) in enumerate(rows):
        y=90+36*i
        color='#22c55e' if accuracy>=.8 else '#f59e0b'
        elements += [f'<text x="26" y="{y+18}" font-size="15">{html.escape(label)}</text>',
                     f'<rect x="290" y="{y}" width="500" height="26" rx="4" fill="#334155"/>',
                     f'<rect x="290" y="{y}" width="{500*accuracy:.2f}" height="26" rx="4" fill="{color}"/>',
                     f'<text x="812" y="{y+18}" font-size="16">{accuracy:.1%}</text>']
    elements += ['<line x1="690" x2="690" y1="80" y2="236" stroke="#f8fafc" stroke-width="2" stroke-dasharray="5 4"/>',
                 '<text x="652" y="257" font-size="13">80% gate</text>',
                 f'<text x="26" y="285" font-size="13" fill="#cbd5e1">Independent: {result["correct"]}/{result["denominator"]}; UNKNOWN counts incorrect. No attack outcome is shown here.</text>',
                 '</g></svg>']
    (root/'docs/figures/study_calibration.svg').write_text('\n'.join(elements)+'\n')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    render(root,json.loads((root/'artifacts/study_v1/calibration_summary.json').read_text()))
