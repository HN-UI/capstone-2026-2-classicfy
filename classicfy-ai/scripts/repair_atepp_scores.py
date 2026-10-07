"""Recover empty bundled score MIDI from XML, retaining the original files."""
from pathlib import Path
import sys
import hashlib
import warnings
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preprocessing.atepp_alignment import load_score_midi


def main():
    from music21 import converter
    root=Path('../ATEPP_dataset').resolve(); rows=[]
    targets=list((root/'ATEPP-1.2').rglob('*.midi'))
    for p in targets:
        try:
            notes=len(load_score_midi(p))
            if notes>=20: continue
            source=Path(str(p)[:-5])
            if not source.is_file(): continue
            with warnings.catch_warnings(record=True) as caught:
                s=converter.parse(str(source)); expanded=False
                try: s=s.expandRepeats(); expanded=True
                except Exception: pass
                dst=root/'repaired_scores'/(hashlib.sha256(str(p).encode()).hexdigest()[:16]+'.mid')
                dst.parent.mkdir(parents=True,exist_ok=True); s.write('midi',fp=str(dst))
            new_notes=len(load_score_midi(dst))
            rows.append(dict(original_score=str(p),source_xml=str(source),repaired_score=str(dst),
                             original_notes=notes,repaired_notes=new_notes,expanded_repeats=expanded,
                             warnings=';'.join(str(w.message) for w in caught),
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             repaired_sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
            print('Recovered',p.parent.name,notes,'->',new_notes,flush=True)
        except Exception as e: rows.append(dict(original_score=str(p),error=f'{type(e).__name__}: {e}'))
    pd.DataFrame(rows).to_csv(root/'score_repairs.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__': main()
