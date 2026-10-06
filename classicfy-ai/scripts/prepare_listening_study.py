"""Prepare three blind listening schedules; downloading audio requires --download."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from listening_study import (SEED, archive_index, audio_name, choose_tasks, create_audio,
                             digest, fixed_windows, schedules, write_csv)
from validate_embedding import collect_groups, summarize, weighted_distances
from validate_interpretation_examples import vector_rows
from preprocessing import ASAPLoader


def resolve_source(root,row,workspace):
    """Recognize original MAESTRO and already-trimmed ASAP WAVs separately."""
    offset=float(row["start"] or 0);relative=row["maestro_audio_performance"].replace("{maestro}/","")
    if root:
        for p in [root/relative,root/"maestro-v2.0.0"/relative]:
            if p.is_file():return p,offset
        trimmed=root/row["audio_performance"]
        if row["audio_performance"] and trimmed.is_file():return trimmed,0.
    metadata=workspace/"datasets/streaming_sample/metadata.json"
    if metadata.exists():
        stored=json.loads(metadata.read_text())
        p=metadata.parent/"source/maestro_ToA01M.wav"
        if stored.get("source_entry")==audio_name(row) and p.is_file():return p,offset
    return None,offset


def write_report(out, organizer):
    task_rows=[]
    for task in organizer["tasks"]:
        roles="A/B/C" if task["kind"]=="cross" else "첫 연주/둘째 연주"
        keys="<br>".join(f"`{k}`" for k in task["keys"])
        full=" / ".join(f"{x:.3f}" for x in task["whole_distances"])
        excerpt=" / ".join(f"{x:.3f}" for x in task["excerpt_distances"])
        task_rows.append(f"| {task['id']} | {task['kind']} · {task['focus']} | {roles}: {keys} | {full} | {excerpt} |")
    quality=organizer["audio_quality"]
    text=f"""# 청취 평가 준비 기록 — 진행자 전용

**평가 전에 참가자에게 보여주지 않는다.** 참가자에게는 음원과 X/Y 화면만 담은 ZIP을 전달한다.
[실행·응답 분석 안내](../../tools/listening_evaluation/README.md)에서 사용 방법을 확인한다.

- Study ID: `{organizer['study_id']}` / seed: `{organizer['seed']}`
- 11문항, 실제 연주 {len({c['key'] for c in organizer['clips']})}개, WAV {len(quality)}개.
- 발췌 길이: {min(c['duration_seconds'] for c in quality):.2f}–{max(c['duration_seconds'] for c in quality):.2f}초.
- 음원은 저장소 밖 `Classicfy/datasets/listening_evaluation/pack/`, 공유 ZIP은 그 상위에 있다.
- 사람의 응답은 아직 수집하지 않았다. 이 폴더는 선정/준비 자료이며 청취 성공률을 담지 않는다.

## 합리적인 기준을 고정한 방법

기존 ASAP/nASAP 분석의 570개 연주·14차원 임베딩을 그대로 재사용한다. 원래 cohort는 같은
작품·동일 beat grid·최소 5연주·32개 공통 유효 beat·50% coverage 조건이다. 원본 캐시 SHA256은
`{organizer['raw_cache_sha256']}`다. 기존 embeddings.csv와 키 순서와 값까지 일치하는지 확인했다.

공식 MAESTRO WAV 매핑이 있고 같은 cohort에서 음원이 3개 이상인 후보만 비교한다.
feature별로 대상 feature의 거리가 작품 내 쌍 거리의 Q75 이상이고 나머지 네 feature 거리가
중앙값 이하인 쌍을 남긴다. `abs(target/Q75 - 1) + other/max(median, 1e-12)`가 가장 작은
후보를 고르며 동점은 키 순서로 결정하고, feature마다 서로 다른 작품을 사용한다.
극단적 최대 거리만 고르지 않고 기준과 가까운 적격 사례를 선택한다. 후보 전체와 제외되지 않은
선정 비용·분위수는 `selection_pool.csv`에 저장한다. 최종 선택은 청취 전에 고정했다.

S06/S07은 같은 기준 연주의 최근접과 작품 내 Q75 이상 첫 후보다. S08은 동일 음원,
S09는 S01의 X/Y를 뒤집은 반복이다. C01/C02는 기존 다른 작품 검색의 anchor/typical 사례를
다시 고르지 않고 사용한다. 전곡 기준 B가 가깝지만, 청취에서 B를 정답으로 취급하지 않는다.

## 실제 연주와 거리

아래 거리는 다섯 feature block을 동일 가중한 전체 거리다. pair는 첫째–둘째, cross는 A–B / A–C 순이다.
공개 X/Y 위치는 참가자별로 다르며 `organizer.json`의 schedules로 복원한다.

| 문항 | 목적 | 실제 연주 ID | 전곡 거리 | 발췌 거리 |
|---|---|---|---:|---:|
{chr(10).join(task_rows)}

## 발췌를 해석할 때

작품의 25%·60% score interval에서 시작한다. 끝 위치는 원래 분석 cohort 전체 연주의
중앙 재생 시간이 약 16초가 되는 score beat로 결정한다. feature 차이를 탐색해 구간을 이동하지 않는다.
X/Y의 score 구간은 같고 실제 재생 시간은 다를 수 있다. MAESTRO 원본에는 ASAP 시작 offset을
더하며, 이미 잘린 ASAP WAV에는 이 offset을 다시 더하지 않는다.

공통 패턴과 scale은 기존 전체 cohort에서 계산한 것을 유지하고, 두 구간의 공통 유효 beat를
이어 붙여 14차원 요약을 다시 계산한다. 해당 두 구간은 청취용 발췌이며 전곡 전체를 대표한다고
보장하지 않는다. **C01은 발췌에서 A–C가 더 가깝다.** S06/S07도 발췌에서는 거리 차이가
전곡보다 작아진다. 청취 결과를 전곡 최근접에만 맞춰 해석하지 않고 두 기준을 따로 비교한다.

## 파일과 재현

| 파일 | 내용 |
|---|---|
| `selection_pool.csv` | 모든 적격 쌍과 선정 비용·원래 분위수·selected 여부 |
| `selection.json` | 고정 문항·private 순서·전곡/발췌 벡터·source provenance |
| `clip_manifest.csv` | 실제 연주 ID·score beat·ASAP/원본 오디오 시간·opaque 파일명 |
| `audio_sources.csv` | 공식 WAV entry와 직접 받은 음원 경로 |
| `audio_quality.csv` | 정확한 PCM frame·길이·샘플레이트·peak·RMS·SHA256 |
| `organizer.json` | study ID·공개 화면 자료·비공개 키·기준·scale·무결성 기록 |

재현 명령은 저장소 루트에서 `.venv/bin/python classicfy-ai/scripts/prepare_listening_study.py --download`다.
기본 실행은 다운로드하지 않으며 `--audio-root`로 로컬 음원을 받을 수 있다. 전체 ZIP을 받지 않고
선택 음원의 prefix만 가져온다. 같은 WAV에서 추출한 두 ASAP 작품도 별도 시간 구간으로 처리한다.
발췌는 native 16-bit PCM stereo이며 gain·속도·샘플레이트를 바꾸지 않는다.
원본 prefix는 전체 entry CRC를 검증할 수 없으므로 고정 ETag/HTTPS/발췌 SHA256로 검증 범위를 명시한다.
MAESTRO Google LLC / International Piano-e-Competition 출처와 CC BY-NC-SA 4.0 조건은 공개 pack의 LICENSE에 있다.

3인·소수 선정 사례이므로 전곡/전체 작품의 정확도·선호 추천 성능·모든 feature의 필수성을 주장하지 않는다.
MIDI의 작품 내 상대적 특징과 귀로 들리는 절대적 빠르기/음량은 다를 수 있고 녹음 gain·음색·잔향도 섞인다.
동일/반복 문항은 진단 기록이며 응답자를 자동 제외하는 필터가 아니다.
"""
    (out/"README.md").write_text(text)


def build(args):
    ai=Path(__file__).resolve().parents[1];workspace=ai.parent.parent
    args.min_performances=5;args.min_beats=32;args.min_coverage=.5
    args.out.mkdir(parents=True,exist_ok=True)
    groups,_,scales,provenance=collect_groups(args);rows,vectors=vector_rows(groups)
    previous_csv=pd.read_csv(ai/"analysis/_feature_search_roles/embeddings.csv")
    if list(previous_csv.key)!=[r["key"] for r in rows]:raise ValueError("Existing embedding corpus differs")
    np.testing.assert_allclose(vectors,previous_csv.filter(regex="^coordinate_").to_numpy(),atol=1e-12,rtol=0)
    metadata=pd.read_csv(args.asap_root/"metadata.csv").fillna("").set_index("midi_performance")
    available=np.array([audio_name(metadata.loc[r["key"]]) is not None for r in rows])
    previous=json.loads((ai/"analysis/_feature_search_roles/stats.json").read_text())
    tasks,pool=choose_tasks(rows,vectors,available,previous)
    keys=sorted({k for t in tasks for k in t["keys"]});group_for={k:g for g in groups for k in g["keys"]}
    loader=ASAPLoader(args.asap_root,args.nasap_root);windows={};specs=[];sources={};excerpt={}
    for key in keys:
        group=group_for[key];cohort=(group["piece"],group["grid"])
        if cohort not in windows:windows[cohort]=fixed_windows(group,[loader.get_sample(k) for k in group["keys"]])
        parts=windows[cohort];sample=loader.get_sample(key);row=metadata.loc[key]
        source,offset=resolve_source(args.audio_root,row,workspace)
        name=audio_name(row);source_id=digest([name,str(source) if source else None])[:16]
        sources[source_id]=(name,source)
        own=group["keys"].index(key);indices=np.concatenate([p["shared_positions"] for p in parts])
        excerpt[key]=summarize(group["values"][own][:,indices])
        for number,part in enumerate(parts,1):
            a,b=float(sample.performance_beats[part["start"]]),float(sample.performance_beats[part["end"]])
            if row["end"] and float(row["start"] or 0)+b>float(row["end"]):raise ValueError("Excerpt exceeds ASAP trim")
            specs.append(dict(key=key,piece=group["piece"],grid=group["grid"],part=number,
                score_start=part["start"],score_end_exclusive=part["end"],shared_beats=len(part["shared_positions"]),
                audio="audio/"+digest([SEED,key,number])[:16]+".wav",asap_start_seconds=a,asap_end_seconds=b,
                source_start_seconds=offset+a,source_end_seconds=offset+b,source_entry=name,source_id=source_id))
    for task in tasks:
        v=np.array([excerpt[k] for k in task["keys"]]);task["excerpt_vectors"]=v.tolist()
        task["excerpt_distances"]=[float(x) for x in weighted_distances(v[:1],v[1:],"all")[0]]
    orders=schedules(tasks)
    write_csv(args.out/"selection_pool.csv",pool);write_csv(args.out/"clip_manifest.csv",specs)
    write_csv(args.out/"audio_sources.csv",[dict(source_entry=name,local_path=str(p) if p else "",
        download_url="https://magenta.withgoogle.com/datasets/maestro#v200") for _,(name,p) in sorted(sources.items())])
    draft=dict(seed=SEED,tasks=tasks,schedules=orders,clips=specs,provenance=provenance)
    (args.out/"selection.json").write_text(json.dumps(draft,ensure_ascii=False,indent=2))
    if args.plan_only:
        print(f"Plan saved: {len(tasks)} tasks / {len(keys)} performances / {len(specs)} clips. No audio downloaded.")
        return
    archive=archive_index(args.pack.parent/"cache") if args.download else None
    if archive:
        for name,_ in sources.values():
            if name not in archive["entries"]:raise ValueError("Selected WAV is not present in the official ZIP")
    missing=[name for name,p in sources.values() if p is None]
    if missing and not args.download:
        raise ValueError(f"{len(missing)} source WAVs missing. Supply --audio-root, or explicitly add --download. See audio_sources.csv.")
    args.pack.mkdir(parents=True,exist_ok=True);receipts=[]
    by_source={sid:[s for s in specs if s["source_id"]==sid] for sid in sources}
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs={executor.submit(create_audio,sources[sid][0],items,args.pack,args.pack.parent/"receipts",
            local=sources[sid][1],archive=archive):sid for sid,items in by_source.items()}
        for future in as_completed(jobs):
            receipts.append(future.result());print(f"Audio ready {len(receipts)}/{len(jobs)}",flush=True)
    quality=sorted([c for r in receipts for c in r["clips"]],key=lambda c:(c["key"],c["part"]))
    write_csv(args.out/"audio_quality.csv",quality)
    public={"schema_version":1,"participants":{},"task_count":len(tasks)}
    for participant,order in orders.items():
        entries=[]
        for item in order:
            stimuli={}
            for label,field in [("A","reference_key"),("X","x_key"),("Y","y_key")]:
                key=item[field]
                if key:stimuli[label]=[{"audio":s["audio"],"part":s["part"]} for s in specs if s["key"]==key]
            entries.append(dict(task_id=item["task_id"],mode=item["mode"],stimuli=stimuli))
        public["participants"][participant]=entries
    study_id=digest(dict(public=public,quality=[c["audio_sha256"] for c in quality],tasks=tasks))[:20]
    public["study_id"]=study_id
    organizer={**draft,"study_id":study_id,"public":public,"scales":scales,
        "selection_rule":"Audio mappings only, >=3 recordings per cohort; feature distance >=Q75 and remaining-four <=median; minimize target/Q75-1 + other/median; different works. Total near/far uses same Scherzo anchor, nearest vs Q75. Cross-work examples reused without reselection.",
        "excerpt_rule":"Score interval floor(25%/60% of length), end closest to median 16s over full original cohort; scales/common fixed; no feature-based excerpt search",
        "normalization":"Existing 14D / five equal blocks; unchanged source feature extraction. Full vs excerpt summaries both retained.",
        "audio_quality":quality,"downloaded_bytes":sum(r["downloaded_bytes"] for r in receipts),
        "audio_policy":"Native PCM WAV, no loudness normalization/resampling/speed changes; original recording level/timbre can confound hearing.",
        "raw_cache_sha256":hashlib.sha256(args.cache.read_bytes()).hexdigest()}
    (args.out/"organizer.json").write_text(json.dumps(organizer,ensure_ascii=False,indent=2))
    write_report(args.out,organizer)
    web=ai/"tools/listening_evaluation"
    for name in ["index.html","app.js","style.css"]:shutil.copyfile(web/name,args.pack/name)
    (args.pack/"study.js").write_text("window.LISTENING_STUDY = "+json.dumps(public,ensure_ascii=False)+";\n")
    license_text="MAESTRO v2.0.0, Google LLC / International Piano-e-Competition; ASAP mapping.\nCC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/\nhttps://magenta.withgoogle.com/datasets/maestro\nhttps://github.com/fosfrancesco/asap-dataset\nModification: excerpts only; no gain normalization, resampling, speed changes or synthesis.\n"
    (args.pack/"LICENSE.txt").write_text(license_text)
    (args.pack/"START_HERE.txt").write_text("Open index.html in Chrome/Safari/Firefox. Choose a different P1/P2/P3 number per person.\nEvaluate independently, then download JSON. Organizer files are intentionally not included.\n")
    expected={s["audio"] for s in specs}
    for old in (args.pack/"audio").glob("*.wav"):
        if "audio/"+old.name not in expected:old.unlink()
    zipped=args.pack.parent/"listening_participant_pack.zip"
    with zipfile.ZipFile(zipped,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        for path in sorted(args.pack.rglob("*")):
            if path.is_file():out.write(path,path.relative_to(args.pack))
    print(f"Ready: {study_id}, {len(tasks)} tasks, {len(keys)} performances, {len(specs)} WAVs")
    print(f"Participant page: {args.pack/'index.html'}\nShare only: {zipped}\nOrganizer: {args.out/'organizer.json'}")


def main():
    ai=Path(__file__).resolve().parents[1];workspace=ai.parent.parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asap-root",type=Path,default=workspace/"datasets/ASAP")
    p.add_argument("--nasap-root",type=Path,default=workspace/"datasets/nASAP")
    p.add_argument("--cache",type=Path,default=workspace/"datasets/feature_normalization_raw.npz")
    p.add_argument("--out",type=Path,default=ai/"analysis/_listening_evaluation")
    p.add_argument("--pack",type=Path,default=workspace/"datasets/listening_evaluation/pack")
    p.add_argument("--audio-root",type=Path,help="Original MAESTRO root or already-trimmed ASAP WAV root")
    p.add_argument("--download",action="store_true",help="Explicitly download only required prefixes of selected source WAVs")
    p.add_argument("--plan-only",action="store_true",help="Selection/manifests only; no audio/network/pack")
    args=p.parse_args()
    try:build(args)
    except (ValueError,FileNotFoundError) as e:p.exit(2,str(e)+"\n")


if __name__=="__main__":main()
