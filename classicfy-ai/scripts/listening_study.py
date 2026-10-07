"""Shared selection, audio and response validation for an exploratory listening study."""
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import wave
import zlib

import numpy as np

from validate_embedding import BLOCKS, weighted_distances
from validate_feature_search_roles import block_components, cohort_ids

SEED = 20261005
FIELDS = ("overall", "tempo", "rubato", "dynamics", "articulation", "pedaling")
ARCHIVE_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def audio_name(row):
    name = row["maestro_audio_performance"]
    return "maestro-v2.0.0/" + name.replace("{maestro}/", "") if name.endswith(".wav") else None


def choose_tasks(rows, vectors, available, previous):
    """Choose before listening; preserve existing full/excerpt disagreements."""
    comp = block_components(vectors); distances = np.sqrt(comp.mean(axis=0))
    cohorts = cohort_ids(rows); ids = {r["key"]:i for i,r in enumerate(rows)}
    tasks, selection, used = [], [], set()
    for f, name in enumerate(BLOCKS):
        pool = []
        for cohort in sorted(set(cohorts)):
            ix = np.flatnonzero((cohorts == cohort) & available)
            if len(ix) < 3: continue
            a,b = np.triu_indices(len(ix),1); a,b = ix[a],ix[b]
            target = np.sqrt(comp[f,a,b]); other = np.sqrt(np.delete(comp,f,axis=0)[:,a,b].mean(axis=0))
            q75, median = float(np.quantile(target,.75)),float(np.median(other))
            if q75 <= 1e-6: continue
            for i,j,t,o in zip(a,b,target,other):
                if t >= q75 and o <= median:
                    pool.append(dict(feature=name, key_x=rows[i]["key"],key_y=rows[j]["key"],
                        cost=float(abs(t/q75-1)+o/max(median,1e-12)),target_distance=float(t),
                        other_distance=float(o),target_q75=q75,other_median=median))
        pool.sort(key=lambda p:(p["cost"],p["key_x"],p["key_y"]))
        winner = next((p for p in pool if rows[ids[p["key_x"]]]["piece"] not in used),None)
        if winner is None: raise ValueError(f"No eligible audio comparison for {name}")
        used.add(rows[ids[winner["key_x"]]]["piece"])
        tasks.append(dict(id=f"S{f+1:02d}",kind="contrast",focus=name,keys=[winner["key_x"],winner["key_y"]]))
        selection.extend({**p,"selected":p is winner} for p in pool)
    source = ids[tasks[4]["keys"][0]]
    eligible = np.flatnonzero((cohorts==cohorts[source]) & available & (np.arange(len(rows))!=source))
    near = min(eligible,key=lambda j:(distances[source,j],rows[j]["key"]))
    q75 = float(np.quantile(distances[source,eligible],.75))
    far = min((j for j in eligible if distances[source,j]>=q75),key=lambda j:(distances[source,j],rows[j]["key"]))
    for number,kind,j in [(6,"near",near),(7,"far",far)]:
        tasks.append(dict(id=f"S{number:02d}",kind=kind,focus="all",keys=[rows[source]["key"],rows[j]["key"]],
                          selection=dict(rule="nearest" if kind=="near" else "first at/above Q75",q75=q75)))
    tasks.append(dict(id="S08",kind="identical",focus="check",keys=[tasks[1]["keys"][0]]*2))
    tasks.append(dict(id="S09",kind="repeat",focus="Tempo",keys=tasks[0]["keys"].copy(),repeat_of="S01"))
    for number,example in enumerate(["anchor","typical"],1):
        keys = [previous["examples"][example][r]["key"] for r in "ABC"]
        if not all(k in ids and available[ids[k]] for k in keys):
            raise ValueError("Existing cross-work example has no audio mapping")
        tasks.append(dict(id=f"C{number:02d}",kind="cross",focus="all",keys=keys,example=example))
    for task in tasks:
        keys=task["keys"]; vector=np.array([vectors[ids[k]] for k in keys])
        task["whole_vectors"]=vector.tolist()
        task["whole_distances"]=[float(x) for x in weighted_distances(vector[:1],vector[1:],"all")[0]]
        task["piece"]=rows[ids[keys[0]]]["piece"]
        task["grids"]=[rows[ids[k]]["grid"] for k in keys]
    return tasks,selection


def fixed_windows(group, samples, seconds=16):
    """Fixed score positions; end determined by timing, never feature contrast."""
    times=np.array([s.performance_beats for s in samples]);n=times.shape[1]-1
    windows=[]
    for fraction in [.25,.60]:
        start=int(fraction*n)
        duration=np.median(times[:,start+1:]-times[:,start,None],axis=0)
        end=start+int(np.argmin(abs(duration-seconds)))+1
        shared=np.flatnonzero((group["shared_indices"]>=start)&(group["shared_indices"]<end))
        if len(shared)<2:raise ValueError("Fixed excerpt has fewer than two shared valid beats")
        windows.append(dict(start=start,end=end,shared_positions=shared.tolist()))
    return windows


def schedules(tasks):
    rng=np.random.default_rng(SEED)
    base=list(rng.permutation([t["id"] for t in tasks if t["kind"]!="cross"]))
    while min(abs(base.index("S01")-base.index("S09")),len(base)-abs(base.index("S01")-base.index("S09")))<3:
        base=list(rng.permutation(base))
    result={}
    for p in range(3):
        order=base[p*3:]+base[:p*3]+(["C01","C02"] if p%2==0 else ["C02","C01"])
        entries=[]
        for tid in order:
            task=next(t for t in tasks if t["id"]==tid);i=tasks.index(task)
            keys=task["keys"][-2:]
            if (p+i+(task["kind"]=="repeat"))%2:keys=keys[::-1]
            entries.append(dict(task_id=tid,mode="cross" if task["kind"]=="cross" else "pair",
                reference_key=task["keys"][0] if task["kind"]=="cross" else None,x_key=keys[0],y_key=keys[1]))
        result[f"P{p+1}"]=entries
    return result


def fetch_range(start,end,etag):
    with tempfile.TemporaryDirectory() as tmp:
        body,headers=Path(tmp)/"body",Path(tmp)/"headers"
        subprocess.run(["curl","-fsS","--retry","2","--max-time","90","-H",f"If-Match: {etag}",
                        "-r",f"{start}-{end}","-D",str(headers),"-o",str(body),ARCHIVE_URL],check=True)
        raw=body.read_bytes()
        if len(raw)!=end-start+1 or f"content-range: bytes {start}-{end}/" not in headers.read_text().lower():
            raise ValueError("Archive response did not match requested byte range")
        return raw


def parse_directory(raw):
    entries,pos={},0
    while pos<len(raw):
        h=struct.unpack_from("<4s6H3I5H2I",raw,pos)
        if h[0]!=b"PK\x01\x02":raise ValueError("Bad ZIP directory")
        n,e,c=h[10:13];name=raw[pos+46:pos+46+n].decode();extra=raw[pos+46+n:pos+46+n+e]
        values=[h[9],h[8],h[-1]];cursor=0
        while cursor+4<=len(extra):
            tag,size=struct.unpack_from("<HH",extra,cursor);data=extra[cursor+4:cursor+4+size];cursor+=4+size
            if tag==1:
                offset=0
                for i,value in enumerate(values):
                    if value==0xffffffff:values[i]=struct.unpack_from("<Q",data,offset)[0];offset+=8
        entries[name]=dict(size=values[0],compressed=values[1],offset=values[2],method=h[4],crc=h[7])
        pos+=46+n+e+c
    return entries


def archive_index(cache):
    raw=subprocess.check_output(["curl","-fsSI","--max-time","30",ARCHIVE_URL],text=True)
    fields={k.lower():v.strip() for k,v in (line.split(":",1) for line in raw.splitlines() if ":" in line)}
    size,etag=int(fields["content-length"]),fields["etag"]
    path=cache/"archive_index.json";cache.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=json.loads(path.read_text())
        if old["etag"]==etag and old["size"]==size:return old
    tail=fetch_range(size-65536,size-1,etag)
    eocd=struct.unpack_from("<4s4H2IH",tail,tail.rfind(b"PK\x05\x06"));length,offset=eocd[5:7]
    if offset==0xffffffff or length==0xffffffff:
        z=struct.unpack_from("<4sQ2H2I4Q",tail,tail.rfind(b"PK\x06\x06"));length,offset=z[-2:]
    raw=fetch_range(offset,offset+length-1,etag)
    result=dict(url=ARCHIVE_URL,size=size,etag=etag,directory_sha256=hashlib.sha256(raw).hexdigest(),entries=parse_directory(raw))
    path.write_text(json.dumps(result));return result


def wav_layout(raw):
    if len(raw)<12:return None
    if raw[:4]!=b"RIFF" or raw[8:12]!=b"WAVE":raise ValueError("Expected WAV")
    pos,fmt=12,None
    while pos+8<=len(raw):
        tag,size=struct.unpack_from("<4sI",raw,pos)
        if tag==b"data":
            if fmt is None:raise ValueError("WAV data before format")
            return fmt,pos+8,size
        if pos+8+size>len(raw):return None
        if tag==b"fmt ":
            code,ch,rate,_,align,bits=struct.unpack_from("<HHIIHH",raw,pos+8)
            if code!=1 or bits!=16 or align!=ch*2:raise ValueError("Expected 16-bit PCM WAV")
            fmt=(ch,rate,align)
        pos+=8+size+size%2
    return None


def write_clip(path,pcm,channels,rate):
    path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),"wb") as audio:
        audio.setnchannels(channels);audio.setsampwidth(2);audio.setframerate(rate);audio.writeframes(pcm)
    signal=np.frombuffer(pcm,dtype="<i2").astype(float)/32768
    return dict(audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sample_rate=rate,channels=channels,
                duration_seconds=len(pcm)/2/channels/rate,peak=float(abs(signal).max()),rms=float(np.sqrt(np.mean(signal**2))))


def create_audio(name,specs,pack,receipt_dir,*,local=None,archive=None):
    """Use supplied WAV or download only the prefix needed for selected excerpts."""
    receipt=receipt_dir/(digest([name,str(local) if local else None])[:16]+".json")
    signature=digest(dict(name=name,specs=specs,local=str(local) if local else None,
                         local_mtime=local.stat().st_mtime_ns if local else None,etag=archive["etag"] if archive else None))
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old["signature"]==signature and all((pack/s["audio"]).exists() and hashlib.sha256((pack/s["audio"]).read_bytes()).hexdigest()==s["audio_sha256"] for s in old["clips"]):return old
    buffers=[];frames=[];received=0
    if local:
        with wave.open(str(local),"rb") as audio:
            rate,ch=audio.getframerate(),audio.getnchannels()
            if audio.getsampwidth()!=2:raise ValueError("Expected 16-bit source")
            for s in specs:
                a,b=round(s["source_start_seconds"]*rate),round(s["source_end_seconds"]*rate)
                if not 0<=a<b<=audio.getnframes():raise ValueError("Excerpt outside source")
                audio.setpos(a);buffers.append(audio.readframes(b-a));frames.append((a,b))
    else:
        if archive is None:raise ValueError("Audio is missing; supply audio root or explicitly use --download")
        entry=archive["entries"][name];off=entry["offset"]
        h=struct.unpack("<4s5H3I2H",fetch_range(off,off+29,archive["etag"]))
        if h[0]!=b"PK\x03\x04" or h[3]!=8 or h[2]&1:raise ValueError("Unsupported ZIP entry")
        start=off+30+h[-2]+h[-1];inflater=zlib.decompressobj(-15)
        position,prefix,layout=0,b"",None;buffers=[bytearray() for _ in specs];bounds=None
        while received<entry["compressed"]:
            length=min(8*1024*1024,entry["compressed"]-received)
            chunk=fetch_range(start+received,start+received+length-1,archive["etag"]);received+=len(chunk)
            raw=inflater.decompress(chunk)
            if layout is None:
                prefix+=raw;layout=wav_layout(prefix)
                if layout is None:continue
                raw=prefix;prefix=b"";position=0
                (ch,rate,align),data_offset,data_size=layout
                frames=[(round(s["source_start_seconds"]*rate),round(s["source_end_seconds"]*rate)) for s in specs]
                bounds=[(data_offset+a*align,data_offset+b*align) for a,b in frames]
                if any(a<data_offset or a>=b or b>data_offset+data_size for a,b in bounds):raise ValueError("Excerpt outside WAV data")
            for buf,(a,b) in zip(buffers,bounds):
                lo,hi=max(position,a),min(position+len(raw),b)
                if lo<hi:buf.extend(raw[lo-position:hi-position])
            position+=len(raw)
            if position>=max(b for a,b in bounds):break
        if layout is None:raise ValueError("Incomplete WAV header")
    clips=[]
    for spec,pcm,(a,b) in zip(specs,buffers,frames):
        if len(pcm)!=(b-a)*ch*2:raise ValueError("Incomplete PCM excerpt")
        clips.append({**spec,**write_clip(pack/spec["audio"],pcm,ch,rate),"start_frame":a,"end_frame":b})
    result=dict(signature=signature,source=name,local_source=str(local) if local else None,
                downloaded_bytes=received,integrity="local source" if local else "HTTPS, pinned ETag, excerpt SHA256; prefix has no full-source CRC check",clips=clips)
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2));return result


def validate_response(data,organizer,*,allow_partial=False):
    """Reject stale study, altered schedules, malformed answers and invalid scales."""
    if not isinstance(data,dict) or data.get("schema_version")!=1 or data.get("study_id")!=organizer["study_id"]:
        raise ValueError("Response belongs to a different study/version")
    participant=data.get("participant")
    if not isinstance(participant,str) or participant not in organizer["public"]["participants"]:raise ValueError("Unknown participant")
    expected=organizer["public"]["participants"][participant]
    if data.get("schedule")!=expected:raise ValueError("Response schedule differs from generated study")
    answers=data.get("answers")
    if not isinstance(answers,dict):raise ValueError("Answers must be a mapping")
    ids={t["task_id"] for t in expected}
    if set(answers)-ids or (not allow_partial and set(answers)!=ids):raise ValueError("Unexpected or missing tasks")
    rows=[]
    for t in expected:
        if t["task_id"] not in answers:continue
        answer=answers[t["task_id"]]
        if not isinstance(answer,dict):raise ValueError("Malformed answer")
        allowed={"X","Y","tie","unclear"} if t["mode"]=="cross" else {"1","2","3","4","5","unclear"}
        for field in FIELDS:
            if not isinstance(answer.get(field),str) or answer[field] not in allowed:
                if allow_partial and field not in answer:continue
                raise ValueError(f"Invalid {field} rating")
        if not isinstance(answer.get("confidence"),str) or answer["confidence"] not in {"1","2","3"}:
            if not (allow_partial and "confidence" not in answer):raise ValueError("Invalid confidence")
        comment=answer.get("comment","")
        if not isinstance(comment,str) or len(comment)>10000:raise ValueError("Invalid comment")
        if all(f in answer for f in (*FIELDS,"confidence")):
            rows.append(dict(participant=participant,task_id=t["task_id"],mode=t["mode"],
                **{f:answer[f] for f in FIELDS},confidence=answer["confidence"],comment=comment))
    return rows
