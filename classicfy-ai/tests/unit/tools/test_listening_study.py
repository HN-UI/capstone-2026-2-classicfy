"""Protect blind schedules, real PCM excerpt offsets and honest response reporting."""
from copy import deepcopy
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave
import zipfile

import numpy as np

AI=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(AI/"scripts"))
from listening_study import (FIELDS, audio_name, create_audio, fixed_windows, parse_directory,
                             schedules, validate_response, wav_layout)
from prepare_listening_study import resolve_source
from analyze_listening_study import closest, collect, summarize_responses
spec=importlib.util.spec_from_file_location("listening_server",AI/"tools/listening_evaluation/serve.py")
server_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(server_module)


def fixture():
    tasks=[]
    for i in range(1,10):
        tasks.append(dict(id=f"S{i:02d}",kind="identical" if i==8 else "repeat" if i==9 else "contrast",
            focus="Tempo",keys=["a","b"],whole_vectors=[[0]*14,[1]*14],
            excerpt_vectors=[[0]*14,[1]*14],whole_distances=[1.],excerpt_distances=[1.]))
    for i in (1,2):
        tasks.append(dict(id=f"C{i:02d}",kind="cross",focus="all",keys=["a","b","c"],
            whole_vectors=[[0]*14,[1]*14,[2]*14],excerpt_vectors=[[0]*14,[2]*14,[1]*14],
            whole_distances=[1.,2.],excerpt_distances=[2.,1.]))
    orders=schedules(tasks)
    public={p:[dict(task_id=t["task_id"],mode=t["mode"],stimuli={}) for t in order] for p,order in orders.items()}
    organizer=dict(study_id="study-test",tasks=tasks,schedules=orders,public=dict(participants=public))
    responses=[]
    for p,order in public.items():
        answers={t["task_id"]:{**{f:"3" if t["mode"]=="pair" else "X" for f in FIELDS},"confidence":"2","comment":""} for t in order}
        responses.append(dict(schema_version=1,study_id="study-test",participant=p,schedule=order,answers=answers))
    return organizer,responses


def wave_bytes():
    frames=np.arange(2000,dtype="<i2").reshape(1000,2)
    buf=io.BytesIO()
    with wave.open(buf,"wb") as f:
        f.setnchannels(2);f.setsampwidth(2);f.setframerate(100);f.writeframes(frames.tobytes())
    return buf.getvalue(),frames


class ListeningStudyTest(unittest.TestCase):
    def test_blank_audio_mapping_is_not_a_zip_entry(self):
        self.assertIsNone(audio_name(dict(maestro_audio_performance="")))
        self.assertEqual(audio_name(dict(maestro_audio_performance="{maestro}/2017/a.wav")),"maestro-v2.0.0/2017/a.wav")

    def test_zip64_directory_reads_independent_large_sizes_and_offset(self):
        name=b"example.wav";extra=struct.pack("<HHQQQ",1,24,2**33,2**32+7,2**34)
        header=struct.pack("<4s6H3I5H2I",b"PK\x01\x02",45,45,0,8,0,0,123,0xffffffff,0xffffffff,len(name),len(extra),0,0,0,0,0xffffffff)
        result=parse_directory(header+name+extra)[name.decode()]
        self.assertEqual((result["size"],result["compressed"],result["offset"]),(2**33,2**32+7,2**34))

    def test_partial_wav_header_waits_and_rejects_wrong_format(self):
        raw,_=wave_bytes();self.assertIsNone(wav_layout(raw[:25]))
        self.assertEqual(wav_layout(raw[:44]),((2,100,4),44,4000))
        with self.assertRaises(ValueError):wav_layout(b"not wave at all")

    def test_local_pcm_exact_frames_no_gain_or_resampling(self):
        raw,frames=wave_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source.wav";source.write_bytes(raw)
            specs=[dict(key="a",part=1,audio="audio/a.wav",source_start_seconds=.1,source_end_seconds=.3)]
            result=create_audio("source",specs,root/"pack",root/"receipts",local=source)
            with wave.open(str(root/"pack/audio/a.wav")) as f:
                self.assertEqual(f.getframerate(),100);self.assertEqual(f.getnframes(),20)
                self.assertEqual(f.readframes(20),frames[10:30].tobytes())
            self.assertEqual(result["downloaded_bytes"],0)
            self.assertEqual(result["clips"][0]["duration_seconds"],.2)

    def test_remote_prefix_extracts_pcm_and_reuses_verified_cache(self):
        raw,frames=wave_bytes();buf=io.BytesIO();name="source.wav"
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:z.writestr(name,raw)
        zipped=buf.getvalue();entry=parse_directory(zipped[zipped.find(b"PK\x01\x02"):zipped.find(b"PK\x05\x06")])[name]
        archive=dict(etag='"test"',entries={name:entry})
        def fetch(a,b,etag):self.assertEqual(etag,'"test"');return zipped[a:b+1]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);specs=[dict(key="a",part=1,audio="audio/a.wav",source_start_seconds=.6,source_end_seconds=.8)]
            with patch("listening_study.fetch_range",side_effect=fetch) as download:
                result=create_audio(name,specs,root/"pack",root/"receipts",archive=archive)
                count=download.call_count
                self.assertEqual(create_audio(name,specs,root/"pack",root/"receipts",archive=archive),result)
                self.assertEqual(download.call_count,count)
            with wave.open(str(root/"pack/audio/a.wav")) as f:self.assertEqual(f.readframes(20),frames[60:80].tobytes())
            self.assertIn("no full-source CRC",result["integrity"])

    def test_truncated_remote_source_is_rejected(self):
        raw,_=wave_bytes();buf=io.BytesIO();name="source.wav"
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:z.writestr(name,raw[:-100])
        zipped=buf.getvalue();entry=parse_directory(zipped[zipped.find(b"PK\x01\x02"):zipped.find(b"PK\x05\x06")])[name]
        with tempfile.TemporaryDirectory() as tmp,patch("listening_study.fetch_range",side_effect=lambda a,b,e:zipped[a:b+1]):
            with self.assertRaisesRegex(ValueError,"Incomplete PCM"):
                create_audio(name,[dict(audio="audio/a.wav",source_start_seconds=9.,source_end_seconds=10.)],Path(tmp)/"pack",Path(tmp)/"receipts",archive=dict(etag="e",entries={name:entry}))

    def test_two_trimmed_pieces_sharing_archive_name_keep_separate_caches(self):
        raw,_=wave_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);specs=[dict(key="a",part=1,audio="audio/a.wav",source_start_seconds=0.,source_end_seconds=.1)]
            for name in ("prelude","fugue"):
                source=root/(name+".wav");source.write_bytes(raw)
                create_audio("same-maestro-entry",specs,root/"pack",root/"receipts",local=source)
            self.assertEqual(len(list((root/"receipts").glob("*.json"))),2)

    def test_incomplete_remote_header_cannot_create_an_empty_receipt(self):
        raw,_=wave_bytes();buf=io.BytesIO();name="source.wav"
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:z.writestr(name,raw[:24])
        zipped=buf.getvalue();entry=parse_directory(zipped[zipped.find(b"PK\x01\x02"):zipped.find(b"PK\x05\x06")])[name]
        with tempfile.TemporaryDirectory() as tmp,patch("listening_study.fetch_range",side_effect=lambda a,b,e:zipped[a:b+1]):
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,"Incomplete WAV header"):
                create_audio(name,[dict(audio="audio/a.wav",source_start_seconds=0.,source_end_seconds=.1)],root/"pack",root/"receipts",archive=dict(etag="e",entries={name:entry}))
            self.assertFalse((root/"receipts").exists())

    def test_original_and_trimmed_audio_offsets_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row=dict(start="20",maestro_audio_performance="{maestro}/2017/a.wav",audio_performance="Bach/a.wav")
            (root/"2017").mkdir();original=root/"2017/a.wav";original.touch()
            self.assertEqual(resolve_source(root,row,root),(original,20.))
            original.unlink();(root/"Bach").mkdir();trimmed=root/"Bach/a.wav";trimmed.touch()
            self.assertEqual(resolve_source(root,row,root),(trimmed,0.))

    def test_windows_use_shared_score_positions_and_cohort_median_time(self):
        times=np.arange(101,dtype=float)
        group=dict(shared_indices=np.arange(100))
        windows=fixed_windows(group,[SimpleNamespace(performance_beats=times),SimpleNamespace(performance_beats=times*2)],seconds=16)
        self.assertEqual([(w["start"],w["end"]) for w in windows],[(25,36),(60,71)])
        self.assertEqual(windows[0]["shared_positions"],list(range(25,36)))

    def test_schedules_hide_roles_balance_order_and_reverse_repeat(self):
        organizer,_=fixture();orders=organizer["schedules"]
        for order in orders.values():
            self.assertEqual(len(order),11);self.assertEqual(len({t["task_id"] for t in order}),11)
            ids=[t["task_id"] for t in order]
            self.assertGreaterEqual(abs(ids.index("S01")-ids.index("S09")),3)
            first,repeat=[next(t for t in order if t["task_id"]==i) for i in ("S01","S09")]
            self.assertEqual((first["x_key"],first["y_key"]),(repeat["y_key"],repeat["x_key"]))
        self.assertNotEqual(orders["P1"][0]["task_id"],orders["P2"][0]["task_id"])

    def test_complete_response_and_three_person_collection(self):
        organizer,responses=fixture();rows,ids=collect(responses,organizer)
        self.assertEqual(len(rows),33);self.assertEqual(ids,["P1","P2","P3"])

    def test_stale_or_altered_schedule_rejected(self):
        organizer,responses=fixture()
        for field,value in [("study_id","stale"),("participant",[]),("schedule",[])]:
            data=deepcopy(responses[0]);data[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_response(data,organizer)

    def test_missing_null_and_malformed_ratings_rejected(self):
        organizer,responses=fixture()
        for value in [None,[],dict(overall="6"),{**responses[0]["answers"]["S01"],"confidence":[]},
                      {**responses[0]["answers"]["S01"],"tempo":True}]:
            data=deepcopy(responses[0]);data["answers"]["S01"]=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_response(data,organizer)
        data=deepcopy(responses[0]);del data["answers"]["S01"]
        with self.assertRaises(ValueError):validate_response(data,organizer)

    def test_partial_response_is_explicit_and_duplicate_id_rejected(self):
        organizer,responses=fixture();data=deepcopy(responses[0]);data["answers"]={"S01":data["answers"]["S01"],"S02":dict(overall="3")}
        self.assertEqual(len(validate_response(data,organizer,allow_partial=True)),1)
        with self.assertRaises(ValueError):collect([data],organizer)
        self.assertEqual(len(collect([data],organizer,True)[0]),1)
        with self.assertRaisesRegex(ValueError,"Duplicate"):collect([responses[0],responses[0]],organizer,True)

    def test_cross_choices_remap_blind_labels_and_keep_unclear_denominator(self):
        organizer,responses=fixture()
        for i,data in enumerate(responses):
            task=next(t for t in organizer["schedules"][data["participant"]] if t["task_id"]=="C01")
            data["answers"]["C01"]["overall"]=("X" if task["x_key"]=="b" else "Y") if i==0 else "tie" if i==1 else "unclear"
        rows,_=collect(responses,organizer);_,cross,_=summarize_responses(rows,organizer)
        result=next(r for r in cross if r["task_id"]=="C01" and r["field"]=="overall")
        self.assertEqual((result["B"],result["C"],result["tie"],result["unclear"]),(1,0,1,1))
        self.assertEqual((result["n_decisive"],result["whole_agreement"],result["excerpt_agreement"]),(1,1.,0.))
        self.assertEqual((result["whole_closest"],result["excerpt_closest"]),("B","C"))

    def test_numeric_median_excludes_unclear_and_repeat_keeps_raters(self):
        organizer,responses=fixture()
        for data,value in zip(responses,["1","5","unclear"]):data["answers"]["S01"]["overall"]=value
        rows,_=collect(responses,organizer);pairs,_,checks=summarize_responses(rows,organizer)
        result=next(p for p in pairs if p["task_id"]=="S01" and p["field"]=="overall")
        self.assertEqual((result["median_rating"],result["n_numeric"],result["n_unclear"]),(3.,2,1))
        self.assertEqual(len(checks),18);self.assertIsNone(checks[-6]["absolute_repeat_difference"])

    def test_candidate_tie_and_equal_block_weight(self):
        values=np.zeros((3,14));values[1,8:]=1;values[2,:2]=1
        self.assertEqual(closest(values,"overall"),"tie")
        self.assertEqual(closest(values,"pedaling"),"C")


class QuietHandler(server_module.RangeHandler):
    def log_message(self,*args):pass


class ListeningHTTPTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/"pack";self.root.mkdir()
        self.raw=bytes(range(256))*4;(self.root/"sample.wav").write_bytes(self.raw)
        outside=Path(self.tmp.name)/"outside.txt";outside.write_text("private")
        (self.root/"private.wav").symlink_to(outside)
        (self.root/"folder").mkdir();(self.root/"folder/index.html").symlink_to(outside)
        self.server=ThreadingHTTPServer(("127.0.0.1",0),partial(QuietHandler,directory=str(self.root)))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def request(self,path,range_header=None,method="GET"):
        c=HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        c.request(method,path,headers={"Range":range_header} if range_header else {})
        r=c.getresponse();result=r.status,dict(r.getheaders()),r.read();c.close();return result

    def test_real_http_range_open_suffix_and_head(self):
        for value,lo,hi in [("bytes=10-19",10,19),("bytes=1000-",1000,1023),("bytes=-5",1019,1023)]:
            with self.subTest(value=value):
                status,headers,body=self.request("/sample.wav",value)
                self.assertEqual(status,206);self.assertEqual(body,self.raw[lo:hi+1])
                self.assertEqual(headers["Content-Range"],f"bytes {lo}-{hi}/1024")
                self.assertEqual(headers["Accept-Ranges"],"bytes")
        status,headers,body=self.request("/sample.wav","bytes=10-19","HEAD")
        self.assertEqual((status,headers["Content-Length"],body),(206,"10",b""))

    def test_unsatisfiable_and_multiple_ranges_return_416(self):
        for value in ("bytes=2000-","bytes=9-2","bytes=-0","bytes=0-1,5-6"):
            with self.subTest(value=value):
                status,headers,body=self.request("/sample.wav",value)
                self.assertEqual((status,headers["Content-Range"],body),(416,"bytes */1024",b""))

    def test_pack_does_not_expose_external_symlinks_or_directory_listings(self):
        self.assertEqual(self.request("/private.wav")[0],403)
        self.assertEqual(self.request("/folder/")[0],403)
        self.assertEqual(self.request("/")[0],404)


if __name__=="__main__":unittest.main()
