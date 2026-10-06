"""Serve the participant pack locally with HTTP byte-range audio seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


def byte_range(value,size):
    match=re.fullmatch(r"bytes=(\d*)-(\d*)",value or "")
    if not match or size<=0:raise ValueError("Invalid range")
    left,right=match.groups()
    if not left and not right:raise ValueError("Empty range")
    if left:
        start=int(left);end=int(right) if right else size-1
        if start>=size or end<start:raise ValueError("Unsatisfiable range")
        return start,min(end,size-1)
    suffix=int(right)
    if suffix<=0:raise ValueError("Empty suffix")
    return max(0,size-suffix),size-1


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version="HTTP/1.1"

    def send_head(self):
        self.bounds=None
        root=Path(self.directory).resolve();path=Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(root):self.send_error(403);return None
        if path.is_dir():path=(path/"index.html").resolve()
        if not path.is_relative_to(root):self.send_error(403);return None
        if not path.is_file():self.send_error(404);return None
        stream=path.open("rb");size=path.stat().st_size
        try:
            if self.headers.get("Range"):
                try:self.bounds=byte_range(self.headers["Range"],size)
                except ValueError:
                    stream.close();self.send_response(416);self.send_header("Content-Range",f"bytes */{size}")
                    self.send_header("Content-Length","0");self.end_headers();return None
            start,end=self.bounds or (0,size-1)
            stream.seek(start);self.send_response(206 if self.bounds else 200)
            self.send_header("Content-Type",self.guess_type(str(path)))
            self.send_header("Content-Length",str(max(0,end-start+1)))
            self.send_header("Accept-Ranges","bytes")
            self.send_header("X-Content-Type-Options","nosniff")
            if self.bounds:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.end_headers();return stream
        except Exception:stream.close();raise

    def copyfile(self,source,output):
        remaining=self.bounds[1]-self.bounds[0]+1 if self.bounds else None
        try:
            while remaining is None or remaining>0:
                data=source.read(65536 if remaining is None else min(65536,remaining))
                if not data:break
                output.write(data)
                if remaining is not None:remaining-=len(data)
        except (BrokenPipeError,ConnectionResetError):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[4]/"datasets/listening_evaluation/pack")
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    if not (args.root/"study.js").is_file():parser.error("Generate the participant pack first")
    with ThreadingHTTPServer(("127.0.0.1",args.port),partial(RangeHandler,directory=str(args.root))) as server:
        print(f"Listening page: http://127.0.0.1:{args.port}/ (Ctrl+C to stop)",flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=="__main__":main()
