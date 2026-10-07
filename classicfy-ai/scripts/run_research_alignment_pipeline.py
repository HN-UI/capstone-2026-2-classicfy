"""Reproducible benchmark -> choose -> ATEPP alignment/features -> validation/report."""
import argparse
from pathlib import Path
import subprocess
import os
import sys


def main():
    p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--torch-threads',type=int,default=4)
    p.add_argument('--glue-backend',choices=['torch','openvino_cpu','openvino_gpu'],default='torch')
    p.add_argument('--skip-benchmark',action='store_true')
    p.add_argument('--skip-reference-refresh',action='store_true',help='Reuse already computed reference errors and summarize saved results')
    p.add_argument('--paired-only',action='store_true'); args=p.parse_args()
    scripts=Path(__file__).resolve().parent
    os.environ['CLASSICFY_TORCH_THREADS']=str(args.torch_threads)
    os.environ['CLASSICFY_GLUE_BACKEND']=args.glue_backend
    root=scripts.parents[1]
    def run(script,*arguments):
        subprocess.run([sys.executable,str(scripts/script),*arguments],cwd=root,check=True)
    if not args.skip_benchmark:
        run('benchmark_research_alignment.py','--workers',str(args.workers),'--torch-threads',str(args.torch_threads),
            '--glue-backend',args.glue_backend,'--retry-errors')
    run('benchmark_research_alignment.py','--workers',str(args.workers),'--torch-threads',str(args.torch_threads),
        '--glue-backend',args.glue_backend,'--summarize-only' if args.skip_reference_refresh else '--refresh-reference',
        *(['--paired-only'] if args.paired_only else []))
    run('write_research_alignment_reports.py','--benchmark-only')
    run('extend_atepp.py','--aligner','selected','--output','classicfy-ai/analysis/atepp_research',
        '--workers',str(args.workers),'--resume')
    run('validate_atepp_recommendation.py','--output','classicfy-ai/analysis/atepp_research')
    run('write_research_alignment_reports.py')


if __name__=='__main__': main()
