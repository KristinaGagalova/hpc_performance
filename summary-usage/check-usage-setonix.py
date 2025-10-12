#!/usr/bin/env python3
"""
slurm_quarter_usage.py — Summarise Slurm project (account) usage by quarter

Computes CPU core-hours (and GPU-hours if present) for a given Slurm account (project)
split by quarters (Q1–Q4) in a calendar year. Optionally exports a CSV and prints a
per-quarter table with job counts and breakdowns.

Also reports per-quarter:
- Peak allocated memory (GB)
- Number of jobs with allocated memory ≥ 200 GB
- Basic success/failure stats (Completed vs common failure terminal states)
- Estimated Service Units (SU) using Setonix node specs

Requires: sacct (Slurm accounting) with job-level access to the specified account.

Example:
  python slurm_quarter_usage.py --account aagi --year 2025 --csv usage_2025.csv
  python slurm_quarter_usage.py -A aagi --year 2025 --users --json usage.json
"""

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ----------------------------- Utility types -----------------------------
@dataclass
class JobRec:
    jobid: str
    user: str
    account: str
    state: str
    elapsed_s: int
    nnodes: int
    partition: str
    alloc_tres: Dict[str, float]  # {"cpu": total, "mem": GB_per_node, "gres/gpu": total}

# ----------------------------- Helpers -----------------------------
def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None

def run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {' '.join(cmd)}\n{e.output.decode()}", file=sys.stderr)
        sys.exit(2)
    return out.decode()

def parse_alloc_tres(tres_str: str) -> Dict[str, float]:
    # Example: "cpu=64,mem=128G,gres/gpu=4"
    tres = {}
    for kv in filter(None, tres_str.split(",")):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        try:
            if k == "mem":
                m = re.match(r"([0-9.]+)([KMGTP])?", v)
                if m:
                    num = float(m.group(1))
                    unit = m.group(2) or "G"
                    scale = {"K": 1/1048576, "M": 1/1024, "G": 1, "T": 1024, "P": 1024*1024}[unit]
                    tres[k] = num * scale  # in GB
                else:
                    tres[k] = float(v)
            else:
                tres[k] = float(v)
        except ValueError:
            continue
    return tres

def quarter_ranges(year: int) -> List[Tuple[str, dt.datetime, dt.datetime]]:
    return [
        ("Q1", dt.datetime(year, 1, 1), dt.datetime(year, 4, 1)),
        ("Q2", dt.datetime(year, 4, 1), dt.datetime(year, 7, 1)),
        ("Q3", dt.datetime(year, 7, 1), dt.datetime(year, 10, 1)),
        ("Q4", dt.datetime(year, 10, 1), dt.datetime(year + 1, 1, 1)),
    ]

def fetch_jobs(account: str, start: dt.datetime, end: dt.datetime,
               states: Optional[List[str]] = None) -> List[JobRec]:
    if not have_cmd("sacct"):
        print("ERROR: 'sacct' not found. Run on a Slurm login node with accounting enabled.", file=sys.stderr)
        sys.exit(1)

    state_filter = ",".join(states) if states else None
    # Add NNodes and Partition so we can estimate SU
    fmt = ["JobID", "User", "Account", "State", "ElapsedRaw", "NNodes", "Partition", "AllocTRES"]
    cmd = [
        "sacct", "-a", "-X", "--parsable2",
        "-A", account,
        "-S", start.strftime("%Y-%m-%dT%H:%M:%S"),
        "-E", end.strftime("%Y-%m-%dT%H:%M:%S"),
        "-o", ",".join(fmt),
    ]
    if state_filter:
        cmd += ["-s", state_filter]

    raw = run(cmd)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    col_idx = {name: i for i, name in enumerate(header)}
    jobs: List[JobRec] = []

    for ln in lines[1:]:
        parts = ln.split("|")
        jobid = parts[col_idx["JobID"]]
        if "." in jobid:  # skip steps
            continue
        state = parts[col_idx["State"]].split()[0]
        elapsed = int(parts[col_idx["ElapsedRaw"]] or 0)
        nnodes = int(parts[col_idx["NNodes"]] or 0)
        partition = (parts[col_idx["Partition"]] or "").strip()
        tres = parse_alloc_tres(parts[col_idx["AllocTRES"]] or "")
        jobs.append(JobRec(
            jobid=jobid,
            user=parts[col_idx["User"]],
            account=parts[col_idx["Account"]],
            state=state,
            elapsed_s=elapsed,
            nnodes=nnodes,
            partition=partition,
            alloc_tres=tres,
        ))
    return jobs

# ----------------------------- Setonix SU model -----------------------------
# Defaults based on Pawsey docs
SETONIX_CAPS = {
    "cpu":      {"cores_per_node": 128, "mem_gb": 256,  "gcds": 0},
    "cpu-hm":   {"cores_per_node": 128, "mem_gb": 1024, "gcds": 0},
    "gpu":      {"cores_per_node": 64,  "mem_gb": 256,  "gcds": 8},
    "gpu-hm":   {"cores_per_node": 64,  "mem_gb": 512,  "gcds": 8},
}

def classify_partition(name: str) -> str:
    n = (name or "").lower()
    if "gpu" in n:
        if "hm" in n or "high" in n:
            return "gpu-hm"
        return "gpu"
    # CPU side
    if "hm" in n or "high" in n:
        return "cpu-hm"
    return "cpu"

def safe_div(a: float, b: float) -> float:
    return (a / b) if (b and b > 0) else 0.0

def estimate_job_su(j: JobRec) -> float:
    """Estimate SUs for one job using Setonix capacities and dominant-resource rule."""
    cls = classify_partition(j.partition)
    caps = SETONIX_CAPS.get(cls, SETONIX_CAPS["cpu"])

    # Totals in AllocTRES are job-wide; derive per-node
    nn = max(j.nnodes, 1)
    cpus_total = j.alloc_tres.get("cpu", 0.0)
    gpus_total = 0.0
    for key in ("gres/gpu", "gpu"):
        if key in j.alloc_tres:
            gpus_total = max(gpus_total, j.alloc_tres[key])
    # mem in AllocTRES is already per-node GB (Slurm reports per-node mem)
    mem_per_node_gb = j.alloc_tres.get("mem", 0.0)

    cpus_per_node = cpus_total / nn if cpus_total else 0.0
    gcds_per_node = gpus_total / nn if gpus_total else 0.0

    f_core = min(1.0, safe_div(cpus_per_node, caps["cores_per_node"]))
    f_mem  = min(1.0, safe_div(mem_per_node_gb,  caps["mem_gb"]))
    f_gpu  = min(1.0, safe_div(gcds_per_node,    caps["gcds"])) if caps["gcds"] else 0.0

    util = max(f_core, f_mem, f_gpu)

    node_hours = nn * (j.elapsed_s / 3600.0)
    su = node_hours * caps["cores_per_node"] * util  # 1 SU == 1 core-hour
    return su

# ----------------------------- Aggregation -----------------------------
SUCCESS_STATES = {"COMPLETED", "CD"}
FAILURE_STATES = {"FAILED","F","CANCELLED","CA","TIMEOUT","TO",
                  "NODE_FAIL","NF","PREEMPTED","PR","DEADLINE","DL","BOOT_FAIL","BF"}

@dataclass
class QuarterStats:
    jobs: int = 0
    cpu_core_hours: float = 0.0
    gpu_hours: float = 0.0
    users: Counter = None
    states: Counter = None
    peak_mem_gb: float = 0.0
    jobs_mem_ge_200: int = 0
    success_jobs: int = 0
    failed_jobs: int = 0
    su_total: float = 0.0

    def __post_init__(self):
        if self.users is None: self.users = Counter()
        if self.states is None: self.states = Counter()

def accumulate(jobs: List[JobRec]) -> QuarterStats:
    stats = QuarterStats()
    for j in jobs:
        stats.jobs += 1
        stats.states[j.state] += 1
        stats.users[j.user] += 1

        cpus = j.alloc_tres.get("cpu", 0.0)
        stats.cpu_core_hours += (cpus * j.elapsed_s) / 3600.0

        gpus = 0.0
        for key in ("gres/gpu", "gpu"):
            if key in j.alloc_tres:
                gpus = max(gpus, j.alloc_tres[key])
        stats.gpu_hours += (gpus * j.elapsed_s) / 3600.0

        mem_gb = j.alloc_tres.get("mem", 0.0)
        stats.peak_mem_gb = max(stats.peak_mem_gb, mem_gb)
        if mem_gb >= 200: stats.jobs_mem_ge_200 += 1

        st = j.state.upper()
        if st in SUCCESS_STATES: stats.success_jobs += 1
        elif st in FAILURE_STATES: stats.failed_jobs += 1

        stats.su_total += estimate_job_su(j)

    return stats

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Summarise Slurm account usage by quarter")
    ap.add_argument("--account", "-A", required=True, help="Slurm account / project name")
    ap.add_argument("--year", type=int, default=dt.datetime.now().year, help="Calendar year (default: this year)")
    ap.add_argument("--states", nargs="*", default=["BF","CA","CD","DL","F","NF","PR","TO"],
                    help="Job terminal states to include (default: common final states). Examples: CD F TO CA")
    ap.add_argument("--csv", default=None, help="Optional CSV output filepath")
    ap.add_argument("--users", action="store_true", help="Show top users per quarter")
    ap.add_argument("--limit", type=int, default=5, help="Top-N users to show when --users is set (default 5)")
    ap.add_argument("--json", dest="json_out", default=None, help="Also write raw JSON to this file")
    # Optional overrides for node capacities if Pawsey updates them
    ap.add_argument("--cpu-mem", type=int, default=SETONIX_CAPS["cpu"]["mem_gb"], help="CPU node memory GB (default 256)")
    ap.add_argument("--cpu-hm-mem", type=int, default=SETONIX_CAPS["cpu-hm"]["mem_gb"], help="CPU high-mem node memory GB (default 1024)")
    ap.add_argument("--gpu-mem", type=int, default=SETONIX_CAPS["gpu"]["mem_gb"], help="GPU node memory GB (default 256)")
    ap.add_argument("--gpu-hm-mem", type=int, default=SETONIX_CAPS["gpu-hm"]["mem_gb"], help="GPU high-mem node memory GB (default 512)")

    args = ap.parse_args()
    # apply any mem overrides
    SETONIX_CAPS["cpu"]["mem_gb"]     = args.cpu_mem
    SETONIX_CAPS["cpu-hm"]["mem_gb"]  = args.cpu_hm_mem
    SETONIX_CAPS["gpu"]["mem_gb"]     = args.gpu_mem
    SETONIX_CAPS["gpu-hm"]["mem_gb"]  = args.gpu_hm_mem

    ranges = quarter_ranges(args.year)

    print(f"Account: {args.account} | Year: {args.year}")
    print("States included:", ", ".join(args.states))
    print()

    hdr = (
        f"{'Quarter':6}  {'Jobs':>6}  {'CPU core-hrs':>14}  {'GPU-hrs':>10}  "
        f"{'PeakMem(GB)':>11}  {'Jobs≥200GB':>11}  {'Succ':>6}  {'Fail':>6}  {'Succ%':>6}  {'SUs':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []
    json_blob = {}

    for qname, start, end in ranges:
        jobs = fetch_jobs(args.account, start, end, args.states)
        stats = accumulate(jobs)
        success_rate = (100.0 * stats.success_jobs / stats.jobs) if stats.jobs else 0.0

        print(
            f"{qname:6}  {stats.jobs:6d}  {stats.cpu_core_hours:14.2f}  {stats.gpu_hours:10.2f}  "
            f"{stats.peak_mem_gb:11.2f}  {stats.jobs_mem_ge_200:11d}  "
            f"{stats.success_jobs:6d}  {stats.failed_jobs:6d}  {success_rate:6.1f}  {stats.su_total:12.2f}"
        )

        if args.users and stats.jobs:
            topu = stats.users.most_common(args.limit)
            tops = ", ".join([f"{u}:{n}" for u, n in topu])
            print(f"           Top users: {tops}")
            top_states = ", ".join([f"{s}:{c}" for s, c in stats.states.most_common()])
            print(f"           States: {top_states}")

        csv_row = {
            "quarter": qname,
            "jobs": stats.jobs,
            "cpu_core_hours": round(stats.cpu_core_hours, 2),
            "gpu_hours": round(stats.gpu_hours, 2),
            "peak_mem_gb": round(stats.peak_mem_gb, 2),
            "jobs_mem_ge_200": stats.jobs_mem_ge_200,
            "success_jobs": stats.success_jobs,
            "failed_jobs": stats.failed_jobs,
            "success_rate": round(success_rate, 2),
            "sus": round(stats.su_total, 2),
        }
        csv_rows.append(csv_row)

        json_blob[qname] = {
            "jobs": stats.jobs,
            "cpu_core_hours": stats.cpu_core_hours,
            "gpu_hours": stats.gpu_hours,
            "peak_mem_gb": stats.peak_mem_gb,
            "jobs_mem_ge_200": stats.jobs_mem_ge_200,
            "success_jobs": stats.success_jobs,
            "failed_jobs": stats.failed_jobs,
            "success_rate": success_rate,
            "sus": stats.su_total,
            "states": stats.states,
            "users": stats.users,
        }

    if args.csv:
        fields = list(csv_rows[0].keys()) if csv_rows else [
            "quarter","jobs","cpu_core_hours","gpu_hours","peak_mem_gb",
            "jobs_mem_ge_200","success_jobs","failed_jobs","success_rate","sus"
        ]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"\nWrote CSV -> {args.csv}")

    if args.json_out:
        blob_serialisable = {
            q: {
                **{k: v for k, v in v.items() if k not in ("states","users")},
                "states": dict(v["states"]),
                "users": dict(v["users"]),
            }
            for q, v in json_blob.items()
        }
        with open(args.json_out, "w") as fh:
            json.dump(blob_serialisable, fh, indent=2)
        print(f"Wrote JSON -> {args.json_out}")

if __name__ == "__main__":
    main()
