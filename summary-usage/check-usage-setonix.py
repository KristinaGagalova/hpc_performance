#!/usr/bin/env python3
"""
check-setonix-usage.py — Summarise Slurm project (account) usage by quarter

This version is designed to track Pawsey project usage more closely.

Key behaviour
-------------
1. Prefers Slurm's authoritative billing TRES when present.
2. Applies a conversion factor so reported usage aligns with pawseyAccountBalance.
3. Skips job steps such as:
      12345.batch
      12345.extern
      12345.0
4. Skips array parent jobs when array task records exist:
      skip 39642022
      keep 39642022_105, 39642022_106, ...
5. Supports multiple accounts:
      -A pawsey1168 pawsey1168-gpu
6. Optionally computes percent used from a user-provided quarterly allocation:
      --quarterly-usage 400000

Example
-------
./check-setonix-usage.py -A pawsey1168 --year 2026 --quarterly-usage 400000
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
from typing import Dict, List, Optional, Tuple


# Conversion factor to align billing-based usage with pawseyAccountBalance output.
# Based on observed behaviour for this account.
BILLING_TO_PABS_FACTOR = 0.5


class JobRec:
    def __init__(self, jobid, user, account, state, elapsed_s, nnodes, partition, alloc_tres):
        self.jobid = jobid
        self.user = user
        self.account = account
        self.state = state
        self.elapsed_s = elapsed_s
        self.nnodes = nnodes
        self.partition = partition
        self.alloc_tres = alloc_tres


class QuarterStats:
    def __init__(self):
        self.jobs = 0
        self.cpu_core_hours = 0.0
        self.gpu_hours = 0.0
        self.users = Counter()
        self.accounts = Counter()
        self.states = Counter()
        self.peak_mem_gb = 0.0
        self.jobs_mem_ge_200 = 0
        self.success_jobs = 0
        self.failed_jobs = 0
        self.su_total = 0.0


def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        try:
            msg = e.output.decode()
        except Exception:
            msg = str(e.output)
        print("ERROR running {}\n{}".format(" ".join(cmd), msg), file=sys.stderr)
        sys.exit(2)
    return out.decode()


def parse_mem_to_gb(v: str) -> Optional[float]:
    """
    Parse memory strings such as:
      8G, 1024M, 1T, 4000Mn, 200Gc
    into GB.
    """
    if not v:
        return None

    m = re.match(r"^\s*([0-9.]+)([KMGTP]?)([cn]?)\s*$", v, re.IGNORECASE)
    if not m:
        return None

    num = float(m.group(1))
    unit = (m.group(2) or "G").upper()

    scale = {
        "K": 1.0 / 1048576.0,
        "M": 1.0 / 1024.0,
        "G": 1.0,
        "T": 1024.0,
        "P": 1024.0 * 1024.0,
    }
    return num * scale[unit]


def parse_alloc_tres(tres_str: str) -> Dict[str, float]:
    """
    Example:
      billing=10,cpu=10,energy=20215,mem=8G,node=1
      cpu=64,mem=128G,gres/gpu=4
    """
    tres = {}
    for kv in filter(None, tres_str.split(",")):
        if "=" not in kv:
            continue

        k, v = kv.split("=", 1)
        k = k.strip().lower()
        v = v.strip()

        try:
            if k == "mem":
                mem_gb = parse_mem_to_gb(v)
                if mem_gb is not None:
                    tres[k] = mem_gb
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


def fetch_jobs(accounts: List[str], start: dt.datetime, end: dt.datetime,
               states: Optional[List[str]] = None) -> List[JobRec]:
    if not have_cmd("sacct"):
        print("ERROR: 'sacct' not found. Run on a Slurm login node with accounting enabled.", file=sys.stderr)
        sys.exit(1)

    state_filter = ",".join(states) if states else None
    fmt = ["JobID", "User", "Account", "State", "ElapsedRaw", "NNodes", "Partition", "AllocTRES"]

    cmd = [
        "sacct", "-a", "-X", "--parsable2",
        "-A", ",".join(accounts),
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

    raw_rows = []
    array_parents = set()

    for ln in lines[1:]:
        parts = ln.split("|")
        if len(parts) < len(header):
            continue

        jobid = parts[col_idx["JobID"]]

        # Skip job steps: 12345.batch, 12345.extern, 12345.0
        if "." in jobid:
            continue

        raw_rows.append(parts)

        # Detect array children, e.g. 39642022_105 -> parent 39642022
        if "_" in jobid:
            parent = jobid.split("_", 1)[0]
            array_parents.add(parent)

    jobs = []

    for parts in raw_rows:
        jobid = parts[col_idx["JobID"]]

        # Skip array parent if child tasks exist
        if jobid in array_parents and "_" not in jobid:
            continue

        state = (parts[col_idx["State"]] or "").split()[0]

        try:
            elapsed = int(parts[col_idx["ElapsedRaw"]] or 0)
        except ValueError:
            elapsed = 0

        try:
            nnodes = int(parts[col_idx["NNodes"]] or 0)
        except ValueError:
            nnodes = 0

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


SETONIX_CAPS = {
    "cpu": {"cores_per_node": 128, "gpus_per_node": 0},
    "cpu-hm": {"cores_per_node": 128, "gpus_per_node": 0},
    "gpu": {"cores_per_node": 64, "gpus_per_node": 8},
    "gpu-hm": {"cores_per_node": 64, "gpus_per_node": 8},
}


def classify_partition(name: str) -> str:
    n = (name or "").lower()

    if "gpu" in n:
        if "high" in n or "hm" in n:
            return "gpu-hm"
        return "gpu"

    if "high" in n or "hm" in n:
        return "cpu-hm"

    return "cpu"


def safe_div(a: float, b: float) -> float:
    return (a / b) if (b and b > 0) else 0.0


CHARGE_RATES = {
    "cpu": 128.0,
    "cpu-hm": 128.0,
    "gpu": 512.0,
    "gpu-hm": 512.0,
}

ACCOUNT_MEM_CAP_GB = {
    "cpu": 235.0,
    "cpu-hm": 1011.0,
    "gpu": 235.0,
    "gpu-hm": 460.0,
}


def estimate_job_su_fallback(j: JobRec) -> float:
    """
    Fallback estimate if billing is absent.
    """
    cls = classify_partition(j.partition)
    caps = SETONIX_CAPS.get(cls, SETONIX_CAPS["cpu"])
    rate = CHARGE_RATES.get(cls, 128.0)
    mem_cap = ACCOUNT_MEM_CAP_GB.get(cls, 235.0)

    nn = max(j.nnodes, 1)
    hours = j.elapsed_s / 3600.0

    cpus_total = j.alloc_tres.get("cpu", 0.0)

    gpus_total = 0.0
    for key in ("gres/gpu", "gpu"):
        if key in j.alloc_tres:
            gpus_total = max(gpus_total, j.alloc_tres[key])

    mem_alloc_gb = j.alloc_tres.get("mem", 0.0)

    cpus_per_node = (cpus_total / nn) if cpus_total else 0.0
    gpus_per_node = (gpus_total / nn) if gpus_total else 0.0
    mem_per_node_gb = mem_alloc_gb

    core_prop = safe_div(cpus_per_node, caps["cores_per_node"])
    mem_prop = safe_div(mem_per_node_gb, mem_cap)
    gpu_prop = safe_div(gpus_per_node, caps["gpus_per_node"]) if caps["gpus_per_node"] else 0.0

    max_prop = min(1.0, max(core_prop, mem_prop, gpu_prop))
    return rate * max_prop * nn * hours * BILLING_TO_PABS_FACTOR


def estimate_job_su(j: JobRec) -> float:
    """
    Prefer billing TRES if present, then apply conversion factor.
    """
    hours = j.elapsed_s / 3600.0
    billing = j.alloc_tres.get("billing", 0.0)

    if billing and billing > 0:
        return billing * hours * BILLING_TO_PABS_FACTOR

    return estimate_job_su_fallback(j)


SUCCESS_STATES = {"COMPLETED", "CD"}
FAILURE_STATES = {
    "FAILED", "F",
    "CANCELLED", "CA",
    "TIMEOUT", "TO",
    "NODE_FAIL", "NF",
    "PREEMPTED", "PR",
    "DEADLINE", "DL",
    "BOOT_FAIL", "BF",
}


def accumulate(jobs: List[JobRec]) -> QuarterStats:
    stats = QuarterStats()

    for j in jobs:
        stats.jobs += 1
        stats.states[j.state] += 1
        stats.users[j.user] += 1
        stats.accounts[j.account] += 1

        cpus = j.alloc_tres.get("cpu", 0.0)
        stats.cpu_core_hours += (cpus * j.elapsed_s) / 3600.0

        gpus = 0.0
        for key in ("gres/gpu", "gpu"):
            if key in j.alloc_tres:
                gpus = max(gpus, j.alloc_tres[key])
        stats.gpu_hours += (gpus * j.elapsed_s) / 3600.0

        mem_gb = j.alloc_tres.get("mem", 0.0)
        if mem_gb > stats.peak_mem_gb:
            stats.peak_mem_gb = mem_gb
        if mem_gb >= 200:
            stats.jobs_mem_ge_200 += 1

        st = (j.state or "").upper()
        if st in SUCCESS_STATES:
            stats.success_jobs += 1
        elif st in FAILURE_STATES:
            stats.failed_jobs += 1

        stats.su_total += estimate_job_su(j)

    return stats


def main():
    ap = argparse.ArgumentParser(description="Summarise Slurm account usage by quarter")
    ap.add_argument(
        "--account", "-A",
        nargs="+",
        required=True,
        help="One or more Slurm accounts / project names. Example: -A pawsey1168 pawsey1168-gpu"
    )
    ap.add_argument("--year", type=int, default=dt.datetime.now().year, help="Calendar year")
    ap.add_argument(
        "--quarterly-usage",
        type=float,
        default=None,
        help="Quarterly allocation to use for calculating percent used"
    )
    ap.add_argument(
        "--states",
        nargs="*",
        default=["BF", "CA", "CD", "DL", "F", "NF", "PR", "TO", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"],
        help="Job terminal states to include"
    )
    ap.add_argument("--csv", default=None, help="Optional CSV output filepath")
    ap.add_argument("--json", dest="json_out", default=None, help="Optional JSON output filepath")
    ap.add_argument("--users", action="store_true", help="Show top users per quarter")
    ap.add_argument("--accounts", action="store_true", help="Show top accounts per quarter")
    ap.add_argument("--limit", type=int, default=5, help="Top N for --users / --accounts")

    ap.add_argument("--cpu-cores", type=int, default=SETONIX_CAPS["cpu"]["cores_per_node"])
    ap.add_argument("--cpu-hm-cores", type=int, default=SETONIX_CAPS["cpu-hm"]["cores_per_node"])
    ap.add_argument("--gpu-cores", type=int, default=SETONIX_CAPS["gpu"]["cores_per_node"])
    ap.add_argument("--gpu-hm-cores", type=int, default=SETONIX_CAPS["gpu-hm"]["cores_per_node"])
    ap.add_argument("--gpu-per-node", type=int, default=SETONIX_CAPS["gpu"]["gpus_per_node"])
    ap.add_argument("--gpu-hm-per-node", type=int, default=SETONIX_CAPS["gpu-hm"]["gpus_per_node"])

    ap.add_argument("--rate-cpu", type=float, default=CHARGE_RATES["cpu"])
    ap.add_argument("--rate-cpu-hm", type=float, default=CHARGE_RATES["cpu-hm"])
    ap.add_argument("--rate-gpu", type=float, default=CHARGE_RATES["gpu"])
    ap.add_argument("--rate-gpu-hm", type=float, default=CHARGE_RATES["gpu-hm"])

    ap.add_argument("--acc-mem-cpu", type=float, default=ACCOUNT_MEM_CAP_GB["cpu"])
    ap.add_argument("--acc-mem-cpu-hm", type=float, default=ACCOUNT_MEM_CAP_GB["cpu-hm"])
    ap.add_argument("--acc-mem-gpu", type=float, default=ACCOUNT_MEM_CAP_GB["gpu"])
    ap.add_argument("--acc-mem-gpu-hm", type=float, default=ACCOUNT_MEM_CAP_GB["gpu-hm"])

    args = ap.parse_args()

    SETONIX_CAPS["cpu"]["cores_per_node"] = args.cpu_cores
    SETONIX_CAPS["cpu-hm"]["cores_per_node"] = args.cpu_hm_cores
    SETONIX_CAPS["gpu"]["cores_per_node"] = args.gpu_cores
    SETONIX_CAPS["gpu-hm"]["cores_per_node"] = args.gpu_hm_cores
    SETONIX_CAPS["gpu"]["gpus_per_node"] = args.gpu_per_node
    SETONIX_CAPS["gpu-hm"]["gpus_per_node"] = args.gpu_hm_per_node

    CHARGE_RATES["cpu"] = args.rate_cpu
    CHARGE_RATES["cpu-hm"] = args.rate_cpu_hm
    CHARGE_RATES["gpu"] = args.rate_gpu
    CHARGE_RATES["gpu-hm"] = args.rate_gpu_hm

    ACCOUNT_MEM_CAP_GB["cpu"] = args.acc_mem_cpu
    ACCOUNT_MEM_CAP_GB["cpu-hm"] = args.acc_mem_cpu_hm
    ACCOUNT_MEM_CAP_GB["gpu"] = args.acc_mem_gpu
    ACCOUNT_MEM_CAP_GB["gpu-hm"] = args.acc_mem_gpu_hm

    print("Accounts: {} | Year: {}".format(", ".join(args.account), args.year))
    print("States included: {}".format(", ".join(args.states)))
    if args.quarterly_usage is not None:
        print("Quarterly usage reference: {:.2f}".format(args.quarterly_usage))
    print()

    if args.quarterly_usage is not None:
        hdr = (
            "{:6}  {:>6}  {:>14}  {:>10}  {:>11}  {:>11}  {:>6}  {:>6}  {:>6}  {:>12}  {:>8}".format(
                "Quarter", "Jobs", "CPU core-hrs", "GPU-hrs",
                "PeakMem(GB)", "Jobs≥200GB", "Succ", "Fail", "Succ%", "SUs", "%Used"
            )
        )
    else:
        hdr = (
            "{:6}  {:>6}  {:>14}  {:>10}  {:>11}  {:>11}  {:>6}  {:>6}  {:>6}  {:>12}".format(
                "Quarter", "Jobs", "CPU core-hrs", "GPU-hrs",
                "PeakMem(GB)", "Jobs≥200GB", "Succ", "Fail", "Succ%", "SUs"
            )
        )

    print(hdr)
    print("-" * len(hdr))

    csv_rows = []
    json_blob = {}

    for qname, start, end in quarter_ranges(args.year):
        jobs = fetch_jobs(args.account, start, end, args.states)
        stats = accumulate(jobs)
        success_rate = (100.0 * stats.success_jobs / stats.jobs) if stats.jobs else 0.0

        percent_used = None
        if args.quarterly_usage is not None and args.quarterly_usage > 0:
            percent_used = 100.0 * stats.su_total / args.quarterly_usage

        if percent_used is not None:
            print(
                "{:6}  {:6d}  {:14.2f}  {:10.2f}  {:11.2f}  {:11d}  {:6d}  {:6d}  {:6.1f}  {:12.2f}  {:8.1f}".format(
                    qname,
                    stats.jobs,
                    stats.cpu_core_hours,
                    stats.gpu_hours,
                    stats.peak_mem_gb,
                    stats.jobs_mem_ge_200,
                    stats.success_jobs,
                    stats.failed_jobs,
                    success_rate,
                    stats.su_total,
                    percent_used,
                )
            )
        else:
            print(
                "{:6}  {:6d}  {:14.2f}  {:10.2f}  {:11.2f}  {:11d}  {:6d}  {:6d}  {:6.1f}  {:12.2f}".format(
                    qname,
                    stats.jobs,
                    stats.cpu_core_hours,
                    stats.gpu_hours,
                    stats.peak_mem_gb,
                    stats.jobs_mem_ge_200,
                    stats.success_jobs,
                    stats.failed_jobs,
                    success_rate,
                    stats.su_total,
                )
            )

        if args.users and stats.jobs:
            topu = ", ".join(["{}:{}".format(u, n) for u, n in stats.users.most_common(args.limit)])
            print("           Top users: {}".format(topu))

        if args.accounts and stats.jobs:
            topa = ", ".join(["{}:{}".format(a, n) for a, n in stats.accounts.most_common(args.limit)])
            print("           Top accounts: {}".format(topa))

        if (args.users or args.accounts) and stats.jobs:
            tops = ", ".join(["{}:{}".format(s, c) for s, c in stats.states.most_common()])
            print("           States: {}".format(tops))

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
            "percent_used": round(percent_used, 2) if percent_used is not None else None,
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
            "percent_used": percent_used,
            "states": dict(stats.states),
            "users": dict(stats.users),
            "accounts": dict(stats.accounts),
        }

    if args.csv:
        fields = list(csv_rows[0].keys()) if csv_rows else [
            "quarter", "jobs", "cpu_core_hours", "gpu_hours", "peak_mem_gb",
            "jobs_mem_ge_200", "success_jobs", "failed_jobs", "success_rate", "sus", "percent_used"
        ]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print("\nWrote CSV -> {}".format(args.csv))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(json_blob, fh, indent=2)
        print("Wrote JSON -> {}".format(args.json_out))


if __name__ == "__main__":
    main()
