#!/usr/bin/env python
"""Find the minimal Python / torch versions acids_transforms works with.

For every (python, torch) pair it creates an isolated venv with uv, installs torch + the matching torchaudio,
installs acids_transforms (editable, no deps), then runs:
    - an import check (fails fast on missing APIs, e.g. torch.library.custom_op)
    - the test suite (or a subset with --tests / -k)

Usage:
    python scripts/check_versions.py                                  # default matrix, quick test subset
    python scripts/check_versions.py --full                            # full test suite (~2 min per combo)
    python scripts/check_versions.py --python 3.9 3.10 --torch 2.3 2.4 2.5
    python scripts/check_versions.py --stop-at-first                   # per python, stop at the first passing torch

Requires uv (https://docs.astral.sh/uv/). Each venv is ~600MB, so venvs are created in a temp dir and deleted after
each combo (use --keep-venvs to cache them in --venvdir). Logs are written to build/version_matrix/.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PYTHONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]
DEFAULT_TORCHES = ["1.10", "1.11", "1.12", "1.13", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8",
                   "2.9", "2.10", "2.11"]
QUICK_TESTS = ["test/test_stft.py"]


def torchaudio_for(torch_version: str) -> str:
    """torch 1.X <-> torchaudio 0.X, torch 2.X <-> torchaudio 2.X"""
    major, minor = torch_version.split(".")[:2]
    return f"0.{minor}" if major == "1" else f"2.{minor}"


def vtuple(v: str):
    return tuple(int(x) for x in v.split(".")[:2])


def run(cmd, log, timeout=None, env=None):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    try:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=timeout, env=env)
        return p.returncode
    except subprocess.TimeoutExpired:
        log.write("TIMEOUT\n")
        return -1


def check(py: str, tv: str, args) -> str:
    try:
        return _check(py, tv, args)
    finally:
        if not args.keep_venvs:
            shutil.rmtree(args.venvdir / f"py{py}-torch{tv}", ignore_errors=True)


def _check(py: str, tv: str, args) -> str:
    name = f"py{py}-torch{tv}"
    venv = args.venvdir / name
    log_path = args.workdir / f"{name}.log"
    pybin = venv / "bin" / "python"
    with open(log_path, "w") as log:
        if not pybin.exists():
            if run(["uv", "venv", "-q", "--python", py, str(venv)], log) != 0:
                shutil.rmtree(venv, ignore_errors=True)
                return "NO_PYTHON"
            tav = torchaudio_for(tv)
            pkgs = [f"torch=={tv}.*", f"torchaudio=={tav}.*", "numpy", "pytest", "soundfile"]
            if vtuple(tv) < (2, 3):
                pkgs.append("numpy<2")  # torch < 2.3 was built against numpy 1.x
            if vtuple(tv) < (1, 11):
                pkgs.append("setuptools<81")  # torch 1.10 imports pkg_resources
            if vtuple(tv) >= (2, 9):
                pkgs.append("torchcodec")  # torchaudio >= 2.9 delegates load/save to torchcodec
            if run(["uv", "pip", "install", "-q", "--python", str(pybin), *pkgs], log) != 0:
                shutil.rmtree(venv, ignore_errors=True)
                return "NO_WHEEL"
            if run(["uv", "pip", "install", "-q", "--python", str(pybin), "--no-deps", "-e", str(ROOT)], log) != 0:
                return "INSTALL_FAIL"
        run([str(pybin), "-c", "import torch, torchaudio, numpy; "
             "print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'numpy', numpy.__version__)"],
            log)
        if run([str(pybin), "-c", "import acids_transforms"], log) != 0:
            return "IMPORT_FAIL"
        if args.no_tests:
            return "OK"
        tests = [] if args.full else (args.tests or QUICK_TESTS)
        cmd = [str(pybin), "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests]
        if args.k:
            cmd += ["-k", args.k]
        rc = run(cmd, log, timeout=args.timeout)
        return "OK" if rc in (0, 5) else ("TIMEOUT" if rc == -1 else "TEST_FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", nargs="+", default=DEFAULT_PYTHONS)
    parser.add_argument("--torch", nargs="+", default=DEFAULT_TORCHES, help="torch minor versions (e.g. 2.4)")
    parser.add_argument("--full", action="store_true", help="run the whole test suite")
    parser.add_argument("--tests", nargs="+", help=f"test paths to run (default: {QUICK_TESTS})")
    parser.add_argument("-k", help="pytest -k expression")
    parser.add_argument("--no-tests", action="store_true", help="only check install + import")
    parser.add_argument("--stop-at-first", action="store_true",
                        help="for each python, stop at the first torch version that passes")
    parser.add_argument("--timeout", type=int, default=1800, help="pytest timeout per combo (s)")
    parser.add_argument("--workdir", type=Path, default=ROOT / "build" / "version_matrix", help="logs directory")
    parser.add_argument("--venvdir", type=Path, default=Path(tempfile.gettempdir()) / "acids_transforms_matrix",
                        help="where venvs are created (keep it out of synced folders)")
    parser.add_argument("--keep-venvs", action="store_true", help="keep venvs between runs (~600MB each)")
    args = parser.parse_args()

    if shutil.which("uv") is None:
        sys.exit("uv is required: https://docs.astral.sh/uv/getting-started/installation/")
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.venvdir.mkdir(parents=True, exist_ok=True)
    torches = sorted(args.torch, key=vtuple)

    results = {}
    for py in args.python:
        for tv in torches:
            t0 = time.time()
            status = check(py, tv, args)
            results[py, tv] = status
            print(f"python {py:<5} torch {tv:<5} -> {status:<12} ({time.time() - t0:.0f}s)", flush=True)
            if status == "NO_PYTHON" or (args.stop_at_first and status == "OK"):
                break

    # summary table
    cell = {"OK": "ok", "NO_WHEEL": "-", "NO_PYTHON": "?", "IMPORT_FAIL": "IMPORT", "TEST_FAIL": "FAIL",
            "INSTALL_FAIL": "INST", "TIMEOUT": "T/O"}
    print("\n" + "torch".ljust(8) + "".join(py.ljust(8) for py in args.python))
    for tv in torches:
        print(tv.ljust(8) + "".join(cell.get(results.get((py, tv)), " ").ljust(8) for py in args.python))
    print("\nlegend: ok=pass, -=no wheel for this platform, ?=python unavailable, IMPORT=import fails, "
          f"FAIL=tests fail. Logs: {args.workdir}/<combo>.log")

    ok = [k for k, v in results.items() if v == "OK"]
    if ok:
        print(f"minimal python: {min((k[0] for k in ok), key=vtuple)}   "
              f"minimal torch: {min((k[1] for k in ok), key=vtuple)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
