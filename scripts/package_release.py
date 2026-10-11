"""Build a native Release archive with exact source revision and checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from importlib.metadata import version
from pvz_gym.env_file import repository_root
from pvz_gym.runtime.game_validation import read_pe

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=repository_root() / "bin/pvz_gym_native.dll")
    parser.add_argument("--output", type=Path, default=repository_root() / "dist")
    args = parser.parse_args()
    if read_pe(args.dll).machine != 0x8664:
        parser.error("Release DLL must be x64")
    root = repository_root()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        parser.error("Commit reviewed source changes before packaging a Release")
    ver = version("pvz-rl")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"pvz-rl-native-{ver}-windows-x64.zip"
    dll = args.dll.read_bytes()
    metadata = {"version": ver, "source_commit": revision, "machine": "x64",
                "runtime_interface_version": 1, "game_version": "1.2.0.1065",
                "dll_sha256": hashlib.sha256(dll).hexdigest()}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("pvz_gym_native.dll", dll)
        z.writestr("metadata.json", json.dumps(metadata, indent=2) + "\n")
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            z.write(root / name, name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(archive)

if __name__ == "__main__":
    main()
