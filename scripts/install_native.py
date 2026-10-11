"""Install a version-matched native archive without extracting arbitrary paths."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import os
import zipfile
from pvz_gym.env_file import repository_root
from pvz_gym.runtime.game_validation import read_pe

def install_archive(archive: Path, destination: Path | None = None):
    expected = Path(str(archive) + ".sha256").read_text(encoding="ascii").split()[0]
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Native archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != {"pvz_gym_native.dll", "metadata.json", "LICENSE", "THIRD_PARTY_NOTICES.md"} or len(z.namelist()) != 4:
            raise ValueError("Unexpected native archive contents")
        metadata = json.loads(z.read("metadata.json"))
        dll = z.read("pvz_gym_native.dll")
    if metadata.get("version") != version("pvz-rl") or metadata.get("machine") != "x64" or metadata.get("runtime_interface_version") != 1:
        raise ValueError("Native archive version/architecture/interface does not match this source")
    if hashlib.sha256(dll).hexdigest() != metadata.get("dll_sha256"):
        raise ValueError("DLL checksum mismatch")
    target = destination or repository_root() / "bin"
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "pvz_gym_native.dll.incoming"
    temporary.write_bytes(dll)
    try:
        if read_pe(temporary).machine != 0x8664:
            raise ValueError("Archive DLL is not an x64 PE image")
        os.replace(temporary, target / "pvz_gym_native.dll")
    finally:
        temporary.unlink(missing_ok=True)
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(f"Native DLL installed: {install_archive(args.archive)}")

if __name__ == "__main__":
    main()
