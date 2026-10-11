"""Download matching GitHub Release assets, including authenticated private releases."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request


class ReleaseDownloadError(RuntimeError):
    pass


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https":
            raise ReleaseDownloadError("Release redirects must use HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and target.netloc != urllib.parse.urlsplit(req.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def _git_token() -> str | None:
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    try:
        result = subprocess.run(["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n", capture_output=True,
            text=True, env=environment, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return fields.get("password")


def _read_api(path: str, token: str | None, *, asset: bool = False) -> bytes:
    headers = {"Accept": "application/octet-stream" if asset else "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "pvz-rl-setup"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request("https://api.github.com" + path, headers=headers)
    with urllib.request.build_opener(SafeRedirectHandler()).open(request, timeout=60) as response:
        return response.read()


def download_native(repository: str, destination: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Expected a GitHub owner/repository")
    package_version = version("pvz-rl")
    tag = "v" + package_version
    release_path = f"/repos/{repository}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or None
    try:
        payload = _read_api(release_path, token)
    except urllib.error.HTTPError as error:
        if error.code not in (401, 403, 404) or token:
            raise ReleaseDownloadError(f"GitHub Release request failed (HTTP {error.code})") from None
        token = _git_token()
        if not token:
            raise ReleaseDownloadError("Release unavailable. Sign in to GitHub with Git Credential Manager, "
                                       "or use -NativeArchive / -BuildNative.") from None
        try:
            payload = _read_api(release_path, token)
        except urllib.error.HTTPError as authenticated_error:
            raise ReleaseDownloadError(f"Cannot access {repository} {tag} (HTTP {authenticated_error.code}). "
                                       "Check repository access and Release availability; "
                                       "or use -NativeArchive / -BuildNative.") from None
    release = json.loads(payload)
    filename = f"pvz-rl-native-{package_version}-windows-x64.zip"
    required = (filename, filename + ".sha256")
    assets = {item.get("name"): item.get("id") for item in release.get("assets", [])}
    if any(type(assets.get(name)) is not int or assets[name] <= 0 for name in required):
        raise ReleaseDownloadError(f"Release {tag} is missing the x64 archive or SHA256 sidecar")
    # Resolve IDs against a fixed GitHub API host; never attach credentials to an arbitrary URL.
    try:
        contents = {name: _read_api(f"/repos/{repository}/releases/assets/{assets[name]}",
                                    token, asset=True) for name in required}
    except urllib.error.HTTPError as error:
        raise ReleaseDownloadError(f"Release asset download failed (HTTP {error.code})") from None
    import hashlib
    checksum_fields = contents[filename + ".sha256"].decode("ascii").split()
    if not checksum_fields:
        raise ReleaseDownloadError("Downloaded SHA256 sidecar is empty")
    expected = checksum_fields[0]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected) or hashlib.sha256(contents[filename]).hexdigest() != expected.lower():
        raise ReleaseDownloadError("Downloaded native archive SHA256 mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    for name in required:
        temporary = destination / (name + ".incoming")
        try:
            temporary.write_bytes(contents[name])
            temporary.replace(destination / name)
        finally:
            temporary.unlink(missing_ok=True)
    return destination / filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(download_native(args.repository, args.destination))
    except (ReleaseDownloadError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
