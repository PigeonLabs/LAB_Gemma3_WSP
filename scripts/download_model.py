"""Download the user-authorized public mirror and verify upstream weight hashes."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/gemma-3-4b-it"


def main():
    upstream = json.loads((ROOT / "artifacts/google_model_metadata.json").read_text())
    mirror = json.loads((ROOT / "artifacts/unsloth_model_metadata.json").read_text())
    official = {f["rfilename"]: f for f in upstream["siblings"]}
    files = [f for f in mirror["siblings"] if f["rfilename"].endswith(
        (".json", ".model", ".safetensors", ".jinja")) or
        f["rfilename"] in {"README.md", "LICENSE", "LICENSE.txt", "NOTICE"}]
    DEST.mkdir(parents=True, exist_ok=True)

    def download(item):
        name = item["rfilename"]
        expected = item.get("lfs", {}).get("sha256")
        original = official.get(name, {}).get("lfs", {}).get("sha256")
        if name.endswith(".safetensors") and (not expected or expected != original):
            raise ValueError(f"Upstream weight hash mismatch: {name}")
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{mirror['id']}/resolve/{mirror['sha']}/{name}"
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".part")
            for attempt in range(3):
                try:
                    start = temp.stat().st_size if temp.exists() else 0
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        append = start > 0 and response.status == 206
                        with temp.open("ab" if append else "wb") as out:
                            while data := response.read(8 * 1024 * 1024):
                                out.write(data)
                    if item.get("size") is not None and temp.stat().st_size != item["size"]:
                        raise ValueError(f"Incomplete file: {name}")
                    temp.rename(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        actual = digest.hexdigest()
        if expected and actual != expected:
            raise ValueError(f"Downloaded hash mismatch: {name}")
        result = {"file": name, "size": path.stat().st_size, "sha256": actual,
                  "upstream_weight_match": bool(original and original == actual)}
        print(json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download, files))
    manifest = {"source": mirror["id"], "source_revision": mirror["sha"],
                "upstream": upstream["id"], "upstream_revision": upstream["sha"],
                "files": results, "status": "verified"}
    (ROOT / "artifacts/model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DOWNLOAD_AND_HASH_VERIFICATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
