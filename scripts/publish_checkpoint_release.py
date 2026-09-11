#!/usr/bin/env python3
"""Verify an exported checkpoint folder, publish it, then write verified links."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "checkpoint_manifest.json").read_text())
    for row in manifest["checkpoints"]:
        path = (folder / row["path"]).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f"Invalid checkpoint path: {row['path']}")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {row['path']}")
    if not manifest["checkpoints"]:
        raise ValueError("Checkpoint manifest is empty")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--links-output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = verify_local(args.folder)
    if args.verify_only:
        print(f"Verified {len(manifest['checkpoints'])} local checkpoints")
        return

    from huggingface_hub import HfApi

    api = HfApi()  # Standard HF_TOKEN or hf auth login; never a token in this script.
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=False)
    allowed = [row["path"] for row in manifest["checkpoints"]] + ["README.md", "checkpoint_manifest.json"]
    api.upload_folder(repo_id=args.repo_id, folder_path=args.folder, allow_patterns=allowed,
                      commit_message="Publish checksum-verified paired SSR inference checkpoints")
    info = api.model_info(args.repo_id, files_metadata=True)
    files = {item.rfilename: item for item in info.siblings}
    rows = []
    for row in manifest["checkpoints"]:
        remote = files[row["path"]]
        if remote.size != row["bytes"] or remote.lfs is None or remote.lfs.sha256 != row["sha256"]:
            raise RuntimeError(f"Remote checksum verification failed: {row['path']}")
        rows.append({**row, "url": f"https://huggingface.co/{args.repo_id}/resolve/{info.sha}/{row['path']}"})
    result = {"status": "published_verified", "repo_id": args.repo_id,
              "revision": info.sha, "checkpoints": rows, "missing_weights": manifest.get("missing_weights", [])}
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.index_output.write_text(json.dumps(result, indent=2) + "\n")
    args.links_output.parent.mkdir(parents=True, exist_ok=True)
    args.links_output.write_text(
        "# Verified checkpoint downloads\n\n"
        f"[Hugging Face model repository](https://huggingface.co/{args.repo_id}/tree/{info.sha})\n\n"
        f"Revision: `{info.sha}`. All {len(rows)} file sizes and SHA-256 digests verified.\n\n"
        "| Cohort | Seed | Arm | Checkpoint |\n| --- | --- | --- | --- |\n" +
        "".join(f"| {r['cohort']} | {r['seed']} | {r['arm']} | [safetensors]({r['url']}) |\n" for r in rows)
    )
    print(f"Published and verified {len(rows)} checkpoints at {args.repo_id}@{info.sha}")


if __name__ == "__main__":
    main()
