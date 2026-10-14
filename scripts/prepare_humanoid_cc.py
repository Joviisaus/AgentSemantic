"""Prepare local humanoid STL miniatures as blind connected-component test cases.

These are web-sourced printable miniatures stored locally. Binary STL carries
NO group/object/material/joint/comment namespace, so it is inherently
semantically anonymous at the geometry level. We additionally strip the
semantic FILENAME by exporting to a generic blind id, so the agent never sees
"Human_Paladin" etc.

Output: testmodels/humanoid_cc/{blind_id}/{blind_id}_anonymized.ply
plus a manifest.json (internal only; not shown to the agent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import trimesh

# blind_id -> local STL path. Filenames are intentionally NOT exposed to agent.
SRC = {
    "humanoid_blind_001": "testmodels/models/Human_Paladin.stl",
    "humanoid_blind_002": "testmodels/models/Human_Sphinx.STL",
    "humanoid_blind_003": "testmodels/models/Odogaron.stl",
    "humanoid_blind_004": "testmodels/models/Link.stl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="testmodels/humanoid_cc")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for blind_id, src in SRC.items():
        p = Path(src)
        print(f"[prep] {blind_id} <- {p.name}", flush=True)
        m = trimesh.load(str(p), process=False)
        if not isinstance(m, trimesh.Trimesh):
            m = trimesh.util.concatenate(
                [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        # weld coincident vertices so each physical body becomes ONE component
        m.merge_vertices()
        d = out / blind_id
        d.mkdir(parents=True, exist_ok=True)
        anon = d / f"{blind_id}_anonymized.ply"
        m.export(str(anon))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest.append({
            "blind_id": blind_id,
            "original_filename": p.name,
            "source": "local cache (web-sourced printable miniature, binary STL, "
                      "no internal part names)",
            "sha256": sha,
            "n_faces": int(len(m.faces)),
            "n_vertices": int(len(m.vertices)),
            "anonymized_ply": str(anon),
        })
        print(f"   -> {anon.name} faces={len(m.faces):,} verts={len(m.vertices):,}",
              flush=True)

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
