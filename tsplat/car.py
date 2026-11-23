"""Fetch a car model from Sketchfab for local use in the viewer (`otrv car fetch <uid>`).

The repo ships no car models: the default ego car is drawn in code. Users can download a model
they are licensed to use with their own Sketchfab API token (https://sketchfab.com/settings/password,
"API Token"). The model and its credits stay on the user's machine (viewer/models/ is git-ignored);
the viewer reads viewer/models/car.json and shows the attribution CC licences require.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.sketchfab.com/v3/models"
# Licences that permit use in a local app/video. NC = non-commercial only (we warn); ND/unknown refused.
OK_LICENCES = {"by", "cc0", "by-sa"}
NC_LICENCES = {"by-nc", "by-nc-sa"}


def _get(url: str, token: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "open-tesla-reality-view"})
    if token:
        req.add_header("Authorization", f"Token {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def metadata(uid: str) -> dict:
    d = json.loads(_get(f"{API}/{uid}"))
    lic = d.get("license") or {}
    return {
        "uid": uid,
        "name": d.get("name"),
        "author": (d.get("user") or {}).get("username"),
        "author_url": (d.get("user") or {}).get("profileUrl"),
        "url": d.get("viewerUrl"),
        "license": lic.get("label"),
        "license_slug": lic.get("slug"),
        "license_url": lic.get("url"),
        "downloadable": bool(d.get("isDownloadable")),
        "faces": d.get("faceCount"),
    }


def check_licence(meta: dict, allow_nc: bool) -> str | None:
    """Return an error message if the model can't be used, else None."""
    slug = (meta.get("license_slug") or "").lower()
    if not meta.get("downloadable"):
        return "this model isn't downloadable on Sketchfab"
    if slug in OK_LICENCES:
        return None
    if slug in NC_LICENCES:
        return None if allow_nc else f"licence is {meta['license']} (non-commercial); pass --allow-nc to use it anyway"
    return f"licence {meta.get('license')!r} isn't one we can use automatically"


def credit_line(meta: dict) -> str:
    return f'"{meta["name"]}" by {meta["author"]} ({meta["url"]}), {meta["license"]}'


def install_model(blob: bytes, dest_dir: Path) -> Path:
    """Install a Sketchfab download: a raw .glb, or a zip holding a .glb or a .gltf + bin + textures.
    Returns the model path relative to dest_dir's parent use (car.glb or car/<file>.gltf)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if blob[:4] == b"glTF":
        out = dest_dir / "car.glb"
        out.write_bytes(blob)
        return out
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
        glbs = [n for n in names if n.lower().endswith(".glb")]
        if glbs:
            out = dest_dir / "car.glb"
            out.write_bytes(z.read(max(glbs, key=lambda n: z.getinfo(n).file_size)))
            return out
        gltfs = [n for n in names if n.lower().endswith(".gltf")]
        if not gltfs:
            raise SystemExit("download had no .glb/.gltf file; try a different model")
        folder = dest_dir / "car"
        root = Path(gltfs[0]).parent
        for n in names:
            rel = Path(n).relative_to(root) if root != Path(".") and Path(n).is_relative_to(root) else Path(n)
            target = (folder / rel).resolve()
            if not str(target).startswith(str(folder.resolve())):  # zip-slip guard
                raise SystemExit(f"unsafe path in archive: {n}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(n))
        return folder / Path(gltfs[0]).name


def fetch(uid: str, dest_dir: Path, token: str | None, allow_nc: bool = False, paint: str | None = "#f2f3f5") -> Path:
    meta = metadata(uid)
    err = check_licence(meta, allow_nc)
    if err:
        raise SystemExit(f"can't use {uid}: {err}")
    token = token or os.environ.get("SKETCHFAB_TOKEN")
    if not token:
        raise SystemExit(
            "Sketchfab needs your API token to download: copy it from https://sketchfab.com/settings/password "
            "and set SKETCHFAB_TOKEN (it is only sent to api.sketchfab.com)"
        )
    links = json.loads(_get(f"{API}/{uid}/download", token))
    fmt = links.get("glb") or links.get("gltf")
    if not fmt:
        raise SystemExit("Sketchfab returned no glb/gltf download for this model")
    print(f"downloading {meta['name']} by {meta['author']} ({meta['license']}, {meta['faces']} faces)…")
    glb = install_model(_get(fmt["url"]), dest_dir)
    cfg = {
        "file": str(glb.relative_to(dest_dir)),
        "lengthM": 4.75,
        "paint": paint,
        "credit": credit_line(meta),
        "source": meta,
    }
    (dest_dir / "car.json").write_text(json.dumps(cfg, indent=1))
    if (meta.get("license_slug") or "") in NC_LICENCES:
        print("note: non-commercial licence: don't use renders with this car commercially")
    print(f"installed {glb} (+ car.json with credits); the viewer will use it on next load")
    return glb


def models_dir() -> Path:
    """Where user car models live: $OTRV_MODELS, else ~/.otrv/models (never inside the repo)."""
    return Path(os.environ.get("OTRV_MODELS", Path.home() / ".otrv" / "models")).expanduser()


def clear(dest: Path) -> None:
    for name in ("car.glb", "car.json"):
        (dest / name).unlink(missing_ok=True)
    shutil.rmtree(dest / "car", ignore_errors=True)
    print("removed; the viewer uses the built-in car again")


def sync_into(viewer_dir: Path) -> None:
    """Copy the installed model (if any) next to the served viewer as models/; remove stale copies."""
    out = viewer_dir / "models"
    shutil.rmtree(out, ignore_errors=True)
    src = models_dir()
    if not (src / "car.json").exists():
        return
    out.mkdir(parents=True)
    for item in ("car.glb", "car.json", "car"):
        p = src / item
        if p.is_dir():
            shutil.copytree(p, out / item)
        elif p.exists():
            shutil.copy2(p, out / item)
