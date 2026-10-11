"""
Feature extraction: videos -> trajectory stores.

Each video is decoded to at most 60 frames, preprocessed, and passed once through a
frozen backbone. For every readout layer we store the Gram matrix of the frame
trajectory (one per "atom": flattened patch tokens, mean-pooled tokens, CLS) and the
per-token step statistics needed by the token-path signals. Nothing is trained.

Store layout, one folder per (videos, backbone, preprocessing):

  <dataset>_<backbone>_<prep>/          detection (<dataset>: likephys, intphys2, ...)
  v2v_<backbone>_<prep>/                physicsiq, all 16 candidates of every scenario
      gram.npy                [N, L, A, 60, 60]  float32
      t2_{raw,l2,spd}_sq.npy  [N, L, P, 59]      float32   squared token steps
      t2_{raw,l2,spd}_dot.npy [N, L, P, 58]      float32   consecutive step products
      done.npy, meta.json

Usage
  python -m geophys.extract --config configs/geophys_physicsiq.json      # the plan
  python -m geophys.extract --task physicsiq --backbone dinov2 --prep direct --layers 12
  python -m geophys.extract --task detection --dataset likephys --backbone cornet \
                            --prep hf --pairs data/likephys_pairs.json --data-root /path/to/LikePhys
  python -m geophys.extract --status
"""
import os
import json
import glob
import argparse

import numpy as np

from .backbones import get_backbone, load_frames, prep_frames, TMAX

STORES = os.environ.get("GEOPHYS_STORES", os.path.expanduser("~/geophys_stores"))
VARS = ["raw", "l2", "spd"]

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ indexing --
def index_physicsiq(candidates_root):
    """<candidates_root>/<scenario>/candidate_XXX.mp4 -> (scenarios, video paths, index).
    Each index entry records the scenario and the candidate number XXX."""
    scen = sorted(d for d in os.listdir(candidates_root)
                  if os.path.isdir(os.path.join(candidates_root, d)))
    vids, idx = [], []
    for s in scen:
        for f in sorted(os.listdir(os.path.join(candidates_root, s))):
            if f.startswith("candidate_") and f.endswith(".mp4"):
                idx.append({"scen": s, "cand": int(f[10:-4]), "vi": len(vids)})
                vids.append(os.path.join(candidates_root, s, f))
    return scen, vids, idx


def index_detection(pairs_file, data_root=""):
    """pairs_file: JSON list of {group, sub, viol, idx, valid, invalid} where
    'valid'/'invalid' are paths to the plausible and implausible video of a pair."""
    pairs = json.load(open(pairs_file))
    if data_root:                      # paths in the file may be relative to it
        for p in pairs:
            for k in ("valid", "invalid"):
                if not os.path.isabs(p[k]):
                    p[k] = os.path.join(data_root, p[k])
    vids, vmap = [], {}
    for p in pairs:
        for k in ("valid", "invalid"):
            if p[k] not in vmap:
                vmap[p[k]] = len(vids); vids.append(p[k])
    return pairs, vids, vmap


# ----------------------------------------------------------------- extraction --
def _open(path, shape, dtype):
    if os.path.exists(path):
        return np.lib.format.open_memmap(path, mode="r+")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def extract(backbone, prep, vids, meta_extra, out_dir, device="cuda", keep_layers=None):
    """Encode every video once and write its store (see the module docstring)."""
    import torch
    from tqdm import tqdm

    ext = get_backbone(backbone, device)
    Lns, Atoms = ext.layers, ext.ATOMS
    if keep_layers:
        missing = [L for L in keep_layers if L not in Lns]
        if missing:
            raise ValueError("backbone %s has no layer(s) %s" % (backbone, missing))
        Lns = [L for L in Lns if L in keep_layers]
        print("  keeping %d of %d layers: %s" % (len(Lns), len(ext.layers), ",".join(Lns)))
    N = len(vids)
    os.makedirs(out_dir, exist_ok=True)

    # probe the token count on the first video
    probe = prep_frames(load_frames(vids[0])[:8], prep)
    P = _probe_tokens(ext, probe, device)
    print("%s: %d videos | %d layers | P=%d" % (os.path.basename(out_dir), N, len(Lns), P))

    G = _open(os.path.join(out_dir, "gram.npy"),
              (N, len(Lns), len(Atoms), TMAX, TMAX), np.float32)
    SQ = {v: _open(os.path.join(out_dir, "t2_%s_sq.npy" % v),
                   (N, len(Lns), P, TMAX - 1), np.float32) for v in VARS}
    DT = {v: _open(os.path.join(out_dir, "t2_%s_dot.npy" % v),
                   (N, len(Lns), P, TMAX - 2), np.float32) for v in VARS}
    DN = _open(os.path.join(out_dir, "done.npy"), (N,), np.bool_)

    meta = dict(videos=vids, layers=Lns, atoms=Atoms, P=P)
    meta.update(meta_extra)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))

    print("  %d already done" % int(DN[:].sum()))
    for i in tqdm(range(N)):
        if DN[i]:
            continue
        frames = load_frames(vids[i])
        if len(frames) < 4:
            DN[i] = True
            continue
        fx = prep_frames(frames, prep)
        pooled, pat = _forward(ext, fx, device)      # dict L -> (pooled_atoms, tokens)

        for li, L in enumerate(Lns):        # Lns may be a subset; keyed by name
            atoms, tok = pooled[L], pat[L]
            for ai, at in enumerate(Atoms):
                z = atoms[at]
                gm = (z @ z.T).float().cpu().numpy()
                t = gm.shape[0]
                G[i, li, ai, :t, :t] = gm
            # raw token steps, steps of l2-normalised tokens, and raw steps divided by
            # each token's mean speed (speed-normalised, "spd")
            for var in ("raw", "l2"):
                zz = tok if var == "raw" else tok / (tok.norm(dim=-1, keepdim=True) + 1e-8)
                v = zz[1:] - zz[:-1]
                sq = (v * v).sum(-1)
                dt = (v[:-1] * v[1:]).sum(-1)
                SQ[var][i, li, :sq.shape[1], :sq.shape[0]] = sq.T.cpu().numpy().astype(np.float32)
                DT[var][i, li, :dt.shape[1], :dt.shape[0]] = dt.T.cpu().numpy().astype(np.float32)
                if var == "raw":
                    mu = sq.sqrt().mean(0, keepdim=True) + 1e-8
                    SQ["spd"][i, li, :sq.shape[1], :sq.shape[0]] = (sq / mu ** 2).T.cpu().numpy().astype(np.float32)
                    DT["spd"][i, li, :dt.shape[1], :dt.shape[0]] = (dt / mu ** 2).T.cpu().numpy().astype(np.float32)
        DN[i] = True
        if (i + 1) % 50 == 0:
            for A in [G, DN] + list(SQ.values()) + list(DT.values()):
                A.flush()
        torch.cuda.empty_cache()

    for A in [G, DN] + list(SQ.values()) + list(DT.values()):
        A.flush()
    print("done -> %s" % out_dir)


def ensure_store(prefix, backbone, prep, vids, meta_extra, layers, device="cuda"):
    """Extract <prefix>_<backbone>_<prep> for these videos unless it is already complete.
    Returns the store directory."""
    out = os.path.join(STORES, "%s_%s_%s" % (prefix, backbone, prep))
    meta = os.path.join(out, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        if m.get("videos") != list(vids):
            raise SystemExit("%s was extracted from a different set of videos;\n"
                             "use another --tag, or delete that folder" % out)
        if not set(layers) <= set(m.get("layers", [])):
            raise SystemExit("%s holds layers %s but %s are needed;\n"
                             "use another --tag, or delete that folder"
                             % (out, ",".join(m.get("layers", [])), ",".join(layers)))
        done = os.path.join(out, "done.npy")
        if os.path.exists(done) and bool(np.load(done, mmap_mode="r").all()):
            print("  cached: %s" % os.path.basename(out))
            return out
        layers = m["layers"]                     # resume with the same layout
    print("  extracting %s / %s (layers %s)" % (backbone, prep, ",".join(layers)))
    extract(backbone, prep, vids, meta_extra, out, device, list(layers))
    return out


def _probe_tokens(ext, frames, device):
    import torch
    if ext.IS_VIT:
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device) / 255.
        x = (x - ext.mean) / ext.std
        with torch.inference_mode():
            hs = ext.m(x, output_hidden_states=True).hidden_states
        return hs[1].shape[1] - ext.nsp
    return 256


def _forward(ext, frames, device):
    """Returns (pooled_atoms, tokens) dicts keyed by layer.
    pooled_atoms[L] maps atom name -> [T, d]; tokens[L] is [T, P, d]."""
    import torch
    import torch.nn.functional as F
    if ext.IS_VIT:
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device) / 255.
        x = (x - ext.mean) / ext.std
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            hs = ext.m(x, output_hidden_states=True).hidden_states
        pooled, toks = {}, {}
        for li, L in enumerate(ext.layers, start=1):
            h = hs[li].float()
            cls, pat = h[:, 0, :], h[:, ext.nsp:, :]
            T = h.shape[0]
            pooled[L] = {"flat": pat.reshape(T, -1), "mean": pat.mean(1), "cls": cls}
            toks[L] = pat
        return pooled, toks

    # convnet: hooks captured during the forward pass; VOneNet adds random noise in its
    # front-end, seeded per video (GEOPHYS_SEED, default 0) so every run gives the same features
    torch.manual_seed(int(os.environ.get("GEOPHYS_SEED", "0")))
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device) / 255.
    x = (x - ext.mean) / ext.std
    acc = {L: [] for L in ext.layers}
    with torch.inference_mode():
        for i in range(0, x.shape[0], 8):     # fixed chunk size
            ext.m(x[i:i + 8])
            for L in ext.layers:
                acc[L].append(ext.hooks[L].o.float())
    pooled, toks = {}, {}
    for L in ext.layers:
        fm = torch.cat(acc[L])
        T = fm.shape[0]
        gg = F.adaptive_avg_pool2d(fm, 16)
        pat = gg.reshape(T, fm.shape[1], 256).permute(0, 2, 1)
        # "mean" is the mean over the 16x16 adaptive-pooled grid (pat), not over the
        # raw feature map - adaptive_avg_pool2d from 7x7 to 16x16 replicates cells
        # unevenly, so the two differ by several percent
        pooled[L] = {"flat": fm.reshape(T, -1), "mean": pat.mean(1)}
        toks[L] = pat
    return pooled, toks


# ------------------------------------------------------------------- driver --
def manifest_from_config(config_path):
    """Which (store, prep, layers) a configuration actually needs.
    Extracting only these keeps the feature store to a few GB."""
    cfg = json.load(open(config_path))
    members = cfg.get("members")
    if members is None:
        members = [m for block in cfg["datasets"].values() for m in block["members"]]
    need = {}
    for m in members:
        key = (m["backbone"], m["prep"])
        need.setdefault(key, set()).add(str(m["layer"]))
    return {k: sorted(v) for k, v in need.items()}


def main():
    ap = argparse.ArgumentParser(description="GeoPhys feature extraction")
    ap.add_argument("--config", help="extract only what this configuration needs")
    ap.add_argument("--plan", action="store_true",
                    help="with --config: print the extraction plan and exit")
    ap.add_argument("--task", choices=["detection", "physicsiq"])
    ap.add_argument("--dataset", choices=["likephys", "intphys2"],
                    help="detection only")
    ap.add_argument("--backbone", choices=["dinov2", "dinov3", "cornet", "vonenet"])
    ap.add_argument("--prep", choices=["direct", "hf"], default="direct")
    ap.add_argument("--pairs", help="detection: JSON file listing the video pairs")
    ap.add_argument("--data-root", default="",
                    help="detection: directory that relative paths in --pairs start from")
    ap.add_argument("--candidates", default=os.path.join(HERE, "videos", "v2v_candidates"),
                    help="physicsiq: directory of <scenario>/candidate_XXX.mp4")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer names to keep (default: all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.config:
        need = manifest_from_config(a.config)
        print("extraction plan for %s" % os.path.basename(a.config))
        for (store, prep), layers in sorted(need.items()):
            print("  %-16s %-7s layers: %s" % (store, prep, ",".join(layers)))
        print("  %d stores" % len(need))
        if a.plan:
            return
        if not a.backbone:
            print("\nrun one line per store, e.g.:")
            for (store, prep), layers in sorted(need.items()):
                print("  python -m geophys.extract --task %s --backbone %s --prep %s --layers %s"
                      % (a.task or "physicsiq", store, prep, ",".join(layers)))
            return

    if a.status:
        for d in sorted(glob.glob(os.path.join(STORES, "*_*_*"))):
            try:
                dn = np.load(os.path.join(d, "done.npy"), mmap_mode="r")
                print("  %-44s %5d/%5d" % (os.path.basename(d), int(dn.sum()), len(dn)))
            except Exception:
                pass
        return

    if a.task == "physicsiq":
        scen, vids, idx = index_physicsiq(a.candidates)
        out = os.path.join(STORES, "v2v_%s_%s" % (a.backbone, a.prep))
        extract(a.backbone, a.prep, vids, dict(scenarios=scen, index=idx), out, a.device,
                a.layers.split(",") if a.layers else None)
    else:
        pairs, vids, vmap = index_detection(a.pairs, a.data_root)
        out = os.path.join(STORES, "%s_%s_%s" % (a.dataset, a.backbone, a.prep))
        meta_pairs = [{k: p[k] for k in ("group", "sub", "viol", "idx") if k in p} |
                      {"vi": vmap[p["valid"]], "ii": vmap[p["invalid"]]} for p in pairs]
        extract(a.backbone, a.prep, vids, dict(pairs=meta_pairs), out, a.device,
                a.layers.split(",") if a.layers else None)


if __name__ == "__main__":
    main()
