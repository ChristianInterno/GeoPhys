"""
Frozen backbones and frame preprocessing.

Four encoders, none of them trained or fine-tuned here: DINOv2 and DINOv3 (ViT-L) and
two convolutional networks modelled on the primate ventral stream, CORnet-S and
VOneNet. Each frame of a video is passed through the network once (see extract.py).

VOneNet note: the official VOneNet checkpoint fails to load from its published URL.
We therefore instantiate the VOneNet architecture with a ResNet50 backend and load
ImageNet weights into that backend. This is what produced every VOneNet number we
report, and the loader prints which path it took.
"""
import numpy as np

TMAX = 60          # frames kept per video

# the readout layer of each encoder used by every configuration
READOUT = {"dinov2": "12", "dinov3": "18", "cornet": "IT", "vonenet": "vone_block"}


# ------------------------------------------------------------------- frames --
def load_frames(path, max_frames=TMAX):
    """Decode a video. If it has n > max_frames frames, keep every k-th frame with
    k = n // max_frames, and the first max_frames of those."""
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
    return frames


def prep_frames(frames, prep):
    """Two preprocessing modes.
    direct : resize straight to 224x224 (changes aspect ratio)
    hf     : resize short side to 256, centre-crop 224 (the HuggingFace convention)
    """
    from PIL import Image
    if prep not in ("direct", "hf"):
        raise ValueError("unknown preprocessing: %s (direct or hf)" % prep)
    out = []
    for a in frames:
        im = Image.fromarray(a)
        if prep == "direct":
            im = im.resize((224, 224), Image.BICUBIC)
        else:
            w, h = im.size
            sc = 256 / min(w, h)
            im = im.resize((int(round(w * sc)), int(round(h * sc))), Image.BICUBIC)
            w, h = im.size
            l, t = (w - 224) // 2, (h - 224) // 2
            im = im.crop((l, t, l + 224, t + 224))
        out.append(np.asarray(im))
    return np.stack(out)


# ----------------------------------------------------------------- backbones --
class VitBackbone:
    """DINOv2-large or DINOv3-ViT-L/16, readout at any of the 24 blocks."""
    ATOMS, IS_VIT = ["flat", "mean", "cls"], True

    def __init__(self, name, device="cuda"):
        import torch
        from transformers import AutoModel
        hf = ("facebook/dinov2-large" if name == "dinov2"
              else "facebook/dinov3-vitl16-pretrain-lvd1689m")
        try:
            self.m = AutoModel.from_pretrained(hf, local_files_only=True)
        except Exception:
            self.m = AutoModel.from_pretrained(hf)
        self.m = self.m.eval().to(device)
        self.nsp = 1 if name == "dinov2" else 5      # non-patch tokens (CLS, registers)
        self.layers = [str(i) for i in range(1, 25)]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


class _Grab:
    def __init__(self): self.o = None
    def __call__(self, m, i, o): self.o = o.detach()


class ConvBackbone:
    """CORnet-S (V1, V2, V4, IT) or VOneNet (vone_block, layer1-4); layer outputs are
    captured with forward hooks."""
    ATOMS, IS_VIT = ["flat", "mean"], False

    def __init__(self, name, device="cuda"):
        import torch
        self.hooks = {}
        if name == "cornet":
            from cornet import cornet_s
            m = cornet_s(pretrained=True, map_location="cpu")
            m = getattr(m, "module", m).eval().to(device)
            self.layers = ["V1", "V2", "V4", "IT"]
            mods = {L: getattr(m, L) for L in self.layers}
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            import vonenet
            try:
                m = vonenet.get_model(model_arch="resnet50", pretrained=True)
                print("  VOneNet: official checkpoint loaded")
            except Exception:
                print("  VOneNet checkpoint unavailable -> ResNet50 ImageNet backend")
                m = vonenet.get_model(model_arch="resnet50", pretrained=False)
                from torchvision.models import resnet50, ResNet50_Weights
                rsd = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
                mm = getattr(m, "module", m)
                bsd = mm.model.state_dict()
                ld = {k: v for k, v in rsd.items() if k in bsd and bsd[k].shape == v.shape}
                mm.model.load_state_dict(ld, strict=False)
                print("  loaded %d/%d backend tensors" % (len(ld), len(bsd)))
            m = getattr(m, "module", m).eval().to(device)
            self.layers = ["vone_block", "layer1", "layer2", "layer3", "layer4"]
            mods = {"vone_block": m.vone_block, "layer1": m.model.layer1,
                    "layer2": m.model.layer2, "layer3": m.model.layer3,
                    "layer4": m.model.layer4}
            self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        for L, mod in mods.items():
            h = _Grab(); mod.register_forward_hook(h); self.hooks[L] = h
        self.m = m


def get_backbone(name, device="cuda"):
    """name: dinov2 | dinov3 | cornet | vonenet"""
    if name in ("dinov2", "dinov3"):
        return VitBackbone(name, device)
    return ConvBackbone(name, device)
