import torch
import torch.nn.functional as F


def _to_1d(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: [1, T] or [T]
    return: [T]
    """
    if wav.dim() == 2:
        return wav.squeeze(0)
    return wav


def collate_fn(dataset_items: list[dict]):
    """
    Expected keys per item:
      wav:      Tensor [1, T] or [T]
      gt_wav:   Tensor [1, T] or [T]
      melspec:  Tensor [n_mels, frames]
      gt_melspec: Tensor [n_mels, frames]
      path: str
    """

    paths = [it["path"] for it in dataset_items]

    wavs = [_to_1d(it["wav"]) for it in dataset_items]
    gt_wavs = [_to_1d(it["gt_wav"]) for it in dataset_items]

    mels = [it["melspec"] for it in dataset_items]
    gt_mels = [it["gt_melspec"] for it in dataset_items]

    # lengths (Tensor[int])
    initial_len = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    initial_gt_len = torch.tensor([w.shape[-1] for w in gt_wavs], dtype=torch.long)

    initial_melspec_len = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
    initial_gt_melspec_len = torch.tensor([m.shape[-1] for m in gt_mels], dtype=torch.long)

    # pad wavs to max T
    max_len_wav = int(initial_len.max().item())
    max_len_gt_wav = int(initial_gt_len.max().item())

    padded_wavs = torch.stack([F.pad(w, (0, max_len_wav - w.shape[-1])) for w in wavs])  # [B, T]
    padded_gt_wavs = torch.stack([F.pad(w, (0, max_len_gt_wav - w.shape[-1])) for w in gt_wavs])  # [B, T]

    # pad mels to max frames
    max_len_spec = int(initial_melspec_len.max().item())
    max_len_gt_spec = int(initial_gt_melspec_len.max().item())

    padded_mels = torch.stack([F.pad(m, (0, max_len_spec - m.shape[-1], 0, 0)) for m in mels])  # [B, n_mels, F]
    padded_gt_mels = torch.stack([F.pad(m, (0, max_len_gt_spec - m.shape[-1], 0, 0)) for m in gt_mels])  # [B, n_mels, F]

    return {
        "wav": padded_wavs.unsqueeze(1),              # [B, 1, T]
        "gt_wav": padded_gt_wavs.unsqueeze(1),        # [B, 1, T]
        "melspec": padded_mels,                       # [B, n_mels, F]
        "gt_melspec": padded_gt_mels,                 # [B, n_mels, F]
        "initial_len": initial_len,
        "initial_gt_len": initial_gt_len,
        "initial_melspec_len": initial_melspec_len,
        "initial_gt_melspec_len": initial_gt_melspec_len,
        "path": paths,
    }
