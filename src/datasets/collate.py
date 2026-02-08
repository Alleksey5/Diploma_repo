import torch
import torch.nn.functional as F

def closest_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

def custom_collate(batch):
    audio_list, tg_list = [], []
    audio_lens, tg_lens = [], []
    file_ids, sizes = [], []

    for sample in batch:
        for seg in sample:
            a = seg["audio"]
            t = seg["tg_audio"]

            # гарантируем форму (1, T) чтобы stack был предсказуемый
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if t.dim() == 1:
                t = t.unsqueeze(0)

            audio_list.append(a)
            tg_list.append(t)

            audio_lens.append(a.shape[-1])
            tg_lens.append(t.shape[-1])

            file_ids.append(seg.get("file_id", ""))
            sizes.append(seg.get("size", 0))

    max_len = max(max(audio_lens), max(tg_lens))
    pad_len = closest_power_of_two(max_len)

    def pad_right(x):
        pad_size = pad_len - x.shape[-1]
        return F.pad(x, (0, pad_size), value=0.0)

    audio = torch.stack([pad_right(x) for x in audio_list], dim=0)      # (B, 1, pad_len)
    tg_audio = torch.stack([pad_right(x) for x in tg_list], dim=0)      # (B, 1, pad_len)

    return {
        "audio": audio,
        "tg_audio": tg_audio,
        "audio_len": torch.tensor(audio_lens, dtype=torch.long),  # (B,)
        "tg_len": torch.tensor(tg_lens, dtype=torch.long),        # (B,)
        "file_id": file_ids,
        "size": sizes,
        "pad_len": pad_len,
    }
