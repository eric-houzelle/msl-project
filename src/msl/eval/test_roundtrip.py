"""Test the real objective: round-trip Text -> MSL -> Text.

The goal is NOT to translate between human languages. MSL is the machine's
NATIVE language. The test is: can we compress a text into MSL packets and
decompress it back while preserving the meaning?

  FR text -> encoder -> MSL packets -> decoder -> FR text
  EN text -> encoder -> MSL packets -> decoder -> EN text

If the round-trip preserves meaning, MSL is a faithful representation and the
LLM can think natively in it.

Usage:
    python -u -m msl.eval.test_roundtrip --codec runs/v2_p2_translate_0.pt
"""

from __future__ import annotations

import argparse

import torch

from msl.data.ms1 import MS1, TextRenderer
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec_v2 import build_codec_v2
from msl.utils.seeding import default_device, seed_everything


def roundtrip_one(codec, tok, renderer, state, view_id, lang_id, lang_name, device):
    """Text -> MSL -> Text round-trip for one state."""
    src_text = renderer.render(state, view_id)
    src_ids = torch.tensor([tok.encode(src_text, True, True)], dtype=torch.long, device=device)
    src_mask = (src_ids == tok.pad_id).long()

    with torch.no_grad():
        # Encode: text -> packets
        codes, z_q, _, _ = codec.encode_quantize(src_ids, src_mask)
        # Decode: packets -> text (same language)
        lang_tensor = torch.tensor([lang_id], device=device)
        out_ids = codec.decoder.generate(z_q, lang_tensor, tok.bos_id, tok.eos_id,
                                         max_len=80, top_k=5, temperature=0.7)
    gen_text = tok.decode(out_ids[0].tolist())
    return src_text, gen_text, codes


def count_meaning_overlap(src_text, gen_text):
    """Crude semantic overlap: fraction of (entity, attribute, value) triples preserved.

    Extracts patterns like "object 0 has color=2" or "l'objet 0 a color=2"
    from both texts and counts how many match.
    """
    import re

    def extract_facts(text):
        facts = set()
        # Match "object N has KEY=V" or "objet N a KEY=V" or "objet N : KEY vaut V"
        for m in re.finditer(r'(?:object|objet|élément|element|entité|entity)\s*(\d+).{0,30}?(?:has|a|:)\s*(?:the\s+)?(\w+).{0,5}?[=:]\s*(\d+)', text.lower()):
            facts.add((m.group(1), m.group(2), m.group(3)))
        # Match "object N KEY vaut V"
        for m in re.finditer(r'(?:object|objet)\s*(\d+)\s*:\s*(\w+)\s+vaut\s*(\d+)', text.lower()):
            facts.add((m.group(1), m.group(2), m.group(3)))
        # Match "the KEY of object N is V"
        for m in re.finditer(r'(?:the\s+)?(\w+)\s+of\s+(?:object|objet)\s*(\d+)\s+is\s*(\d+)', text.lower()):
            facts.add((m.group(2), m.group(1), m.group(3)))
        return facts

    src_facts = extract_facts(src_text)
    gen_facts = extract_facts(gen_text)
    if not src_facts:
        return 0.0, 0, 0
    overlap = len(src_facts & gen_facts)
    return overlap / len(src_facts), overlap, len(src_facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="runs/v2_p2_translate_0.pt")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    seed_everything(42)
    device = default_device()
    tok = default_tokenizer()

    ckpt = torch.load(args.codec, map_location=device, weights_only=False)
    codec = build_codec_v2(ckpt["cfg"], len(tok.itos), content_token_ids=set()).to(device)
    codec.load_state_dict(ckpt["codec"], strict=False)
    codec.eval()
    print(f"loaded {args.codec} ({ckpt['n_params']:,} params)")

    gen = MS1(min_k=4, max_k=12)
    renderer = TextRenderer()

    print("\n" + "=" * 80)
    print("ALLER-RETOUR FR -> MSL -> FR (le vrai test: le sens est-il preserve ?)")
    print("=" * 80)

    overlaps = []
    for i in range(args.n):
        s = gen.generate(seed=2_000_100 + i, k=8)
        src, gen_text, codes = roundtrip_one(codec, tok, renderer, s, 0, 0, "FR", device)
        overlap, n_match, n_src = count_meaning_overlap(src, gen_text)
        overlaps.append(overlap)
        print(f"\n--- Exemple {i+1} (k={s.difficulty}) ---")
        print(f"  FR source:    {src[:180]}")
        print(f"  FR via MSL:   {gen_text[:180]}")
        print(f"  Faits source: {n_src}, faits preserves: {n_match} ({overlap*100:.0f}%)")
        print(f"  Paquets:      {codes[0, :3].tolist()} ... ({codes.shape[1]} slots)")

    print("\n" + "=" * 80)
    print("ALLER-RETOUR EN -> MSL -> EN")
    print("=" * 80)

    overlaps_en = []
    for i in range(args.n):
        s = gen.generate(seed=2_000_200 + i, k=8)
        src, gen_text, codes = roundtrip_one(codec, tok, renderer, s, 5, 1, "EN", device)
        overlap, n_match, n_src = count_meaning_overlap(src, gen_text)
        overlaps_en.append(overlap)
        print(f"\n--- Exemple {i+1} (k={s.difficulty}) ---")
        print(f"  EN source:    {src[:180]}")
        print(f"  EN via MSL:   {gen_text[:180]}")
        print(f"  Facts source: {n_src}, preserved: {n_match} ({overlap*100:.0f}%)")

    print("\n" + "=" * 80)
    print("VERDICT ROUND-TRIP")
    print("=" * 80)
    fr_mean = sum(overlaps) / len(overlaps) if overlaps else 0
    en_mean = sum(overlaps_en) / len(overlaps_en) if overlaps_en else 0
    print(f"  FR -> MSL -> FR: {fr_mean*100:.0f}% des faits preserves (moyenne)")
    print(f"  EN -> MSL -> EN: {en_mean*100:.0f}% des faits preserves (moyenne)")
    print(f"  Moyenne globale: {(fr_mean+en_mean)/2*100:.0f}%")


if __name__ == "__main__":
    main()
