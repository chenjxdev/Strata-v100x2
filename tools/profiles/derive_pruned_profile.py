"""Derive the expert profile of an expert-pruned model from the profile of its base model.

The base profile ranks (layer, expert) pairs, most routed first. The pruned model keeps, per layer, a subset of the base
experts in a given order (expert j of layer L is base expert kept[L][j]). The derived profile keeps the base order of the
kept experts and renumbers each one to its index in the pruned model.

    python3 tools/profiles/derive_pruned_profile.py BASE.prof KEPT.json OUT.prof

KEPT.json: {"layers": L, "experts_per_layer": E, "kept_per_layer": K, "kept_experts": {"0": [...], ...}} (ISTA-DASLab's
kept-experts file).
"""
import json
import struct
import sys


def read_profile(path):
    raw = open(path, "rb").read()
    version, n_layers, n_expert, slots, n_ranked = struct.unpack_from("<5I", raw, 0)
    pairs = struct.unpack_from("<%dH" % (2 * n_ranked), raw, 20)
    return version, n_layers, n_expert, slots, [(pairs[2 * i], pairs[2 * i + 1]) for i in range(n_ranked)]


def main():
    base, kept_path, out = sys.argv[1:4]
    version, n_layers, n_expert, slots, ranked = read_profile(base)
    kept = json.load(open(kept_path))
    if kept["layers"] != n_layers or kept["experts_per_layer"] != n_expert:
        sys.exit("the kept-experts file is %dx%d, the base profile %dx%d" % (kept["layers"], kept["experts_per_layer"], n_layers, n_expert))
    k = kept["kept_per_layer"]
    index = []
    for layer in range(n_layers):
        ids = kept["kept_experts"][str(layer)]
        if len(ids) != k or len(set(ids)) != k or not all(0 <= e < n_expert for e in ids):
            sys.exit("layer %d: the kept list is not %d distinct base experts" % (layer, k))
        index.append({e: j for j, e in enumerate(ids)})
    derived = [(layer, index[layer][e]) for layer, e in ranked if e in index[layer]]
    if len(derived) != n_layers * k or len(set(derived)) != len(derived):
        sys.exit("the base profile does not rank every kept expert once (%d pairs)" % len(derived))
    with open(out, "wb") as f:
        f.write(struct.pack("<5I", version, n_layers, k, n_layers * k, len(derived)))
        f.write(struct.pack("<%dH" % (2 * len(derived)), *[v for p in derived for v in p]))
    print("%s: %dx%d, %d pairs" % (out, n_layers, k, len(derived)))


if __name__ == "__main__":
    main()
