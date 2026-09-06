#!/usr/bin/env python3
"""Generate fixed-face reductions of the shipped Tucker high-z maps.

Maximal high-z steps are selected by either the normalized C cap or normalized
s cap.  The accepted untruncated step therefore lies on x_C = +/-0.985 or
x_s = +/-0.985 by construction.  This generator performs that contraction
offline from the shipped binary64 Tucker coefficients at high precision.

Representation is selected for runtime speed, not uniformity:
  * endpoint map: reconstructed 2-D Chebyshev value + normal-derivative
    surfaces for all panels/faces;
  * integral map: reconstructed 2-D surfaces for mid/upper, where measured
    faster; precontracted Tucker face cores for near, whose smaller ranks are
    measured faster than a dense 2-D reconstruction.

There is no physical refit.  Inputs are interpreted as their exact binary64
values, contractions use 100 decimal digits, and emitted coefficients are
rounded once to binary64 hexadecimal literals.
"""
from __future__ import annotations

import argparse
import mpmath as mp
import re
from pathlib import Path

mp.mp.dps = 100
SAFETY = mp.mpf(985) / 1000


def parse_defines(path: Path) -> dict[str, int]:
    text = path.read_text()
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"^#define\s+(\w+)\s+(\d+)\s*$", text, re.M)
    }


def parse_array(path: Path, name: str) -> list[float]:
    text = path.read_text()
    m = re.search(
        rf"const\s+double\s+{re.escape(name)}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}};",
        text,
        re.S,
    )
    if not m:
        raise RuntimeError(f"array {name} not found in {path}")
    vals: list[float] = []
    for tok in m.group(1).replace("\n", " ").split(","):
        tok = tok.strip()
        if tok:
            vals.append(
                float.fromhex(tok)
                if tok.lower().startswith(("0x", "+0x", "-0x"))
                else float(tok)
            )
    return vals


def mpf64(x: float) -> mp.mpf:
    # mp.mpf(float) captures the exact binary64 value at sufficient precision.
    return mp.mpf(x)


def cheb_vd(n: int, x: mp.mpf) -> tuple[list[mp.mpf], list[mp.mpf]]:
    t = [mp.mpf(0)] * n
    d = [mp.mpf(0)] * n
    t[0] = 1
    if n > 1:
        t[1] = x
        d[1] = 1
    for k in range(2, n):
        t[k] = 2 * x * t[k - 1] - t[k - 2]
        d[k] = 2 * t[k - 1] + 2 * x * d[k - 1] - d[k - 2]
    return t, d


def reshape(vals: list[float], shape: tuple[int, ...]):
    total = 1
    for s in shape:
        total *= s
    assert len(vals) == total, (len(vals), shape)
    it = iter(vals)
    if len(shape) == 2:
        n0, n1 = shape
        return [[mpf64(next(it)) for _ in range(n1)] for _ in range(n0)]
    if len(shape) == 3:
        n0, n1, n2 = shape
        return [
            [[mpf64(next(it)) for _ in range(n2)] for _ in range(n1)]
            for _ in range(n0)
        ]
    raise ValueError(shape)


def project_mode(
    U: list[list[mp.mpf]], x: mp.mpf
) -> tuple[list[mp.mpf], list[mp.mpf]]:
    n = len(U)
    r = len(U[0])
    t, d = cheb_vd(n, x)
    a = [mp.mpf(0)] * r
    da = [mp.mpf(0)] * r
    for j in range(r):
        a[j] = mp.fsum(U[k][j] * t[k] for k in range(n))
        da[j] = mp.fsum(U[k][j] * d[k] for k in range(n))
    return a, da


def contract_c_core(G, U1, xface: mp.mpf):
    r0, r1, r2 = len(G), len(G[0]), len(G[0][0])
    b, db = project_mode(U1, xface)
    H = [
        [mp.fsum(G[i][j][k] * b[j] for j in range(r1)) for k in range(r2)]
        for i in range(r0)
    ]
    Hd = [
        [mp.fsum(G[i][j][k] * db[j] for j in range(r1)) for k in range(r2)]
        for i in range(r0)
    ]
    return H, Hd


def contract_s_core(G, U2, xface: mp.mpf):
    r0, r1, r2 = len(G), len(G[0]), len(G[0][0])
    c, dc = project_mode(U2, xface)
    H = [
        [mp.fsum(G[i][j][k] * c[k] for k in range(r2)) for j in range(r1)]
        for i in range(r0)
    ]
    Hd = [
        [mp.fsum(G[i][j][k] * dc[k] for k in range(r2)) for j in range(r1)]
        for i in range(r0)
    ]
    return H, Hd


def reconstruct_c_face(G, U0, U1, U2, xface: mp.mpf):
    H, Hd = contract_c_core(G, U1, xface)
    n0, r0 = len(U0), len(U0[0])
    n2, r2 = len(U2), len(U2[0])
    B = [
        [
            mp.fsum(
                U0[a][i] * H[i][k] * U2[c][k]
                for i in range(r0)
                for k in range(r2)
            )
            for c in range(n2)
        ]
        for a in range(n0)
    ]
    Bd = [
        [
            mp.fsum(
                U0[a][i] * Hd[i][k] * U2[c][k]
                for i in range(r0)
                for k in range(r2)
            )
            for c in range(n2)
        ]
        for a in range(n0)
    ]
    return B, Bd


def reconstruct_s_face(G, U0, U1, U2, xface: mp.mpf):
    H, Hd = contract_s_core(G, U2, xface)
    n0, r0 = len(U0), len(U0[0])
    n1, r1 = len(U1), len(U1[0])
    B = [
        [
            mp.fsum(
                U0[a][i] * H[i][j] * U1[b][j]
                for i in range(r0)
                for j in range(r1)
            )
            for b in range(n1)
        ]
        for a in range(n0)
    ]
    Bd = [
        [
            mp.fsum(
                U0[a][i] * Hd[i][j] * U1[b][j]
                for i in range(r0)
                for j in range(r1)
            )
            for b in range(n1)
        ]
        for a in range(n0)
    ]
    return B, Bd


def load_panel(
    prefix: str,
    src: Path,
    defs: dict[str, int],
    macro_prefix: str,
    panel: str,
):
    P = panel.upper()
    lp = panel.lower()
    n0 = defs[f"{macro_prefix}_{P}_N0"]
    n1 = defs[f"{macro_prefix}_{P}_N1"]
    n2 = defs[f"{macro_prefix}_{P}_N2"]
    r0 = defs[f"{macro_prefix}_{P}_R0"]
    r1 = defs[f"{macro_prefix}_{P}_R1"]
    r2 = defs[f"{macro_prefix}_{P}_R2"]
    G = reshape(parse_array(src, f"{prefix}_{lp}_g"), (r0, r1, r2))
    U0 = reshape(parse_array(src, f"{prefix}_{lp}_u0"), (n0, r0))
    U1 = reshape(parse_array(src, f"{prefix}_{lp}_u1"), (n1, r1))
    U2 = reshape(parse_array(src, f"{prefix}_{lp}_u2"), (n2, r2))
    return (n0, n1, n2, r0, r1, r2, G, U0, U1, U2)


def dense_face_entries(prefix: str, panel: str, G, U0, U1, U2):
    entries = []
    for sign_name, xf in (("p", SAFETY), ("m", -SAFETY)):
        B, Bd = reconstruct_c_face(G, U0, U1, U2, xf)
        entries.append((f"{prefix}_{panel}_cface_{sign_name}", B))
        entries.append((f"{prefix}_{panel}_cface_d_{sign_name}", Bd))
        B, Bd = reconstruct_s_face(G, U0, U1, U2, xf)
        entries.append((f"{prefix}_{panel}_sface_{sign_name}", B))
        entries.append((f"{prefix}_{panel}_sface_d_{sign_name}", Bd))
    return entries


def near_integral_core_entries(prefix: str, G, U1, U2):
    entries = []
    for sign_name, xf in (("p", SAFETY), ("m", -SAFETY)):
        H, Hd = contract_c_core(G, U1, xf)
        entries.append((f"{prefix}_near_cface_core_{sign_name}", H))
        entries.append((f"{prefix}_near_cface_d_core_{sign_name}", Hd))
        H, Hd = contract_s_core(G, U2, xf)
        entries.append((f"{prefix}_near_sface_core_{sign_name}", H))
        entries.append((f"{prefix}_near_sface_d_core_{sign_name}", Hd))
    return entries


def flat_hex(M) -> list[str]:
    return [float(v).hex() for row in M for v in row]


def emit_array(out: list[str], name: str, M) -> int:
    vals = flat_hex(M)
    out.append(f"const double {name}[{len(vals)}] = {{")
    for i in range(0, len(vals), 4):
        out.append(
            "    "
            + ", ".join(vals[i : i + 4])
            + ("," if i + 4 < len(vals) else "")
        )
    out.append("};")
    out.append("")
    return len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.root.resolve()
    gen = root / "generated"

    entries = []

    # Endpoint: direct 2-D face surfaces win for all shipped panels.
    hzt_defs = parse_defines(gen / "highz_tucker_coeffs.h")
    for panel in ("mid", "upper", "near"):
        _, _, _, _, _, _, G, U0, U1, U2 = load_panel(
            "cflow_hzt",
            gen / "highz_tucker_coeffs.c",
            hzt_defs,
            "CFLOW_HZT",
            panel,
        )
        entries += dense_face_entries("cflow_hzt", panel, G, U0, U1, U2)

    # Integral: direct 2-D is faster for mid/upper; near's (6,4,4) Tucker
    # ranks make a precontracted face core faster and smaller.
    int_defs = parse_defines(gen / "integral_tables.h")
    for panel in ("mid", "upper", "near"):
        _, _, _, _, _, _, G, U0, U1, U2 = load_panel(
            "cflow_int_hz",
            gen / "integral_tables.c",
            int_defs,
            "CFLOW_INT_HZ",
            panel,
        )
        if panel == "near":
            entries += near_integral_core_entries("cflow_int_hz", G, U1, U2)
        else:
            entries += dense_face_entries("cflow_int_hz", panel, G, U0, U1, U2)

    h = [
        "#ifndef CFLOW_HIGHZ_FACE_TABLES_H",
        "#define CFLOW_HIGHZ_FACE_TABLES_H",
        "",
        "/* Generated by tools/generate_highz_face_tables.py. */",
    ]
    c = [
        '#include "highz_face_tables.h"',
        "",
        "/* Fixed-face reductions of the shipped Tucker high-z maps. */",
        "",
    ]
    for name, M in entries:
        count = len(M) * len(M[0])
        h.append(f"extern const double {name}[{count}];")
        emitted = emit_array(c, name, M)
        assert emitted == count
    h += ["", "#endif", ""]
    (gen / "highz_face_tables.h").write_text("\n".join(h))
    (gen / "highz_face_tables.c").write_text("\n".join(c))


if __name__ == "__main__":
    main()
