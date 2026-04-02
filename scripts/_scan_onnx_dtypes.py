"""One-off: scan model_assets/*.onnx dtypes (I/O, initializers, value_info)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto

ROOT = Path(__file__).resolve().parent.parent / "model_assets"

DT_NAMES: dict[int, str] = {
    TensorProto.UNDEFINED: "undefined",
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.COMPLEX64: "complex64",
    TensorProto.COMPLEX128: "complex128",
    TensorProto.BFLOAT16: "bfloat16",
}
for attr in dir(TensorProto):
    if attr.startswith("FLOAT8") or attr.startswith("UINT4") or attr.startswith("INT4"):
        v = getattr(TensorProto, attr, None)
        if isinstance(v, int) and v not in DT_NAMES:
            DT_NAMES[v] = attr.lower()

FLOAT_LIKE = frozenset({"float32", "float16", "float64", "bfloat16"})


def dt_label(code: int) -> str:
    return DT_NAMES.get(code, f"dtype_{code}")


def tensor_types_from_value_infos(vinfos) -> Counter:
    c: Counter = Counter()
    for vi in vinfos:
        t = vi.type.tensor_type.elem_type
        if t != TensorProto.UNDEFINED:
            c[dt_label(t)] += 1
    return c


def uniq(seq: list[str]) -> list[str]:
    seen: list[str] = []
    for x in seq:
        if x not in seen:
            seen.append(x)
    return seen


def dominant_float(counter: Counter) -> str | None:
    fc = {k: v for k, v in counter.items() if k in FLOAT_LIKE}
    if not fc:
        return None
    return max(fc.items(), key=lambda x: x[1])[0]


def summarize_model(path: Path) -> dict:
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph

    in_types = [dt_label(i.type.tensor_type.elem_type) for i in g.input]
    out_types = [dt_label(o.type.tensor_type.elem_type) for o in g.output]

    init_counter: Counter = Counter()
    for init in g.initializer:
        init_counter[dt_label(init.data_type)] += 1

    vi_counter = tensor_types_from_value_infos(g.value_info)

    fp8_like = {
        k
        for k in init_counter
        if "float8" in k or "uint4" in k or "int4" in k or k.startswith("dtype_")
    }

    dom_init = dominant_float(init_counter)
    float_inits = {k: v for k, v in init_counter.items() if k in FLOAT_LIKE}

    if fp8_like:
        kind = "FP8 u otro (revisar)"
    elif len(float_inits) > 1:
        parts = [f"{k}:{v}" for k, v in sorted(float_inits.items(), key=lambda x: (-x[1], x))]
        kind = "Mixto pesos — " + ", ".join(parts)
    elif dom_init == "float16":
        kind = "FP16"
    elif dom_init == "float32":
        kind = "FP32"
    elif dom_init == "bfloat16":
        kind = "BF16"
    elif dom_init == "float64":
        kind = "FP64"
    elif init_counter:
        mc = init_counter.most_common(2)
        kind = "No-FP predominante — " + ", ".join(f"{a}:{b}" for a, b in mc)
    else:
        kind = "Sin inicializadores en .onnx"

    io_in = ",".join(uniq(in_types)) if in_types else "-"
    io_out = ",".join(uniq(out_types)) if out_types else "-"

    vi_note = "-"
    if vi_counter:
        vi_note = "; ".join(f"{k}:{v}" for k, v in vi_counter.most_common(4))

    n_init = sum(init_counter.values())

    return {
        "rel": str(path.relative_to(ROOT)),
        "io_in": io_in,
        "io_out": io_out,
        "kind": kind,
        "dom_weights": dom_init or "-",
        "n_init": n_init,
        "init_mix": dict(init_counter),
        "vi_top": vi_note,
    }


def main() -> None:
    skip_name = "unopt-backup"
    paths: list[Path] = []
    for p in ROOT.rglob("*.onnx"):
        if skip_name in p.parts:
            continue
        paths.append(p)
    paths.sort(key=lambda x: str(x).lower())

    rows: list[dict] = []
    for p in paths:
        try:
            rows.append(summarize_model(p))
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "rel": str(p.relative_to(ROOT)),
                    "io_in": "ERROR",
                    "io_out": str(e)[:80],
                    "kind": "-",
                    "dom_weights": "-",
                    "n_init": 0,
                    "init_mix": {},
                    "vi_top": "-",
                }
            )

    print("| Archivo | Pesos (predom.) | Clasificación | Entradas | Salidas | #Init | value_info (top) |")
    print("|---------|-----------------|---------------|----------|---------|-------|------------------|")
    for r in rows:
        vi = r["vi_top"]
        if len(vi) > 55:
            vi = vi[:52] + "…"
        print(
            f"| `{r['rel']}` | {r['dom_weights']} | {r['kind']} | {r['io_in']} | {r['io_out']} | {r['n_init']} | {vi} |"
        )
    print()
    print("Total:", len(rows))
    print()
    print("Notas:")
    print("- Pesos = tipos en graph.initializer (constantes empaquetadas en el ONNX).")
    print("- Mixto = más de un tipo flotante en inicializadores (p. ej. parte FP32 + parte FP16).")
    print("- value_info = anotaciones de tipo en el grafo (a veces incompletas); no sustituye ejecutar ORT.")
    print("- Excluido: carpeta unopt-backup (copias).")


if __name__ == "__main__":
    main()
