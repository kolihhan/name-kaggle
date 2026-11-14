"""Build one private frozen likelihood diagnostic of the exact 0.683 champion."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

from scripts.ntl_areeb8_wemm_paired84_private_20261113 import CHAMPION, CHAMP_CODE_SHA256
from scripts.ntl_wemm_isolated_fold0_smoke_20261113 import cell_text, new_code

ROOT = Path(__file__).resolve().parents[1]
DEST = "kolihhan/ntl-areeb8-glm-observational-nov14"
PUBLIC = "estopadilla/0-689-real-glm-5-3-likelihood-on-2xt4"
PUBLIC_SHA = "78811af1627e045960c60d0c091b5d518aa379aa9f4385448d87cd63b98b1b16"
PAYLOAD_SHA = "b0bb99ed84e417a8249600fb28f1f69c21641659510bc7ede15b3a73a05a785d"
CORPUS = "nischaydnk/ntl-topic-synthetic-18llms"
ASSETS = [CORPUS, "jamesliebel/glm53-iq3xxs-segs-a-mirror",
          "jamesliebel/glm53-iq3xxs-segs-b-mirror", "jamesliebel/glm53-flash-iq3xxs-segs-mirror"]
BINARY_SOURCE = "estopadilla/ntl-glm-likelihood-train"


def likelihood_payload() -> str:
    with urlopen(f"https://api.kaggle.com/v1/kernels/pull/{PUBLIC}", timeout=60) as response:
        pull = json.load(response)
    source = pull["blob"]["source"]
    if pull["metadata"]["currentVersionNumber"] != 6 or hashlib.sha256(source.encode()).hexdigest() != PUBLIC_SHA:
        raise ValueError("public GLM notebook version or source changed")
    assignments = [line for cell in json.loads(source)["cells"] if cell.get("cell_type") == "code"
                   for line in cell_text(cell).splitlines() if line.startswith("V1_PAYLOAD = ")]
    [assignment] = assignments
    payload = ast.literal_eval(assignment.split(" = ", 1)[1])
    if hashlib.sha256(base64.b64decode(payload, validate=True)).hexdigest() != PAYLOAD_SHA:
        raise ValueError("public GLM runtime payload changed")
    return payload


def build(champion_dir: Path, out: Path, payload: str) -> dict:
    [notebook_path] = champion_dir.glob("*.ipynb")
    notebook = json.loads(notebook_path.read_text())
    cells = notebook["cells"]
    source = "\n".join(cell_text(c) for c in cells if c.get("cell_type") == "code")
    meta = json.loads((champion_dir / "kernel-metadata.json").read_text())
    if hashlib.sha256(source.encode()).hexdigest() != CHAMP_CODE_SHA256:
        raise ValueError("scored 0.683 executable source changed")
    if (meta["id"] != CHAMPION or not meta["is_private"]
            or meta["competition_sources"] != ["name-that-llm"]
            or len(meta["dataset_sources"]) != 17 or len(meta["model_sources"]) != 8
            or "FAST_COMMIT = False" not in source or source.count('"gauss": 0.0') != 2
            or "_ga.fit(" in source or "NO_NEW_TRAINING: supervised classifier fitting forbidden" not in source):
        raise ValueError("champion asset or no-fitting controls changed")
    if hashlib.sha256(base64.b64decode(payload, validate=True)).hexdigest() != PAYLOAD_SHA:
        raise ValueError("wrong frozen inference payload")

    runtime = (ROOT / "kaggle/ntl_glm_observational_pair.py").read_text()
    compile(runtime, "ntl_glm_observational_pair.py", "exec")
    needle = re.compile(r"(?m)^DATA = next\(p for p in .*os\.path\.exists\(p \+ '/train\.csv'\)\)$")
    [index] = [i for i, cell in enumerate(cells) if cell.get("cell_type") == "code"
               and needle.search(cell_text(cell))]
    cells[index]["source"] = needle.sub(
        lambda m: m[0] + "\nDATA = str(ntl_glm_runner.prepare(DATA))",
        cell_text(cells[index]), count=1,
    ).splitlines(keepends=True)
    cells.insert(index, new_code(
        f"from pathlib import Path\n"
        f"Path('/kaggle/temp/ntl_glm_runner.py').parent.mkdir(parents=True, exist_ok=True)\n"
        f"Path('/kaggle/temp/ntl_glm_runner.py').write_text({runtime!r})\n"
        f"import sys\nsys.path.insert(0, '/kaggle/temp')\nimport ntl_glm_runner\n", 81))
    cells.append(new_code(f"ntl_glm_runner.compare(DATA, {payload!r})\n", 82))

    for key in ("id_no", "docker_image"):
        meta.pop(key, None)
    meta.update(id=DEST, title=DEST.split("/")[1], is_private=True, enable_gpu=True,
                enable_internet=False, enable_tpu=False, machine_shape="NvidiaTeslaT4")
    for kind, additions in (("dataset_sources", ASSETS), ("kernel_sources", [BINARY_SOURCE])):
        meta[kind] = list(dict.fromkeys([*meta.get(kind, []), *additions]))
    code_file = meta["code_file"]
    if Path(code_file).name != code_file or not code_file.endswith(".ipynb"):
        raise ValueError("unexpected notebook file path")
    out.mkdir(parents=True, exist_ok=False)
    (out / code_file).write_text(json.dumps(notebook, ensure_ascii=False))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"kernel": DEST, "champion_code_sha256": CHAMP_CODE_SHA256,
            "public_likelihood_payload_sha256": PAYLOAD_SHA, "notebook_rows": 252,
            "license": "CC0 source corpus", "scope": "observational development diagnostic"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("NTL_GLM_DIAGNOSTIC_RENDER=" + json.dumps(build(args.champion, args.out, likelihood_payload())), flush=True)
