"""Normalise files written as UTF-16LE (no BOM) to UTF-8. Usage: python fixenc.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent
n = 0
for p in root.rglob("*"):
    if not p.is_file() or p.suffix not in {".py", ".txt", ".yaml", ".yml", ".md", ".csv", ".toml"}:
        continue
    if any(part in {"dataset", "outputs", "__pycache__", ".git"} for part in p.relative_to(root).parts):
        continue
    b = p.read_bytes()
    if len(b) >= 4 and b[1] == 0 and b[3] == 0 and b[0] != 0:
        t = b.decode("utf-16-le")
        if t.startswith("\ufeff"):
            t = t[1:]
        p.write_bytes(t.encode("utf-8"))
        n += 1
        print("fixed", p.relative_to(root))
print("fixed total", n)
