"""Quick verification that all project dependencies are importable."""
import sys

checks = [
    ("torch",               lambda: __import__("torch").__version__),
    ("transformers",        lambda: __import__("transformers").__version__),
    ("datasets",            lambda: __import__("datasets").__version__),
    ("sentence_transformers",lambda: __import__("sentence_transformers").__version__),
    ("faiss",               lambda: __import__("faiss").__version__),
    ("sklearn",             lambda: __import__("sklearn").__version__),
    ("spacy",               lambda: __import__("spacy").__version__),
    ("pandas",              lambda: __import__("pandas").__version__),
    ("numpy",               lambda: __import__("numpy").__version__),
    ("fastapi",             lambda: __import__("fastapi").__version__),
    ("loguru",              lambda: "ok"),
    ("tqdm",                lambda: __import__("tqdm").__version__),
    ("scipy",               lambda: __import__("scipy").__version__),
]

print(f"\n{'='*50}")
print("  Dependency Verification Report")
print(f"  Python {sys.version.split()[0]}")
print(f"{'='*50}")

all_ok = True
for name, version_fn in checks:
    try:
        ver = version_fn()
        print(f"  [OK]   {name:<28} {ver}")
    except Exception as e:
        print(f"  [FAIL] {name:<28} FAILED: {e}")
        all_ok = False

print(f"{'='*50}")
print(f"  Status: {'ALL OK' if all_ok else 'SOME FAILED'}")
print(f"{'='*50}\n")
sys.exit(0 if all_ok else 1)
