"""The model's forward path imports no solver and calls no linear solve: only torch, numpy and the standard library.

  PYTHONPATH=src python -B tests/no_solver.py
"""
import ast, pathlib, re, sys

pkg = pathlib.Path(__file__).resolve().parents[1] / 'src/fsi_mgn'
imports = set()
for f in pkg.glob('*.py'):
    for node in ast.walk(ast.parse(f.read_text())):
        if isinstance(node, ast.Import): imports |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0: imports.add(node.module.split('.')[0])
foreign = sorted(imports - set(sys.stdlib_module_names) - {'torch', 'numpy'})
src = ''.join((pkg / f).read_text() for f in ('model.py', 'net.py', 'physics.py'))
calls = re.findall(r'linalg\.(?:solve|lstsq|inv)|sparse\.linalg|gmres|\bwhile\b', src)
print('imports:', sorted(imports), '| foreign:', foreign or 'none', '| solver calls:', calls or 'none')
sys.exit(1 if foreign or calls else 0)
