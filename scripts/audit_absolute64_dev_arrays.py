"""Identity-only reuse of the independently checked fixed-development arithmetic."""
import hashlib
from pathlib import Path

path = Path(__file__).with_name('audit_g_dev_arrays.py')
source = path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest() == '003006ac4bc7c2ef260135feb3893d9ebe40d719a98da64755d63447df66024b'
assert source.count('G_AR5_RESET') == 1
assert source.count('G_rearCl') == 1 and source.count('G_totalCd') == 1
source = source.replace('G_AR5_RESET', 'B_ABSOLUTE64')
source = source.replace('G_rearCl', 'absolute64_rearCl').replace('G_totalCd', 'absolute64_totalCd')
source = source.replace('B/G matched array', 'B/absolute64 matched array')
exec(compile(source, str(path), 'exec'))
