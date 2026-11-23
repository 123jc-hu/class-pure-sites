import json
from pathlib import Path
import subprocess
import sys


def test_adhd200_id_and_diagnosis():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-I', str(root/'adhd200_prepare.py'), 'check_labels'],
                            check=True,capture_output=True,text=True)
    audit = json.loads(result.stdout)
    assert audit['release_counts']=={'train':776,'test':197}
    assert audit['directory_matches']==973 and audit['leading_zero_id_matches']>0
    assert audit['duplicate_release_ids']==0
    assert (audit['train_dx_checked'],audit['test_dx_checked'],audit['test_dx_eligible'])==(716,197,156)
    assert audit['train_dx_mismatches']==audit['test_dx_mismatches']==0
