import glob
from pathlib import Path
import os
import fitz
import json

from hierarchy import mount_hierarchy
from hierarchy import post_process_hierarchy
from hierarchy import hierarchy_text

import time

JSON_PATH = Path('json')
PDF_PATH = Path('pdf')
os.makedirs(JSON_PATH, exist_ok=True)

t0 = time.time()
for fname in glob.glob(str(PDF_PATH/'*.pdf')):
	if os.path.isfile(fname):
		# continue
		doc = fitz.open(fname)

		pdf = doc.name.split('/')[-1]
		h = mount_hierarchy(doc)
		post_process_hierarchy(h)
		txts = hierarchy_text(h)
		del txts['SEÇÃO 0']
		json.dump(txts,
			open( JSON_PATH/(pdf+'.json'), 'w' )
			, ensure_ascii=False, indent=4 * ' ')

print("time spent:", time.time() - t0)
