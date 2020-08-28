import core
import glob
from pathlib import Path
import os
import time

JSON_PATH = Path('json')
TXT_PATH = Path('pdf/results/txt')
CSV_PATH = Path('csv_better')


t = 0
lis = []

empty_count = 0
not_empty = 0
for extractor in core.extractors:
	t0 = time.time()

	folder_name = str(extractor)
	folder_name = folder_name.strip("<>'")
	folder_name = folder_name.split('.')[-1]
	os.makedirs(CSV_PATH/folder_name, exist_ok=True)
	for txt in glob.glob(str(TXT_PATH/'*.txt')):
		ext = extractor(txt)
		ext.data_frame['texto'] = ext.acts_str
		txt = txt.split('/')[-1][:-3]
		if ext.data_frame.empty:			
			print(f"\tEMPTY {folder_name} for {txt}")
			empty_count += 1
		else:
			ext.data_frame.to_csv(CSV_PATH/folder_name/(txt + 'csv'), index=False)
			not_empty += 1
		
	t1 = time.time()
	lis.append((folder_name, t1 - t0))

print("time spent:")
for name, t in lis:
	print('\t', name, ':', t)
print('total:', sum(i[1] for i in lis))

print('EMPTY:', empty_count)
print('NOT_EMPTY:', not_empty)