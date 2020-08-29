import re
def get_dodf_reverse_date(s):
    date = re.search( r'((\d+)[-](\d+)[-](\d+))', s)
    date = date.group()
    date = '-'.join(date.split('-')[::-1])
    return date

def get_dodf_num(s):
    return int(re.search('(\d+)', s).group())

def get_dodf_tipo(s):
	return re.search(
		r'(\w+)[.]pdf', s
		).group(1).replace(
			'SECAO1', 'NORMAL').replace(
			'INTEGRA', 'NORMAL'
	)

def get_dodf_key(s):
	s = s.split('/')[-1]
	return (
		get_dodf_reverse_date(s),
		get_dodf_num(s),
		get_dodf_tipo(s),
	)