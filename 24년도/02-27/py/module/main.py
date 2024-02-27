import test
from pathlib import Path
FILE = Path(__file__).resolve()
path = str(FILE.parents[1]) + '\\txt\\'
# ROOT = FILE.parents[0]
# ROOT1 = FILE.parents[1]
# print(FILE)
# print(ROOT)
# print(ROOT1)

# path='C:/Users/bluecom003/Downloads/정리폴더/24년도/02-27/py/txt/'
file='val.txt'
with open(path+file, 'r') as f:
    content = f.readlines()

test.prn(content)