import pyhf

obj = {'a': 2.0, 'b': 3.0, 'c': 1.0}
pyhf.utils.digest(obj)
print()
print(pyhf.utils.digest(obj))

pyhf.utils.digest(obj, algorithm='md5')
print()
print(pyhf.utils.digest(obj, algorithm='md5'))

pyhf.utils.digest(obj, algorithm='sha1')
print()
print(pyhf.utils.digest(obj, algorithm='sha1'))

pyhf.utils.citation(oneline=True)
print()
print(pyhf.utils.citation(oneline=True))
