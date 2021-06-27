print("Going to run 'import pyhf'")
import pyhf

print(pyhf)

pyhf.set_backend("jax")
a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
print(pyhf.tensorlib.clip(a, -1, 1))

import pyhf

pyhf.set_backend("jax")
a = pyhf.tensorlib.astensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(pyhf.tensorlib.erf(a))

import pyhf

pyhf.set_backend("jax")
a = pyhf.tensorlib.astensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(pyhf.tensorlib.erfinv(pyhf.tensorlib.erf(a)))

import pyhf

pyhf.set_backend("jax")
a = pyhf.tensorlib.astensor([[1.0], [2.0]])
print(pyhf.tensorlib.tile(a, (1, 2)))

import pyhf

pyhf.set_backend("jax")
tensorlib = pyhf.tensorlib
a = tensorlib.astensor([4])
b = tensorlib.astensor([5])
print(tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b))

import pyhf

pyhf.set_backend("jax")
tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor
print(type(tensor))

import pyhf

pyhf.set_backend("jax")
print(
    pyhf.tensorlib.simple_broadcast(
        pyhf.tensorlib.astensor([1]),
        pyhf.tensorlib.astensor([2, 3, 4]),
        pyhf.tensorlib.astensor([5, 6, 7]),
    )
)

import pyhf

pyhf.set_backend("jax")
tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(pyhf.tensorlib.ravel(tensor))

import pyhf

pyhf.set_backend("jax")
pyhf.tensorlib.poisson(5.0, 6.0)
values = pyhf.tensorlib.astensor([5.0, 9.0])
rates = pyhf.tensorlib.astensor([6.0, 8.0])
print(pyhf.tensorlib.poisson(values, rates))

import pyhf

pyhf.set_backend("jax")
pyhf.tensorlib.normal(0.5, 0.0, 1.0)
values = pyhf.tensorlib.astensor([0.5, 2.0])
means = pyhf.tensorlib.astensor([0.0, 2.3])
sigmas = pyhf.tensorlib.astensor([1.0, 0.8])
print(pyhf.tensorlib.normal(values, means, sigmas))

import pyhf

pyhf.set_backend("jax")
pyhf.tensorlib.normal_cdf(0.8)
values = pyhf.tensorlib.astensor([0.8, 2.0])
print(pyhf.tensorlib.normal_cdf(values))

import pyhf

pyhf.set_backend("jax")
rates = pyhf.tensorlib.astensor([5, 8])
values = pyhf.tensorlib.astensor([4, 9])
poissons = pyhf.tensorlib.poisson_dist(rates)
print(poissons.log_prob(values))

import pyhf

pyhf.set_backend("jax")
means = pyhf.tensorlib.astensor([5, 8])
stds = pyhf.tensorlib.astensor([1, 0.5])
values = pyhf.tensorlib.astensor([4, 9])
normals = pyhf.tensorlib.normal_dist(means, stds)
print(normals.log_prob(values))

import pyhf

pyhf.set_backend("jax")
tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor
numpy_ndarray = pyhf.tensorlib.to_numpy(tensor)
numpy_ndarray
print(type(numpy_ndarray))
