import jax
from time import time


def jittable_test(fun, input_gen, num_iter=20, _print=False):
    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, num_iter)

    inp_ref = input_gen(key)
    top = time()
    fun(*inp_ref)
    duration_ref = time() - top
    if _print:
        print("ref : ", duration_ref, "second")

    for i in range(num_iter):
        inp = input_gen(subkeys[i])
        top = time()
        fun(*inp)
        duration = time() - top
        if _print:
            print("test : ", duration, "second")
        else:
            assert duration < duration_ref / 10
