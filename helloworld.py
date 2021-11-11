import taichi as ti

ti.init()

# Primitive types

@ti.kernel
def say_hello():
    for i in range(100):
        print("hello", i)

say_hello()

