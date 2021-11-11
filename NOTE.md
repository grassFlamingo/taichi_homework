# Taichi Node

## Primitive types

- signed integers: ti.i8, ti.i16, *ti.i32*, ti.i64
- unsigned integers: ti.u8, ti.u16, ti.u32, ti.u64
- floating points: ti.f32, ti.f64

```python
ti.cast(variable, type)

v3f = ti.types.vector(3, ti.f32)
m2f = ti.types.matrix(2,2,ti.f32)
ray = ti.types.struce(ro=v3f, rd=v3f, l=ti.f32)

```

## ti.filed

```python
ti.filed(dtype=ti.f32, shape=(256,256))
```

a global N-d array of elements
- global: both `ti.kernel` and python script
- N-d 
- elements, scalar, vector, matrix, struct
- access elements `[i,j,k,...]`

## ti.kernel and ti.func

outermost scope | for :-) 
automatically parallelized

- Not support `break`
- `+=`
- at most 8 parameter
- python to taichi
- must be **type-hinted**
- scalar only
- pass by **value**
- return one single scalar only
- call kernel form python
- force-inline; not recursive
- all data is static
- static data; static score

```python
a = 42
@ti.kernel
def print_a():
    print('a =', a)

print_a() # a = 42
a = 52
print('a =',a) # a = 53
print_a() # a = 42 <- 
```

try use filed

## print and GUI

- be ware of parall
- `ti.sync()`
- ti.gui <- 2D only

```python
gui = ti.GUI('title', res=(1024,768))

gui.get_events()
gui.get_key_event()
...
```

## ti.template

- reference 
- cannot alter python scope in ti scope
- `ti.grouped(y)`

## ti.static

- forloop: unrolled

## ti.data_oriented

## Advance dense data layout

data-oriented 

packed auto expand to `2**x`
`ti.init(ti.packed=True)`

```python
x = ti.Vector.field(3, ti.f32)
ti.root.dense(ti.i, 16).place(x)
ti.root.dense(ti.i, 4).dense(ti.i, 4).place(x)
ti.root.dense(ti.i, 4).dense(ti.i, 4).place(x)
ti.root.dense(ti.i, N).place(x, y, z)
```

- `ti.i`: row major
- `ti.j`: column major
- `ti.ij`: 

structure of array 
```c
struct A {
    int a[10];
    int b[10];
};

A data;
```

array of structure
```c
struct A{
    int a;
    int b;
};

A data[10];
```

## ti.root.pointer

a sparse SNode-tree

`ti.root.pointer(ti.i, 3).dense(ti.j, 3)`:
- root:
  - pointer
    - a[0]
    - a[1]
    - a[2]
  - pointer
    - a[0]
    - ...
  - ...

`ti.root.pointer(ti.i, 3).pointer(ti.j, 3).place(x)`:
- root
  - pointer
    - pointer
    - pointer
    - ...
  - pointer
    - pointer
    - ...
  - ...


`ti.root.pointer(ti.i, 3).bitmap(ti.j, 3).place(x)`:
- root
  - pointer
    - bitmap; data
    - ...
  - ...

### APIS

- `ti.is_active(block, [0])`
- `ti.activate/deactive(snode, [i,j])`
- `snode.deactivate_all()`: deactivate cell and all its children
- `ti.rescale_index(snode/field, ancestor_snode, index)`