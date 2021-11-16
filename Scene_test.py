import taichi as ti
import unittest
import Scene

ti.init()

@ti.data_oriented
class TeseSene(unittest.TestCase):
    @ti.kernel
    def test_sphere(self):
        ray = Scene.Ray(ti.Vector([1.0, 0.5, 0.5]), ti.Vector([-1.0, 0.0, 0.0]))

        ball = Scene.Sphere(
            ti.Vector([0.5, 0.5, 0.5]),
            0.1,
            Scene.Material.specular,
            ti.Vector([1.0, 0.0, 1.0]),
        )

        out = ball.hit(ray)

        self.assertEqual(out[0], 1)        

        print(out)
    
    @ti.kernel
    def test_plane(self):
        ray = Scene.Ray(ti.Vector([1.000000, 0.500000, 0.500000]), ti.Vector([-0.815317, 0.276975, -0.508472]))

        wall = Scene.Plane(
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0]),
            Scene.Material.specular,
            ti.Vector([1.0, 0.0, 1.0])
        )

        out = wall.hit(ray)

        self.assertEqual(out[0], 1)        

        print(out)


