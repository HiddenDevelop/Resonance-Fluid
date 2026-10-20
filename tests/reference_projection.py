"""CPU reference for MAC projection equations, NOT execution of the CUDA code.
Run with Python 3; standard library only. GPU regression tests are fluid_tests.cu.
"""
import math
import unittest


def divergence(u, v, n):
    return [u[y*(n+1)+x+1]-u[y*(n+1)+x]
            + v[(y+1)*(n+1)+x]-v[y*(n+1)+x]
            for y in range(n) for x in range(n)]


def gradient(p, n):
    u, v = [0.0]*(n+1)**2, [0.0]*(n+1)**2
    for y in range(n):
        for x in range(1, n):
            u[y*(n+1)+x] = p[y*n+x]-p[y*n+x-1]
    for y in range(1, n):
        for x in range(n):
            v[y*(n+1)+x] = p[y*n+x]-p[(y-1)*n+x]
    return u, v


def neighbors(i, n):
    x, y = i % n, i // n
    return ([i-1] if x else []) + ([i+1] if x<n-1 else []) + ([i-n] if y else []) + ([i+n] if y<n-1 else [])


class ProjectionReference(unittest.TestCase):
    def test_divergence_gradient_is_neumann_laplacian(self):
        n = 7
        p = [math.sin(i*0.37) for i in range(n*n)]
        dg = divergence(*gradient(p, n), n)
        laplacian = [sum(p[j]-p[i] for j in neighbors(i, n)) for i in range(n*n)]
        self.assertLess(max(abs(a-b) for a, b in zip(dg, laplacian)), 1e-14)

    def test_projection_reduces_divergence_and_preserves_walls(self):
        n = 32
        u, v = [0.0]*(n+1)**2, [0.0]*(n+1)**2
        for y in range(n):
            for x in range(1, n):
                u[y*(n+1)+x] = 35*math.exp(-((x-16)**2+(y+0.5-16)**2)/18)
        for y in range(1, n):
            for x in range(n):
                v[y*(n+1)+x] = -18*math.exp(-((x+0.5-16)**2+(y-16)**2)/18)
        div = divergence(u, v, n)
        self.assertAlmostEqual(sum(div), 0, places=10)  # solvable Neumann RHS
        p = [0.0]*(n*n)
        adjacency = [neighbors(i, n) for i in range(n*n)]
        for _ in range(1000):
            p = [p[i]/3 + (2/3)*(sum(p[j] for j in js)-div[i])/len(js)
                 for i, js in enumerate(adjacency)]
        gu, gv = gradient(p, n)
        u = [a-b for a, b in zip(u, gu)]
        v = [a-b for a, b in zip(v, gv)]
        after = divergence(u, v, n)
        ratio = math.sqrt(sum(a*a for a in after)/sum(a*a for a in div))
        print(f"CPU reference projection residual ratio: {ratio:.6f}")
        self.assertLess(ratio, 0.1)
        self.assertTrue(all(u[y*(n+1)]==u[y*(n+1)+n]==0 for y in range(n)))
        self.assertTrue(all(v[x]==v[n*(n+1)+x]==0 for x in range(n)))

    def test_decay_independent_of_step_partition(self):
        rate = 0.08
        self.assertAlmostEqual(math.exp(-rate/120)**120, math.exp(-rate/60)**60, places=13)


if __name__ == '__main__':
    unittest.main()
