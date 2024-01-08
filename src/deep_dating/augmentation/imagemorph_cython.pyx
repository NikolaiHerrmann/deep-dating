import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef compute_fields(double[:,:] da_x, double[:,:] da_y, 
                     double[:,:] d_x, double[:,:] d_y, int h, int w, double[:] ker, int kws):
    cdef int i, j, k, v, u
    cdef double sum_x, sum_y

    for i in range(h):
        for j in range(w):
            sum_x = 0.0
            sum_y = 0.0
            for k in range(-kws, kws + 1):
                v = j + k
                if v < 0:
                    v = -v
                if v >= w:
                    v = 2 * w - v - 1
                sum_x += d_x[i][v] * ker[abs(k)]
                sum_y += d_y[i][v] * ker[abs(k)]
            da_x[i][j] = sum_x
            da_y[i][j] = sum_y

    for j in range(w):
        for i in range(h):
            sum_x = 0.0
            sum_y = 0.0
            for k in range(-kws, kws + 1):
                u = i + k
                if u < 0:
                    u = -u
                if u >= h:
                    u = 2 * h - u - 1
                sum_x += da_x[u][j] * ker[abs(k)]
                sum_y += da_y[u][j] * ker[abs(k)]
            d_x[i][j] = sum_x
            d_y[i][j] = sum_y

    return d_x, d_y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef displacement_field(unsigned char[:,:,:] output, unsigned char[:,:,:] input, double[:,:] d_x, double[:,:] d_y, int h, int w):
    cdef int i, j, idx, u, v, u0, v0, 
    cdef double p1, p2, f1, f2,sumr, sumg, sumb, val

    for i in range(h):
        for j in range(w):
            p1 = i + d_y[i][j]
            p2 = j + d_x[i][j]

            u0 = int(np.floor(p1))
            v0 = int(np.floor(p2))

            f1 = p1 - u0
            f2 = p2 - v0

            sumr, sumg, sumb = 0.0, 0.0, 0.0
            for idx in range(4):
                if idx == 0:
                    u, v = u0, v0
                    f = (1.0 - f1) * (1.0 - f2)
                elif idx == 1:
                    u, v = u0 + 1, v0
                    f = f1 * (1.0 - f2)
                elif idx == 2:
                    u, v = u0, v0 + 1
                    f = (1.0 - f1) * f2
                else:
                    u, v = u0 + 1, v0 + 1
                    f = f1 * f2

                u = max(0, min(u, h - 1))
                v = max(0, min(v, w - 1))

                val = input[u][v][0]
                sumr += f * val

                val = input[u][v][1]
                sumg += f * val

                val = input[u][v][2]
                sumb += f * val

            output[i][j][0] = int(sumr)
            output[i][j][1] = int(sumg)
            output[i][j][2] = int(sumb)

    return output