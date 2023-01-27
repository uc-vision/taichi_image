import taichi as ti
import taichi.math as tm


img2d = ti.types.ndarray(ndim=2) # Color image type


@ti.func 
def index_clamped(src:img2d, idx:ti.ivec2):
    return src[tm.clamp(idx, 0, ti.ivec2(src.shape) - 1)]

@ti.func 
def sample_bilinear(src:img2d, t:ti.f32):
    p = t * ti.cast(tm.ivec2(src.shape), ti.f32)
    p1 = ti.cast(p, ti.i32)

    frac = p - ti.cast(p1, ti.f32)
    y1 = tm.mix(index_clamped(src, p1), index_clamped(src, p1 + tm.ivec2(1, 0)), frac.x)
    y2 = tm.mix(index_clamped(src, p1 + tm.ivec2(0, 1)), index_clamped(src, p1 + tm.ivec2(1, 1)), frac.x)
    return tm.mix(y1, y2, frac.y)


    

@ti.kernel
def bilinear(src: img2d, dst: img2d):
    scale = ti.cast(src.shape, ti.f32) / ti.cast(dst.shape, ti.f32) 

    for I in ti.grouped(dst):
        p = ti.cast(I, ti.f32) * scale
        dst[I] = sample_bilinear(src, p)
        

