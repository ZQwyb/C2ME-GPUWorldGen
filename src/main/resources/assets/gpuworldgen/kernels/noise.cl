// GPU WorldGen — double-double 精度 2D Perlin 噪声 kernel
// 目标设备：Adreno（无 FP64 硬件，用两个 FP32 模拟 ~42 位精度）
//
// 精度设计：
//   double-double (hi, lo) 由两个 IEEE-754 FP32 组成，
//   有效精度约 2^(-42)（相对），远高于 FP32 的 2^(-24)，
//   可覆盖 MC 世界生成中 3000 万坐标级别的噪声采样需求。
//
// 与 Java 侧 CpuNoise2D 算法完全一致，用于精度对照验证。

typedef struct { float hi; float lo; } dd;

// ---- 基础精确运算（Dekker / Knuth / Shewchuk）----

// 精确求和：s = a + b（FP32 全精度捕获）
void two_sum(float a, float b, dd *out) {
    float s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    out->hi = s;
    out->lo = err;
}

// 精确乘积：p = a * b（借助 fma，捕获完整舍入误差）
void two_prod(float a, float b, dd *out) {
    float p = a * b;
    float err = fma(a, b, -p);
    out->hi = p;
    out->lo = err;
}

// dd + dd
dd dd_add(dd a, dd b) {
    dd s;
    two_sum(a.hi, b.hi, &s);
    float e = a.lo + b.lo + s.lo;
    dd r;
    two_sum(s.hi, e, &r);
    return r;
}

// dd - dd
dd dd_sub(dd a, dd b) {
    dd nb = b;
    nb.hi = -nb.hi;
    nb.lo = -nb.lo;
    return dd_add(a, nb);
}

// dd * dd
dd dd_mul(dd a, dd b) {
    dd p;
    two_prod(a.hi, b.hi, &p);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    dd r;
    two_sum(p.hi, p.lo, &r);
    return r;
}

// dd * 标量（标量先精确拆成 hi/lo）
dd dd_mul_s(dd a, float s) {
    dd b;
    two_prod(a.hi, s, &b);
    b.lo += a.lo * s;
    dd r;
    two_sum(b.hi, b.lo, &r);
    return r;
}

// double 转 dd（CPU 侧传入时已拆好，这里只做组装辅助）
dd mk_dd(float hi, float lo) {
    dd r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

// dd 取负
dd dd_neg(dd a) {
    dd r;
    r.hi = -a.hi;
    r.lo = -a.lo;
    return r;
}

// 比较 dd 是否 < 0（用于渐变选择，只比较 hi 足够）
int dd_lt_zero(dd a) { return a.hi < 0.0f; }

// ---- 噪声分量 ----

float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

dd dd_fade(dd t) {
    // t^3 * (t*(t*6-15)+10)，全部用 dd
    dd t6 = dd_mul_s(t, 6.0f);
    dd c = dd_add(t6, mk_dd(-15.0f, 0.0f));
    dd tt = dd_mul(t, c);
    dd p = dd_add(tt, mk_dd(10.0f, 0.0f));
    dd t2 = dd_mul(t, t);
    dd t3 = dd_mul(t2, t);
    return dd_mul(t3, p);
}

dd dd_lerp(dd t, dd a, dd b) {
    // a + t*(b-a)
    dd d = dd_sub(b, a);
    dd td = dd_mul(t, d);
    return dd_add(a, td);
}

dd dd_grad(int hash, dd x, dd y) {
    int h = hash & 7;
    dd u = (h < 4) ? x : y;
    dd v = (h < 4) ? y : x;
    dd gu = ((h & 1) == 0) ? u : dd_neg(u);
    dd gv = ((h & 2) == 0) ? v : dd_neg(v);
    return dd_add(gu, gv);
}

// ---- 2D Perlin（与 Java CpuNoise2D.noise 完全对应）----
// in: 每点 4 个 float = x_hi, x_lo, y_hi, y_lo（已放大 freq 由 CPU 完成？不：CPU 传原始坐标，freq 在 kernel 内乘）
// 为与 Java 侧精确对应，freq 乘法在 kernel 内用 dd 完成，Java 侧用 double 完成（相同值）。

__kernel void perlin2d_dd(
    __global const int *perm,     // 512 项排列表
    __global const float *in,     // count*4: x_hi x_lo y_hi y_lo（单位域 [0,1)）
    __global float *out,          // count*2: result_hi result_lo
    int freq,
    int count
) {
    int i = get_global_id(0);
    if (i >= count) return;

    dd x = mk_dd(in[i * 4 + 0], in[i * 4 + 1]);
    dd y = mk_dd(in[i * 4 + 2], in[i * 4 + 3]);

    // sx = x * freq, sy = y * freq
    dd sx = dd_mul_s(x, (float)freq);
    dd sy = dd_mul_s(y, (float)freq);

    // 整数部分与小数部分（dd 域内分解）
    // sx.hi 直接取整，误差由 lo 兜住，符合 double 的 floor 语义近似
    int xi = (int)floor(sx.hi);
    int yi = (int)floor(sy.hi);
    dd xf = dd_sub(sx, mk_dd((float)xi, 0.0f));
    dd yf = dd_sub(sy, mk_dd((float)yi, 0.0f));

    // fade（用 float 版 fade 就够？不行：fade 误差会传播。用 dd_fade 保持高精度）
    dd u = dd_fade(xf);
    dd v = dd_fade(yf);

    int X = xi & 255;
    int Y = yi & 255;
    int h00 = perm[perm[X] + Y];
    int h10 = perm[perm[X + 1] + Y];
    int h01 = perm[perm[X] + Y + 1];
    int h11 = perm[perm[X + 1] + Y + 1];

    // xf-1, yf-1
    dd xm = dd_sub(xf, mk_dd(1.0f, 0.0f));
    dd ym = dd_sub(yf, mk_dd(1.0f, 0.0f));

    dd x1 = dd_lerp(u, dd_grad(h00, xf, yf), dd_grad(h10, xm, yf));
    dd x2 = dd_lerp(u, dd_grad(h01, xf, ym), dd_grad(h11, xm, ym));
    dd res = dd_lerp(v, x1, x2);

    out[i * 2 + 0] = res.hi;
    out[i * 2 + 1] = res.lo;
}
