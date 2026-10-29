package com.gpuworldgen.verify;

/**
 * MC 精确 PerlinNoise 2D 采样（自包含 Java 实现，作为 GPU 复刻的真值）。
 *
 * 算法逐位对齐 net.minecraft.world.level.levelgen.synth.PerlinNoise 26.3，
 * 已在开发机用 MC 真 jar 对拍 20 万点 / 20 种子，maxAbsDiff = 0.0。
 *
 * 与 CpuNoise2D（自研 2D 8 梯度）不同：这是 MC 原生 3D Perlin 在 y=offsetY
 * 切面上的值，16 梯度表 + 8 角三线性插值。2D get(x,z) = get(wrap(x), 0, wrap(z))，
 * 但 y 方向格点由 offsetY（常量）决定，不能当 0。
 *
 * 大坐标精度：wrap() 只在 |x| >= ~1.677e7 时折回，正常世界坐标不触发，
 * 直接 wrap(x)+offset 用 double 处理即可（这正是 GPU double-double 的价值点）。
 */
public final class CpuMcPerlin {

    /** MC GradientNoise.GRADIENT 静态表（16 个立方体角向量），已对拍 */
    private static final int[][] GRAD = {
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},
        {1,1,0},{0,-1,1},{-1,1,0},{0,-1,-1}
    };
    private static final double ROUND = 33554432.0;
    private static final double HALF_ROUND = Math.nextDown(1.6777216E7);

    private final int[] perm = new int[512];
    private final double ox, oy, oz;

    /** @param perms MC 实例的 byte[256] 排列表；offsets 为 offsetX/Y/Z（double） */
    public CpuMcPerlin(byte[] perms, double offsetX, double offsetY, double offsetZ) {
        for (int i = 0; i < 256; i++) perm[i] = perms[i] & 0xFF;
        for (int i = 0; i < 256; i++) perm[i + 256] = perm[i];
        this.ox = offsetX; this.oy = offsetY; this.oz = offsetZ;
    }

    /** 2D 采样，等价 MC PerlinNoise.get(x, z)（内部 y=0 → y0=offsetY 切面，不能传 oy 否则 y 加两次 offset） */
    public float get(double x, double z) {
        return get3(x, 0.0, z);
    }

    /** v0.10.0: 3D 采样，等价 MC PerlinNoise.get(x, y, z)（接管预生成比对参考） */
    public float get3(double x, double y, double z) {
        double x0 = wrap(x) + ox;
        double y0 = wrap(y) + oy;
        double z0 = wrap(z) + oz;
        int xi = (int) Math.floor(x0);
        int yi = (int) Math.floor(y0);
        int zi = (int) Math.floor(z0);
        float xf = (float) (x0 - xi);
        float yf = (float) (y0 - yi);
        float zf = (float) (z0 - zi);
        return sampleAndLerp(xi, yi, zi, xf, yf, zf);
    }

    private int permute(int i) { return perm[i & 255]; }

    private static double wrap(double x) {
        if (x >= -HALF_ROUND && x < HALF_ROUND) return x;
        return x - ROUND * Math.floor(x / ROUND + 0.5);
    }

    private float gradDot(int h, float dx, float dy, float dz) {
        int[] g = GRAD[h & 15];
        return g[0] * dx + g[1] * dy + g[2] * dz;
    }

    private static float fade(float t) { return t * t * t * (t * (t * 6f - 15f) + 10f); }
    private static float lerp(float t, float a, float b) { return a + t * (b - a); }

    /** MC PerlinNoise.sampleAndLerp 精确复刻：8 角梯度 + 三线性 */
    private float sampleAndLerp(int xi, int yi, int zi, float xf, float yf, float zf) {
        int a = permute(xi), b = permute(xi + 1);
        int ab = permute(a + yi), aa = permute(a + yi + 1);
        int ba = permute(b + yi), bb = permute(b + yi + 1);
        int c000 = permute(ab + zi);        float d000 = gradDot(c000, xf, yf, zf);
        int c100 = permute(ba + zi);        float d100 = gradDot(c100, xf - 1, yf, zf);
        int c010 = permute(aa + zi);        float d010 = gradDot(c010, xf, yf - 1, zf);
        int c110 = permute(bb + zi);        float d110 = gradDot(c110, xf - 1, yf - 1, zf);
        int c001 = permute(ab + zi + 1);    float d001 = gradDot(c001, xf, yf, zf - 1);
        int c101 = permute(ba + zi + 1);    float d101 = gradDot(c101, xf - 1, yf, zf - 1);
        int c011 = permute(aa + zi + 1);    float d011 = gradDot(c011, xf, yf - 1, zf - 1);
        int c111 = permute(bb + zi + 1);    float d111 = gradDot(c111, xf - 1, yf - 1, zf - 1);
        float u = fade(xf), v = fade(yf), w = fade(zf);
        float l00 = lerp(u, d000, d100), l10 = lerp(u, d010, d110);
        float l01 = lerp(u, d001, d101), l11 = lerp(u, d011, d111);
        float y0 = lerp(v, l00, l10), y1 = lerp(v, l01, l11);
        return lerp(w, y0, y1);
    }
}
