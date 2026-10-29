package com.gpuworldgen.verify;

import java.util.Random;

/**
 * 经典 2D Perlin 噪声，Java double 实现。
 * 与 GPU kernel（noise.cl）中的 double-double 实现使用完全相同的算法，
 * 用于真值对照验证。
 */
public final class CpuNoise2D {

    private final int[] perm = new int[512];

    public CpuNoise2D(long seed) {
        Random rnd = new Random(seed);
        int[] p = new int[256];
        for (int i = 0; i < 256; i++) p[i] = i;
        // Fisher-Yates shuffle，与 kernel 侧的 CPU 生成保持一致
        for (int i = 255; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = p[i]; p[i] = p[j]; p[j] = t;
        }
        for (int i = 0; i < 512; i++) perm[i] = p[i & 255];
    }

    public int permAt(int i) {
        return perm[i];
    }

    private static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    private static double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }

    private static double grad(int hash, double x, double y) {
        int h = hash & 7;
        double u = h < 4 ? x : y;
        double v = h < 4 ? y : x;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

    /**
     * 在频率 freq 网格上采样：输入坐标已乘 freq。
     */
    public double noise(double x, double y, int freq) {
        double sx = x * freq;
        double sy = y * freq;
        int xi = (int) Math.floor(sx);
        int yi = (int) Math.floor(sy);
        double xf = sx - xi;
        double yf = sy - yi;

        double u = fade(xf);
        double v = fade(yf);

        int X = xi & 255;
        int Y = yi & 255;
        int h00 = perm[perm[X] + Y];
        int h10 = perm[perm[X + 1] + Y];
        int h01 = perm[perm[X] + Y + 1];
        int h11 = perm[perm[X + 1] + Y + 1];

        double x1 = lerp(u, grad(h00, xf, yf), grad(h10, xf - 1, yf));
        double x2 = lerp(u, grad(h01, xf, yf - 1), grad(h11, xf - 1, yf - 1));
        return lerp(v, x1, x2);
    }
}
