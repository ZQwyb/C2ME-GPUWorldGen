package com.gpuworldgen.shadow;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 真实 PerlinNoise 实例数据的共享持有器 + 影子比对采样坐标收集 + v0.10.0 GPU 接管柱缓存。
 *
 * v0.8.0 影子模式第 1 步：mixin 从真实 MC PerlinNoise 实例提取 perms/offsetX/Y/Z，
 * 并按节流收集真实世界生成坐标。渲染线程 (GlComputeRunner.onTick) 周期消费这些
 * 坐标做 GPU 影子比对（GPU 结果只记录丢弃，绝不改变地形）。
 *
 * v0.10.0 GPU 接管：mixin 对"服务实例"（首个被捕获的 3D 实例）的 get(DDD)F 调用
 * 改查 GPU 柱缓存：命中直接返回 GPU 值（不执行 CPU 方法）；miss 走原方法（CPU fallback）
 * 并把所在区块加入 pending 队列。渲染线程每 tick 从 pending 取一个区块做 GPU 批量
 * 预生成（mode21 柱采样），抽检比对 CPU 参考通过后写入 columns 缓存。
 *
 * 线程安全：columns/pending 用 ConcurrentHashMap（写=渲染线程，读=mixin 多线程）；
 * 统计用 AtomicLong；supplyRealData 由 mixin 首次调用（同步锁）。
 */
public final class ShadowShared {

    /** 真实实例数据是否已就绪 */
    public static volatile boolean ready = false;
    /** v0.10.0: 接管已激活（mixin 已确认服务实例并开始提供 GPU 值）。渲染线程据此切换预生成。 */
    public static volatile boolean takeoverActive = false;
    /** MC 真实实例的 256 字节排列表（volatile 保证跨线程可见；这是拷贝副本） */
    public static volatile byte[] realPerms = new byte[256];
    /** v0.10.0: 被捕获实例的原 perms 数组引用（mixin 用引用比较 O(1) 识别服务实例） */
    public static volatile byte[] realPermsRef;
    public static volatile double realOx, realOy, realOz;

    // ---------- v0.10.0 GPU 接管：柱缓存 + 预生成队列 + 统计 ----------
    /** overworld 柱高：y ∈ [-64, 320)，共 384 层（索引 = y - MIN_Y） */
    public static final int MIN_Y = -64;
    public static final int Y_LAYERS = 384;
    /** 柱缓存：colKey=(blockX<<32)|blockZ → float[384]（y 层噪声值，GPU 预生成、抽检通过后写入） */
    public static final ConcurrentHashMap<Long, float[]> columns = new ConcurrentHashMap<>();
    /** 待 GPU 预生成区块集合（有界，超限清空 → miss 全走 CPU fallback） */
    public static final ConcurrentHashMap<Long, Boolean> pending = new ConcurrentHashMap<>();
    public static final int PENDING_MAX = 256;

    public static final AtomicLong gpuHits = new AtomicLong();     // mixin 命中 GPU 缓存返回
    public static final AtomicLong gpuMisses = new AtomicLong();   // mixin miss → CPU fallback
    public static final AtomicLong columnsPregen = new AtomicLong(); // 已预生成区块数
    public static final AtomicLong columnsFailed = new AtomicLong(); // 抽检失败柱数

    // 坐标环形队列（有界，覆盖式）
    private static final int CAP = 256;
    private static final double[] xs = new double[CAP];
    private static final double[] zs = new double[CAP];
    private static int head = 0;
    private static int count = 0;

    /** mixin 首次调用：填充真实实例数据（只填一次）。perms 必须 256 字节。 */
    public static synchronized void supplyRealData(byte[] perms, double ox, double oy, double oz) {
        if (ready) return;
        if (perms == null || perms.length != 256) return;
        realPermsRef = perms;
        System.arraycopy(perms, 0, realPerms, 0, 256);
        realOx = ox;
        realOy = oy;
        realOz = oz;
        ready = true;
        System.out.println("[GPUWorldGen] 影子模式：已捕获真实 PerlinNoise 实例 "
                + "offsetX=" + ox + " offsetY=" + oy + " offsetZ=" + oz);
    }

    // ---------- v0.10.0 接管 API ----------

    public static long colKey(int blockX, int blockZ) {
        return ((long) blockX << 32) | (blockZ & 0xFFFFFFFFL);
    }

    /** mixin 查询：命中返回 GPU 缓存值；未命中返回 NaN（调用方走 CPU fallback）。 */
    public static float lookupColumn(int blockX, int blockY, int blockZ) {
        int idx = blockY - MIN_Y;
        if (idx < 0 || idx >= Y_LAYERS) return Float.NaN;
        float[] col = columns.get(colKey(blockX, blockZ));
        return col == null ? Float.NaN : col[idx];
    }

    /** mixin miss 时调用：把所在区块加入 GPU 预生成队列（有界，防积压）。 */
    public static void requestColumn(int blockX, int blockZ) {
        Long ck = colKey(blockX, blockZ);
        if (columns.containsKey(ck)) return;
        pending.putIfAbsent(ck, Boolean.TRUE);
        if (pending.size() > PENDING_MAX) pending.clear(); // 生成远快于 GPU → 只服务最近区块，其余 CPU fallback
    }

    /** mixin 节流调用：收集一个采样坐标（覆盖式环形队列）。 */
    public static void offer(double x, double z) {
        if (!ready) return;
        synchronized (ShadowShared.class) {
            xs[head] = x;
            zs[head] = z;
            head = (head + 1) % CAP;
            if (count < CAP) count++;
        }
    }

    /** 渲染线程取走最多 n 个坐标（FIFO），并清空已取部分。返回 null 表示暂无数据。 */
    public static synchronized double[][] drain(int n) {
        if (count == 0) return null;
        int take = Math.min(n, count);
        double[][] out = new double[take][2];
        int idx = (head - count + CAP) % CAP;
        for (int i = 0; i < take; i++) {
            out[i][0] = xs[idx];
            out[i][1] = zs[idx];
            idx = (idx + 1) % CAP;
        }
        count -= take;
        return out;
    }

    private ShadowShared() {}
}
