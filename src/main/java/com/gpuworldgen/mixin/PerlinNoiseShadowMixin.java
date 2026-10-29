package com.gpuworldgen.mixin;

import com.gpuworldgen.shadow.ShadowShared;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicLong;

/**
 * v0.8.0+ 影子模式：拦截真实 MC PerlinNoise 噪声查询。
 *
 * v0.10.0 GPU 接管：get(DDD)F（3D，MC 26.3 区块生成实证走这个）在首个被捕获实例
 * （服务实例，perms+offsets 匹配）上改为【真正接管】：
 *   - GPU 柱缓存命中 → cir.setReturnValue(GPU值) + cancel（跳过 CPU 原方法，噪声由 GPU 提供）
 *   - 缓存 miss → 不 cancel（走原方法 = CPU fallback），并把所在区块加入 GPU 预生成队列
 * 缓存值由渲染线程预生成时与 CPU 参考抽检比对（<1e-5）通过后才写入，未通过柱永久 fallback。
 * get(DD)F（2D）不接管，保持原逻辑。
 *
 * 关键点：perms/offsetX/Y/Z 是父类 GradientNoise 的 protected 字段。mixin 的 @Shadow
 * 默认只在目标类自身查找字段（不自动上查父类），且本 mod 无 refMap，@Shadow 会抛
 * InvalidMixinException 导致 required:true 崩溃。因此这里【不用 @Shadow】，改用反射
 * 读父类 GradientNoise 的 protected 字段（mojmap 字段名生产一致）。
 *
 * 接管安全：任何异常被吞掉 → 不 cancel → 走 CPU 原方法，绝不干扰/崩溃游戏。
 */
@Mixin(PerlinNoise.class)
public abstract class PerlinNoiseShadowMixin {

    private static final AtomicLong CALL = new AtomicLong();
    /** 每 256 次 get 调用收集一个采样坐标（2 的幂，用位运算节流） */
    private static final long STRIDE = 256;

    // 父类 GradientNoise 字段的反射句柄（懒初始化，只做一次）
    private static volatile Field F_PERMS, F_OX, F_OY, F_OZ;
    private static volatile boolean fieldsReady;
    private static volatile boolean diag2d;   // 诊断：get(DD)F 是否命中
    private static volatile boolean diag3d;   // 诊断：get(DDD)F 是否命中

    /** v0.10.0: mixin 向目标类注入的实例标志：本 PerlinNoise 实例是否为"服务实例"（GPU 接管）。 */
    private volatile boolean gpuworldgen$served;

    private static void ensureFields() {
        if (fieldsReady) return;
        try {
            Class<?> g = net.minecraft.world.level.levelgen.synth.GradientNoise.class;
            F_PERMS = g.getDeclaredField("perms");
            F_OX = g.getDeclaredField("offsetX");
            F_OY = g.getDeclaredField("offsetY");
            F_OZ = g.getDeclaredField("offsetZ");
            F_PERMS.setAccessible(true);
            F_OX.setAccessible(true);
            F_OY.setAccessible(true);
            F_OZ.setAccessible(true);
            fieldsReady = true;
            System.out.println("[GPUWorldGen][mixin] 反射字段就绪: perms/offsetX/Y/Z");
        } catch (Throwable t) {
            System.out.println("[GPUWorldGen][mixin] 反射字段失败: " + t);
        }
    }

    /** 首次被真实噪声查询调用时，提取真实实例的 perms + offsetX/Y/Z 到 ShadowShared。 */
    private static void tryExtract(Object self) {
        if (ShadowShared.ready) return;
        ensureFields();
        if (fieldsReady) {
            try {
                byte[] perms = (byte[]) F_PERMS.get(self);
                double ox = F_OX.getDouble(self);
                double oy = F_OY.getDouble(self);
                double oz = F_OZ.getDouble(self);
                ShadowShared.supplyRealData(perms, ox, oy, oz);
            } catch (Throwable t) {
                System.out.println("[GPUWorldGen][mixin] 提取真实实例失败: " + t);
            }
        }
    }

    /** v0.10.0: 本实例是否服务实例（perms 引用/内容 + offsets 全部与捕获实例一致才接管）。 */
    private boolean isServedInstance() {
        ensureFields();
        if (!fieldsReady || !ShadowShared.ready) return false;
        try {
            byte[] perms = (byte[]) F_PERMS.get((Object) this);
            if (perms == null || perms.length != 256) return false;
            if (perms == ShadowShared.realPermsRef) return true;   // 同一实例：引用相等 O(1)
            byte[] rp = ShadowShared.realPerms;
            for (int i = 0; i < 256; i++) if (perms[i] != rp[i]) return false;
            double ox = F_OX.getDouble(this), oy = F_OY.getDouble(this), oz = F_OZ.getDouble(this);
            return ox == ShadowShared.realOx && oy == ShadowShared.realOy && oz == ShadowShared.realOz;
        } catch (Throwable t) {
            return false;
        }
    }

    @Inject(method = "get(DD)F", at = @At("HEAD"))
    private void gpuworldgen$shadow(double x, double z, CallbackInfoReturnable<Float> cir) {
        try {
            if (!diag2d) {
                diag2d = true;
                System.out.println("[GPUWorldGen][mixin] 已命中 get(DD)F，首次调用 (x=" + x + " z=" + z + ")");
            }
            tryExtract(this);
            long c = CALL.incrementAndGet();
            if ((c & (STRIDE - 1)) == 0L) {
                ShadowShared.offer(x, z);
            }
        } catch (Throwable ignored) {
            // 影子模式绝不干扰游戏：任何异常直接忽略
        }
    }

    @Inject(method = "get(DDD)F", at = @At("HEAD"))
    private void gpuworldgen$takeover3(double x, double y, double z, CallbackInfoReturnable<Float> cir) {
        try {
            if (!diag3d) {
                diag3d = true;
                System.out.println("[GPUWorldGen][mixin] 已命中 get(DDD)F，首次调用 (x=" + x + " y=" + y + " z=" + z + ")");
            }
            tryExtract(this);
            if (!ShadowShared.ready) return;               // 服务实例未捕获 → 原方法（CPU）
            if (!gpuworldgen$served) {
                if (!isServedInstance()) return;           // 非服务实例 → 原方法（CPU fallback）
                gpuworldgen$served = true;
                ShadowShared.takeoverActive = true;
                System.out.println("[GPUWorldGen][mixin] 接管实例已确定（perms/offset 匹配），GPU 开始提供该实例噪声值");
            }
            int bx = (int) Math.floor(x);
            int by = (int) Math.floor(y);
            int bz = (int) Math.floor(z);
            float v = ShadowShared.lookupColumn(bx, by, bz);
            if (!Float.isNaN(v)) {
                ShadowShared.gpuHits.incrementAndGet();
                cir.setReturnValue(v);
                cir.cancel();                               // 跳过 CPU 原方法 → GPU 值真正被采用
            } else {
                ShadowShared.gpuMisses.incrementAndGet();
                ShadowShared.requestColumn(bx, bz);         // 入 GPU 预生成队列（有界）
            }
        } catch (Throwable ignored) {
            // 接管绝不干扰游戏：任何异常不 cancel → 走 CPU 原方法
        }
    }
}
