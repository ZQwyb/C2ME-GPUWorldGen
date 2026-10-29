package com.gpuworldgen.verify;

import com.gpuworldgen.config.ModConfig;
import com.gpuworldgen.gl.GlComputeRunner;
import com.gpuworldgen.opencl.DlopenProbe;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * v0.6.7 多轮稳定性验证：
 * 1. OpenCL 路径探测（结论性诊断：Android linker namespace 拦截）
 * 2. CPU double 参考值（标准组 + 大坐标组）
 * 3. GlComputeRunner 在一次启动内连跑 5 轮，每轮重新编译 shader + 跑两组噪声
 * 4. 逐轮统计：编译尝试次数、mode=9 整数链不一致点、两组最大误差、轮耗时
 * 5. 汇总裁决 STABLE / UNSTABLE
 *
 * 目的：MobileGlues GLSL 编译器非确定性（此前成功率约 50%），单次 PASS 不足，
 * 必须证明编译自检能否稳定兜住、以及编译成功后精度是否可复现。
 *
 * 结果写入 <游戏目录>/gpu-worldgen-verify.txt，同时输出到日志。
 */
public final class VerifyRunner {

    public static final String REPORT_FILE = "gpu-worldgen-verify.txt";

    private final ModConfig config;
    private final StringBuilder report = new StringBuilder();

    public VerifyRunner(ModConfig config) {
        this.config = config;
    }

    /** 入口：任何异常都被捕获并写入报告，绝不影响游戏启动 */
    public void run() {
        long t0 = System.nanoTime();
        try {
            runInternal();
        } catch (Throwable t) {
            report.append("\n========== 运行失败 ==========\n");
            report.append(t.toString()).append('\n');
            for (StackTraceElement e : t.getStackTrace()) {
                report.append("    at ").append(e).append('\n');
            }
            report.append("\n请把本报告文件内容发给作者。\n");
        } finally {
            long t1 = System.nanoTime();
            report.append("\n总耗时: ").append(String.format("%.3f", (t1 - t0) / 1e9)).append(" s\n");
            finishReport();
        }
    }

    private void runInternal() throws Exception {
        line("GPU WorldGen 手机适配 — v0.10.1 多轮稳定性验证报告（验证在主菜单后跑，加载画面不污染）");
        line("==================================================");
        line("MC 版本目标: 26.3 (Fabric, Java 25)");
        line("采样配置: seed=" + config.seed + " size=" + config.size + "x" + config.size
                + " freq=" + config.freq + " 大坐标偏移=100000");
        line("路线: FFM 直连 libmobileglues.so，fragment shader + FBO 输出（无输入纹理）");
        line("");

        // ---------- 1. 探测 OpenCL 库候选路径（结论性诊断） ----------
        line("[1] 探测 OpenCL 库路径 ...");
        String[] openclCandidates = {
            "OpenCL",
            "/vendor/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/odm/lib64/libOpenCL.so",
            "/product/lib64/libOpenCL.so",
        };
        String workingOpenClPath = null;
        try {
            DlopenProbe probe = new DlopenProbe();
            for (DlopenProbe.Result r : probe.probe(openclCandidates)) {
                line("    " + (r.ok() ? "[OK]  " : "[FAIL]") + " " + r.path() + "  → " + r.dlerror());
                if (r.ok() && workingOpenClPath == null) {
                    workingOpenClPath = r.path();
                }
            }
        } catch (Throwable t) {
            line("    探测器本身异常: " + t);
        }
        if (workingOpenClPath == null) {
            line("结论: OpenCL 全部路径不可用（Android linker namespace 拦截，需 root 才能绕过）。");
        }

        // ---------- 2. CPU 参考值 + 构造输入（两组：标准 [0,1] 与 大坐标偏移 100000） ----------
        int n = config.size * config.size;
        CpuNoise2D cpu = new CpuNoise2D(config.seed);
        double bigOffset = 100000.0;
        // v0.9.0: [4.5] MC 对拍用超大坐标（>2e6 且带小数）——验证 GPU dd 坐标链在真实大坐标
        // 不再丢小数（float 单值在 5e6 处 ULP≈0.5，旧实现必然失准）。与 [4] 的 bigOffset=100000 独立。
        double mcBigOffset = 5000123.456789;

        int[] perm = new int[512];
        for (int i = 0; i < 512; i++) perm[i] = cpu.permAt(i);

        float[] input = new float[n * 4];
        double[] cpuValues = new double[n];
        double[] cpuValuesBig = new double[n];
        for (int row = 0; row < config.size; row++) {
            for (int col = 0; col < config.size; col++) {
                int idx = row * config.size + col;
                double x = (col + 0.5) / config.size;
                double y = (row + 0.5) / config.size;
                float xh = (float) x;
                float xl = (float) (x - xh);
                float yh = (float) y;
                float yl = (float) (y - yh);
                int base = idx * 4;
                input[base] = xh;
                input[base + 1] = xl;
                input[base + 2] = yh;
                input[base + 3] = yl;
                cpuValues[idx] = cpu.noise(x, y, config.freq);
                cpuValuesBig[idx] = cpu.noise(x + bigOffset, y + bigOffset, config.freq);
            }
        }
        line("");
        line("[2] CPU double 参考值已就绪（" + n + " 点 × 2 组）。");

        // ---------- 3. 提交 GL 多轮稳定性验证（渲染线程 tick 执行，内部跑 5 轮） ----------
        line("");
        line("[3] 提交 GL 多轮稳定性验证（每轮重新编译 shader；等待渲染线程 tick）...");
        long gpuT0 = System.nanoTime();
        GlComputeRunner gpu = new GlComputeRunner(report);

        // v0.7.0: MC 精确噪声对拍参数（必须在 runDiagnostics 前设置，onTick 第 1 轮会跑 mode 20）
        // offsets 转 float 再回 double，与 GPU float uniform 完全一致 → 纯算法比对（不受 offsets 精度干扰）
        double mcOx = (double) (float) 187.10481682004246;
        double mcOy = (double) (float) 104.98068774200362;
        double mcOz = (double) (float) 53.17499937528757;
        byte[] mcPerm = new byte[256];
        for (int i = 0; i < 256; i++) mcPerm[i] = (byte) (cpu.permAt(i) & 0xFF);
        CpuMcPerlin mcCpu = new CpuMcPerlin(mcPerm, mcOx, mcOy, mcOz);
        double[] mcRef = new double[n];
        for (int row = 0; row < config.size; row++) {
            for (int col = 0; col < config.size; col++) {
                int idx = row * config.size + col;
                mcRef[idx] = mcCpu.get(mcBigOffset + (col + 0.5) / config.size, mcBigOffset + (row + 0.5) / config.size);
            }
        }
        gpu.requestMcCheck(config.freq, mcBigOffset, mcBigOffset, (float) mcOx, (float) mcOy, (float) mcOz);

        float[][] diag = gpu.runDiagnostics(perm, input, n, config.freq);
        long gpuT1 = System.nanoTime();

        int R = gpu.stabilityRounds();
        int[] attempts = gpu.roundAttempts();
        float[][] rStd = gpu.roundStdNoise();
        float[][] rBig = gpu.roundBigNoise();
        float[][] rM9 = gpu.roundBigMode9();
        long[] rNanos = gpu.roundNanos();

        if (attempts == null) {
            line("    [GL] 执行失败: " + gpu.error());
            line("    结论: GL 通道在本机不可用，需要作者根据日志调整。");
            return;
        }
        line("    GPU 总耗时（含等待渲染线程 tick）: " + String.format("%.3f", (gpuT1 - gpuT0) / 1e9) + " s");

        // ---------- 4.5 (v0.7.0) MC 精确噪声 GPU 复刻对拍（影子模式第 0 步） ----------
        // 参数已在 runDiagnostics 前设置（onTick 第 1 轮跑 mode 20）。GPU 值不被采用，仅测"复刻准不准"。
        line("");
        line("[4.5] MC 精确噪声（PerlinNoise 3D 16 梯度）GPU 复刻对拍：");
        float[] mcGpu = gpu.mcResult();
        if (mcGpu == null) {
            line("    MC 对拍未执行（mode 20 编译/运行失败），跳过。");
        } else {
            double mcMaxErr = 0;
            int mcOverThresh = 0;
            for (int i = 0; i < n; i++) {
                double g = (double) mcGpu[i * 4] + (double) mcGpu[i * 4 + 1];
                double err = Math.abs(mcRef[i] - g);
                if (err > mcMaxErr) mcMaxErr = err;
                if (err > 1e-6) mcOverThresh++;
            }
            line("    CPU(MC 算法) 真值就绪，GPU mode 20 复刻比对：");
            line("    最大误差: " + String.format("%.3e", mcMaxErr) + "；超过 1e-6 的点数: " + mcOverThresh + " / " + n);
            line("    " + (mcMaxErr < 1e-6 ? "结论: GPU 复刻 MC 噪声在真实大坐标上正确（可进入接管步骤）"
                                         : "结论: GPU 复刻与 CPU 有偏差，需排查 shader 算法或 double-double 边界"));
        }

        // ---------- 4. 逐轮结果 ----------
        line("");
        line("[4] 多轮稳定性逐轮结果：");
        line("    轮次  编译尝试  编译    mode9不一致  标准坐标最大误差  大坐标最大误差  轮耗时(ms)");
        int compileOk = 0, firstShotOk = 0, precisionOk = 0;
        int totalBadM9 = 0;
        double worstStd = 0, worstBig = 0, bestBig = 1.0;
        long sumRoundNanos = 0;
        for (int r = 0; r < R; r++) {
            boolean compOk = attempts[r] > 0;
            int att = Math.abs(attempts[r]);
            if (compOk) {
                compileOk++;
                if (att == 1) firstShotOk++;
            }
            int badM9 = -1;
            double stdErr = -1, bigErr = -1;
            if (compOk && rM9[r] != null) {
                badM9 = countMode9Mismatch(rM9[r], config.size, config.freq, bigOffset);
                totalBadM9 += badM9;
            }
            if (compOk && rStd[r] != null) {
                stdErr = maxNoiseError(cpuValues, rStd[r], n);
                worstStd = Math.max(worstStd, stdErr);
            }
            if (compOk && rBig[r] != null) {
                bigErr = maxNoiseError(cpuValuesBig, rBig[r], n);
                worstBig = Math.max(worstBig, bigErr);
                bestBig = Math.min(bestBig, bigErr);
                if (bigErr < 1e-6) precisionOk++;
            }
            sumRoundNanos += rNanos[r];
            line(String.format("    %-4d  %-7d  %-6s  %-11s  %-15s  %-14s  %.0f",
                    r + 1,
                    att,
                    compOk ? "通过" : "失败",
                    badM9 < 0 ? "-" : String.valueOf(badM9),
                    stdErr < 0 ? "-" : String.format("%.3e", stdErr),
                    bigErr < 0 ? "-" : String.format("%.3e", bigErr),
                    rNanos[r] / 1e6));
        }

        // ---------- 5. 汇总与结论 ----------
        line("");
        line("[5] 稳定性汇总：");
        line("    编译：" + compileOk + "/" + R + " 轮成功，其中首轮自检通过 " + firstShotOk + "/" + R
                + "（编译自检+重试对非确定性编译器的兜底情况）");
        line("    精度：" + precisionOk + "/" + R + " 轮大坐标误差 < 1e-6；"
                + "大坐标最大误差区间 [" + String.format("%.3e", bestBig) + ", " + String.format("%.3e", worstBig) + "]");
        line("    mode9 累计不一致点：" + totalBadM9 + "；标准坐标最大误差：" + String.format("%.3e", worstStd));
        line("    耗时：平均 " + String.format("%.0f", (double) sumRoundNanos / 1e6 / R) + " ms/轮");
        line("");
        boolean stable = compileOk == R && precisionOk == R && totalBadM9 == 0;
        if (stable) {
            line("结论: STABLE（稳定）。" + R + " 轮编译全部成功、大坐标精度稳定复现 < 1e-6，");
            line("double-double FBO 管线在 Adreno 830 上可重复工作。可以进入接入世界生成阶段。");
        } else {
            line("结论: UNSTABLE（存在波动）。编译或精度在多轮中未全部通过，需要先解决稳定性。");
            line("若仅个别轮次编译失败但精度全对，可考虑增加编译重试次数；若精度波动则需排查。");
        }
        line("");
        line("========== 验证完成 ==========");
    }

    /** 统计大坐标组 mode=9 整数链（xi/yi/X/Y）与 CPU 期望的不一致点数 */
    private static int countMode9Mismatch(float[] m9, int size, int freq, double bigOffset) {
        int bad = 0;
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                int idx = row * size + col;
                int base = idx * 4;
                double x = (col + 0.5) / size + bigOffset;
                double y = (row + 0.5) / size + bigOffset;
                double sx = x * freq, sy = y * freq;
                int xi = (int) Math.floor(sx), yi = (int) Math.floor(sy);
                int X = xi & 255, Y = yi & 255;
                if (m9[base] != xi || m9[base + 1] != yi
                        || m9[base + 2] != X || m9[base + 3] != Y) bad++;
            }
        }
        return bad;
    }

    /** 计算 GPU 输出（hi+lo）相对 CPU 参考的最大绝对误差 */
    private static double maxNoiseError(double[] ref, float[] out, int n) {
        double maxErr = 0;
        for (int i = 0; i < n; i++) {
            double g = (double) out[i * 4] + (double) out[i * 4 + 1];
            double err = Math.abs(ref[i] - g);
            if (err > maxErr) maxErr = err;
        }
        return maxErr;
    }

    private void finishReport() {
        if (config.writeReport) {
            try {
                Path gameDir = net.fabricmc.loader.api.FabricLoader.getInstance().getGameDir();
                Path file = gameDir.resolve(REPORT_FILE);
                Files.writeString(file, report.toString(), StandardCharsets.UTF_8);
                System.out.println("[GPUWorldGen] 验证报告已写入: " + file.toAbsolutePath());
            } catch (Exception e) {
                System.err.println("[GPUWorldGen] 无法写入报告文件: " + e);
            }
        }
        System.out.println(report);
    }

    private void line(String s) {
        report.append(s).append('\n');
        System.out.println("[GPUWorldGen] " + s);
    }
}
