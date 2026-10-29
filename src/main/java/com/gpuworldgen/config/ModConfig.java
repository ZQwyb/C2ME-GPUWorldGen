package com.gpuworldgen.config;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.fabricmc.loader.api.FabricLoader;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * mod 配置文件：<游戏目录>/config/gpuworldgen.json
 * 不存在时自动创建默认配置。
 */
public final class ModConfig {

    public boolean enabled = true;

    /** 随机种子（permutation 表） */
    public long seed = 42L;

    /** 采样网格边长（size x size 个采样点） */
    public int size = 128;

    /** 噪声频率（网格缩放） */
    public int freq = 4;

    /** 工作组大小，0 = 自动（64） */
    public int workGroupSize = 64;

    /** 首选设备类型：gpu / cpu / any */
    public String deviceType = "gpu";

    /** 尝试加载的 OpenCL 库名 */
    public String libraryName = "libOpenCL.so";

    /** 是否写验证报告文件 */
    public boolean writeReport = true;

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static ModConfig load() {
        Path configDir = FabricLoader.getInstance().getConfigDir();
        Path file = configDir.resolve("gpuworldgen.json");
        if (Files.exists(file)) {
            try {
                String json = Files.readString(file, StandardCharsets.UTF_8);
                ModConfig cfg = GSON.fromJson(json, ModConfig.class);
                if (cfg == null) cfg = new ModConfig();
                cfg.sanitize();
                return cfg;
            } catch (Exception e) {
                System.err.println("[GPUWorldGen] 配置解析失败，使用默认配置: " + e);
                ModConfig cfg = new ModConfig();
                cfg.sanitize();
                return cfg;
            }
        }
        ModConfig cfg = new ModConfig();
        cfg.sanitize();
        try {
            Files.createDirectories(configDir);
            Files.writeString(file, GSON.toJson(cfg), StandardCharsets.UTF_8);
        } catch (IOException e) {
            System.err.println("[GPUWorldGen] 无法写入默认配置: " + e);
        }
        return cfg;
    }

    private void sanitize() {
        if (size < 8) size = 8;
        if (size > 2048) size = 2048;
        if (freq < 1) freq = 1;
        if (freq > 128) freq = 128;
        if (workGroupSize < 1) workGroupSize = 64;
        if (workGroupSize > 1024) workGroupSize = 1024;
    }
}
