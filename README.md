# C2ME-GPUWorldGen

为 **Minecraft Java 版（手机端）** 实现真正 GPU 加速世界生成的 Fabric 适配 mod。

面向无 PC 的手机玩家（Pojav / Zalith Launcher 等），把世界生成里的 PerlinNoise 噪声计算
下沉到手机 GPU（Adreno）执行，并保证与 MC 原版噪声 **逐点一致（误差 < 1e-6）**。

> 当前处于 **GPU 接管阶段**：v0.10.x 已对首个被捕获的真实噪声实例实现 GPU 真提供噪声值
> （命中返 GPU 值、miss 走 CPU fallback、逐点比对通过才采用）。

---

## 能力与当前状态

| 阶段 | 状态 | 结果 |
|---|---|---|
| GLES double-double 噪声管线（fragment shader + FBO） | ✅ 已交付 | 5 轮 STABLE，大坐标误差 ~1e-14 |
| MC 精确噪声 GPU 复刻（PerlinNoise 3D 16 梯度） | ✅ 已交付 | maxErr 5.499e-07，0/16384（超大坐标 5e6） |
| 影子模式（真实实例逐点比对） | ✅ 已验收 | 真实坐标 maxErr 3.43e-07，mismatch 0 |
| double-double 坐标（>2e6 不再丢小数） | ✅ v0.9.0 | 拆分重建误差 1.9e-09，精度上限 2^24 |
| GPU 接管世界生成（首个实例） | ✅ v0.10.1 | mixin cancel 真返回 GPU 值 + CPU fallback + 预生成抽检 |

## 技术路线

- 用 Java **FFM（Foreign Function & Memory）** 直接 `dlopen` 手机上的 `libmobileglues.so`，
  `bind` 导出 GLES 原生函数，在游戏现有 EGL context 内执行。
- **不新建 Vulkan/GL 实例**、不引入额外 native 库，完全复用游戏渲染上下文。
- fragment shader 用 **double-double（DD）模拟双精度** 逐像素计算 MC PerlinNoise，
  经 RGBA32F FBO `glReadPixels` 回读，再与 CPU 端 MC 原算法逐点比对。
- mixin 挂载真实 `PerlinNoise`，用**反射**提取父类字段（perms/offset），拿到真实实例数据；
  v0.10.x 用 `setReturnValue + cancel` 对服务实例**真正接管**：命中 GPU 柱缓存直接返回
  GPU 值（跳过 CPU 原方法），miss 走原方法并把区块加入 GPU 预生成队列（16×16 柱 × 384 层，
  mode21 柱批量采样），预生成时每柱抽检与 CPU 参考比对（<1e-5）通过才写缓存。

**已判死路线（勿重复尝试）**：OpenCL（Android linker namespace 拦截，需 root）、
Vulkan 第二实例（必崩）、LWJGL desktop compute（stub）、LWJGL GLES15 包（classpath 不存在）、
FFM SSBO（索引全 -1）。

## 支持环境

- Minecraft 26.3 / Fabric Loader 0.20.1 / Java 25
- 渲染后端：**OpenGLES / MobileGL**（MobileGlues 2.x），GLES 3.2
- 测试真机：HONOR AAK-AN00（骁龙 8 Elite / Adreno 830）

## 构建

```bash
export JAVA_HOME=/path/to/jdk-25
/path/to/gradle-9.8.0/bin/gradle build --no-daemon --console=plain
# 产物: build/libs/gpu-worldgen-android-<version>.jar
```

需要：JDK 25、Gradle 9.8+、Loom 1.19-SNAPSHOT。

## 安装

1. 渲染后端选 OpenGLES / MobileGL（**不要** DirectVulkan）
2. 把 jar 放进 `mods/`
3. 配置文件 `config/gpuworldgen.json`（可选，默认 enabled）
4. 进游戏 → 进世界，自动跑 GPU 验证，报告写 `<游戏目录>/gpu-worldgen-verify.txt`

## 验证输出

进世界后自动执行：
- GL 5 轮稳定性验证（每轮重新编译 shader + 自检）
- MC 精确噪声对拍（mode 20，16384 点，超大坐标 5e6）
- GPU 接管统计：`提供噪声 XX% (hit / 总调用)`、预生成区块数、失败柱数

## 后续计划

- v0.11+：多实例接管（覆盖更多噪声实例）、预生成吞吐优化（多区块/批、回读压缩）
- 远期：Vulkan 后端支持

## 协议

<div align="center">Mod 采用 MIT 协议，与 C2ME 兼容</div>

<div align="center">© 2026 ZQ-斩情</div>

---

**免责声明**：本项目为独立 mod，与 C2ME（C2ME-Fabric）官方项目无隶属关系，仅命名致敬。引用 MC 类名用于适配，不打包分发 Minecraft 本体。
