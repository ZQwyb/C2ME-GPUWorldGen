package com.gpuworldgen.gl;

import com.gpuworldgen.shadow.ShadowShared;
import com.gpuworldgen.verify.CpuMcPerlin;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;

/**
 * GL Compute 通道 v0.5（FBO + 纹理 + fragment shader 渲染管线）：
 *
 * v0.4.2 真机结果：libmobileglues.so FFM 直连成功、GLSL ES 310 shader 编译链接成功，
 * 但 SSBO block 索引全返回 -1 —— MobileGlues 2.0.1·Dev1 的 SSBO 功能未实现（Dev 版不完整）。
 *
 * v0.5 彻底绕开 compute/SSBO，走游戏渲染本身就在跑的传统管线：
 * - perm / input 存纹理（texelFetch，GLES 3.0 核心）
 * - fragment shader 每像素算一个噪声点（128x128 FBO 全屏渲染）
 * - 结果写 FBO 颜色附件 RGBA32F → glReadPixels 读回
 * - 无 attribute（gl_VertexID 生成全屏 triangle）
 *
 * 诊断模式：mode=1 写常量自证管线通；mode=4 写固定 64.0/82.0 自证 fragment 真执行。
 */
public final class GlComputeRunner {

    // ---------- GLES 常量 ----------
    private static final int GL_VERSION = 0x1F02;
    private static final int GL_RENDERER = 0x1F01;
    private static final int GL_VERTEX_SHADER = 0x8B31;
    private static final int GL_FRAGMENT_SHADER = 0x8B30;
    private static final int GL_COMPILE_STATUS = 0x8B81;
    private static final int GL_LINK_STATUS = 0x8B82;
    private static final int GL_NO_ERROR = 0;

    private static final int GL_TEXTURE_2D = 0x0DE1;
    private static final int GL_TEXTURE0 = 0x84C0;
    private static final int GL_TEXTURE_MIN_FILTER = 0x2801;
    private static final int GL_TEXTURE_MAG_FILTER = 0x2800;
    private static final int GL_NEAREST = 0x2600;
    private static final int GL_RGBA32F = 0x8814;
    private static final int GL_RGBA8 = 0x8058;
    private static final int GL_RGBA = 0x1908;
    private static final int GL_UNSIGNED_BYTE = 0x1401;
    private static final int GL_RGBA_INTEGER = 0x8D99;
    private static final int GL_R32I = 0x8235;
    private static final int GL_RED_INTEGER = 0x8D94;
    private static final int GL_FLOAT = 0x1406;
    private static final int GL_INT = 0x1404;
    private static final int GL_UNPACK_ALIGNMENT = 0xCF5;

    private static final int GL_FRAMEBUFFER = 0x8D40;
    private static final int GL_COLOR_ATTACHMENT0 = 0x8CE0;
    private static final int GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
    private static final int GL_TRIANGLES = 0x0004;
    private static final int GL_BLEND = 0x0BE2;
    private static final int GL_DEPTH_TEST = 0x0B71;
    private static final int GL_SCISSOR_TEST = 0x0C11;
    private static final int GL_COLOR_BUFFER_BIT = 0x4000;
    private static final int GL_CURRENT_PROGRAM = 0x8B8D;
    private static final int GL_VERTEX_ARRAY_BINDING = 0x85B5;
    private static final int GL_TEXTURE_BINDING_2D = 0x8069;
    private static final int GL_ACTIVE_TEXTURE = 0x84E0;
    private static final int GL_COLOR_CLEAR_VALUE = 0x0C22;

    /** vertex shader：gl_VertexID 生成全屏 triangle（无 attribute） */
    private static final String VERT_SRC = """
            #version 320 es
            void main() {
                vec2 v = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
                gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);
            }
            """;

    /** fragment shader：texelFetch 读输入纹理，每像素算一个 double-double 噪声点 */
    private static final String FRAG_SRC = """
            #version 320 es
            precision highp float;
            precision highp int;

            uniform highp vec4 permU[64];    // perm[256]（每 vec4 存 4 个 0-255 值），glUniform4fv 上传
            uniform int uFreq;
            uniform int uTestMode;
            uniform vec2 uX0;               // v0.9.0: 坐标 double-double 拆分（x=hi, y=lo）。大坐标整数部分 float 精确(<2^24)，
                                            // 小数残差放 lo（相对小值 float 精确）→ 网格/插值恢复 double 精度，>2e6 不再丢小数
            uniform vec2 uY0;
            uniform int uColSlice;          // v0.10.0: mode21 柱批量——柱组偏移（0/1 → 区块 256 柱）
            uniform int uYSlice;            // v0.10.0: mode21 高度层偏移（0/1/2 → 384 层）
            uniform float uMinY;            // v0.10.0: mode21 柱起始高度（维度 minY=-64）；y = uMinY + uOffY(offsetY) + yl
            uniform float uOffX;             // v0.7.0: MC offsetX
            uniform float uOffY;             // v0.7.0: MC offsetY（2D 采样 y 切面常量）
            uniform float uOffZ;             // v0.7.0: MC offsetZ

            out vec4 fragColor;

            struct dd { float hi; float lo; };

            // perm 表只有 256 个独立值（标准 Perlin 后 256 是重复），i 用 &255 折回
            int permAt(int i) {
                int j = i & 255;
                vec4 q = permU[j >> 2];
                return int(q[j & 3]);
            }

            void two_sum(float a, float b, out dd s) {
                precise float sum = a + b;
                precise float bb  = sum - a;
                precise float err = (a - (sum - bb)) + (b - bb);
                s.hi = sum;
                s.lo = err;
            }

            // v0.6.0: Dekker 分裂显式逐步计算（防 MobileGlues 优化折叠破坏误差项）
            // v0.6.3: 全部中间变量加 precise（分裂链 ta-(ta-a) 若被化简成 a 则 alo=0，two_prod 退化）
            void two_prod(float a, float b, out dd p) {
                precise float prod = a * b;
                precise float ta = 4096.0 * a;
                precise float ahi = ta - (ta - a);
                precise float alo = a - ahi;
                precise float tb = 4096.0 * b;
                precise float bhi = tb - (tb - b);
                precise float blo = b - bhi;
                precise float e1 = ahi * bhi - prod;
                precise float e2 = e1 + ahi * blo;
                precise float e3 = e2 + alo * bhi;
                precise float err = e3 + alo * blo;
                p.hi = prod;
                p.lo = err;
            }

            dd dd_add(dd a, dd b) {
                dd s;
                two_sum(a.hi, b.hi, s);
                float e = a.lo + b.lo + s.lo;
                dd r;
                two_sum(s.hi, e, r);
                return r;
            }

            dd dd_sub(dd a, dd b) {
                dd nb;
                nb.hi = -b.hi;
                nb.lo = -b.lo;
                return dd_add(a, nb);
            }

            dd dd_mul(dd a, dd b) {
                dd p;
                two_prod(a.hi, b.hi, p);
                float t1 = a.hi * b.lo;
                float t2 = t1 + a.lo * b.hi;
                float e = p.lo + t2;
                dd r;
                two_sum(p.hi, e, r);
                return r;
            }

            dd dd_mul_s(dd a, float s) {
                dd b;
                two_prod(a.hi, s, b);
                float e = b.lo + a.lo * s;
                dd r;
                two_sum(b.hi, e, r);
                return r;
            }

            dd mk_dd(float hi, float lo) {
                dd r; r.hi = hi; r.lo = lo; return r;
            }

            dd dd_neg(dd a) {
                dd r; r.hi = -a.hi; r.lo = -a.lo; return r;
            }

            dd dd_fade(dd t) {
                dd t6 = dd_mul_s(t, 6.0);
                dd c  = dd_add(t6, mk_dd(-15.0, 0.0));
                dd tt = dd_mul(t, c);
                dd p  = dd_add(tt, mk_dd(10.0, 0.0));
                dd t2 = dd_mul(t, t);
                dd t3 = dd_mul(t2, t);
                return dd_mul(t3, p);
            }

            dd dd_lerp(dd t, dd a, dd b) {
                dd d  = dd_sub(b, a);
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

            // v0.7.0: MC GradientNoise.GRADIENT 16 立方体角梯度 dot（3D），
            // 逐位对齐 MC gradDot(int,float,float,float)。选 g[0]*x + g[1]*y + g[2]*z。
            dd mc_grad(int hash, dd x, dd y, dd z) {
                // 16 项表（h & 15）
                int h = hash & 15;
                // 用 switch 精确映射 MC GRADIENT（0..15）
                // (1,1,0)(-1,1,0)(1,-1,0)(-1,-1,0)(1,0,1)(-1,0,1)(1,0,-1)(-1,0,-1)
                // (0,1,1)(0,-1,1)(0,1,-1)(0,-1,-1)(1,1,0)(0,-1,1)(-1,1,0)(0,-1,-1)
                int gx, gy, gz;
                switch (h) {
                    case 0:  gx= 1; gy= 1; gz= 0; break;
                    case 1:  gx=-1; gy= 1; gz= 0; break;
                    case 2:  gx= 1; gy=-1; gz= 0; break;
                    case 3:  gx=-1; gy=-1; gz= 0; break;
                    case 4:  gx= 1; gy= 0; gz= 1; break;
                    case 5:  gx=-1; gy= 0; gz= 1; break;
                    case 6:  gx= 1; gy= 0; gz=-1; break;
                    case 7:  gx=-1; gy= 0; gz=-1; break;
                    case 8:  gx= 0; gy= 1; gz= 1; break;
                    case 9:  gx= 0; gy=-1; gz= 1; break;
                    case 10: gx= 0; gy= 1; gz=-1; break;
                    case 11: gx= 0; gy=-1; gz=-1; break;
                    case 12: gx= 1; gy= 1; gz= 0; break;
                    case 13: gx= 0; gy=-1; gz= 1; break;
                    case 14: gx=-1; gy= 1; gz= 0; break;
                    default: gx= 0; gy=-1; gz=-1; break; // 15
                }
                dd term = dd_mul_s(x, float(gx));
                term = dd_add(term, dd_mul_s(y, float(gy)));
                return dd_add(term, dd_mul_s(z, float(gz)));
            }

            // v0.6.4: 坐标链提成函数（探针复用，缩短 shader；同时验证编译稳定性）
            // v0.6.6: 修复 dd floor 整数边界 bug——
            //   hi 是 dd 高位（float 量化到 0.03125 网格），当精确值 = 整数-0.015625 时
            //   hi tie round-even 向上量化成整数，floor(hi) 比正确值大 1；
            //   必须检查 hi 恰为整数且 lo<0 时 floor 减 1。
            void coords(out int xi, out int yi, out dd xf, out dd yf, out int X, out int Y) {
                ivec2 pxc = ivec2(gl_FragCoord.xy);
                vec2 uvc = (vec2(pxc) + 0.5) * (1.0 / 128.0);
                dd xc = dd_add(mk_dd(uvc.x, 0.0), mk_dd(uX0.x, uX0.y));
                dd yc = dd_add(mk_dd(uvc.y, 0.0), mk_dd(uY0.x, uY0.y));
                dd sxc = dd_mul_s(xc, float(uFreq));
                dd syc = dd_mul_s(yc, float(uFreq));
                xi = int(floor(sxc.hi));
                if (fract(sxc.hi) == 0.0 && sxc.lo < 0.0) xi -= 1;
                yi = int(floor(syc.hi));
                if (fract(syc.hi) == 0.0 && syc.lo < 0.0) yi -= 1;
                xf = dd_sub(sxc, mk_dd(float(xi), 0.0));
                yf = dd_sub(syc, mk_dd(float(yi), 0.0));
                X = xi & 255;
                Y = yi & 255;
            }

            void main() {
                ivec2 px = ivec2(gl_FragCoord.xy);
                int i = px.y * 128 + px.x;

                // ========== v0.7.0: mode 20 = MC 精确 PerlinNoise 2D 复刻 ==========
                // 逐位对齐 MC GradientNoise.GRADIENT（16 立方体角） + gradDot3 + 三线性。
                // 坐标：x = uX0 + uf*px + offsetX, z = uY0 + uf*py + offsetY（y 切面用 uOffY）。
                // MC 2D get(x,z) 内部是 3D get(wrap(x), 0, wrap(z))，y 格点由 offsetY 决定。
                // 与 mode 0（自研 2D 8 梯度）区分，仅用于对拍验证，不影响地形。
                if (uTestMode == 20) {
                    // 大坐标必须逐级 dd_add（先 uX0 hi/lo，再加 offset，再加网格小数），
                    // 单 float 合并会丢大坐标 ulp 残差，精度尽失。
                    dd x0 = mk_dd(uX0.x, uX0.y);
                    x0 = dd_add(x0, mk_dd(uOffX, 0.0));
                    x0 = dd_add(x0, mk_dd((float(px.x) + 0.5) * (1.0 / 128.0), 0.0));
                    dd z0 = mk_dd(uY0.x, uY0.y);
                    z0 = dd_add(z0, mk_dd(uOffZ, 0.0));
                    z0 = dd_add(z0, mk_dd((float(px.y) + 0.5) * (1.0 / 128.0), 0.0));
                    // y 切面：offsetY 常量（不随像素变化）
                    dd y0 = mk_dd(uOffY, 0.0);

                    int xi = int(floor(x0.hi));
                    if (fract(x0.hi) == 0.0 && x0.lo < 0.0) xi -= 1;
                    int yi = int(floor(y0.hi));
                    if (fract(y0.hi) == 0.0 && y0.lo < 0.0) yi -= 1;
                    int zi = int(floor(z0.hi));
                    if (fract(z0.hi) == 0.0 && z0.lo < 0.0) zi -= 1;
                    dd xf = dd_sub(x0, mk_dd(float(xi), 0.0));
                    dd yf = dd_sub(y0, mk_dd(float(yi), 0.0));
                    dd zf = dd_sub(z0, mk_dd(float(zi), 0.0));

                    // 8 角 hash（MC 三层 permute）
                    int a = permAt(xi), b = permAt(xi + 1);
                    int ab = permAt(a + yi), aa = permAt(a + yi + 1);
                    int ba = permAt(b + yi), bb = permAt(b + yi + 1);
                    int c000 = permAt(ab + zi);
                    int c100 = permAt(ba + zi);
                    int c010 = permAt(aa + zi);
                    int c110 = permAt(bb + zi);
                    int c001 = permAt(ab + zi + 1);
                    int c101 = permAt(ba + zi + 1);
                    int c011 = permAt(aa + zi + 1);
                    int c111 = permAt(bb + zi + 1);

                    // 8 角梯度（MC 16 梯度表 gradDot3）
                    dd one = mk_dd(1.0, 0.0);
                    dd d000 = mc_grad(c000, xf, yf, zf);
                    dd d100 = mc_grad(c100, dd_sub(xf, one), yf, zf);
                    dd d010 = mc_grad(c010, xf, dd_sub(yf, one), zf);
                    dd d110 = mc_grad(c110, dd_sub(xf, one), dd_sub(yf, one), zf);
                    dd d001 = mc_grad(c001, xf, yf, dd_sub(zf, one));
                    dd d101 = mc_grad(c101, dd_sub(xf, one), yf, dd_sub(zf, one));
                    dd d011 = mc_grad(c011, xf, dd_sub(yf, one), dd_sub(zf, one));
                    dd d111 = mc_grad(c111, dd_sub(xf, one), dd_sub(yf, one), dd_sub(zf, one));

                    dd u20 = dd_fade(xf), v20 = dd_fade(yf), w20 = dd_fade(zf);
                    dd l00 = dd_lerp(u20, d000, d100);
                    dd l10 = dd_lerp(u20, d010, d110);
                    dd l01 = dd_lerp(u20, d001, d101);
                    dd l11 = dd_lerp(u20, d011, d111);
                    dd y0v = dd_lerp(v20, l00, l10);
                    dd y1v = dd_lerp(v20, l01, l11);
                    dd res20 = dd_lerp(w20, y0v, y1v);

                    fragColor = vec4(res20.hi, res20.lo, xf.hi, zf.hi);
                    return;
                }

                // ========== v0.10.0: mode 21 = GPU 柱批量预生成（区块 16x16 柱 x 384 层）==========
                // 6 次 draw 覆盖整区块：柱组 uColSlice*128（0..255）+ 高度层 uYSlice*128（0..383）。
                // 坐标 = 格点（整数）：x = blockX + col%16，z = blockZ + col/16，y = minY + yl。
                // 与 mode20 同一 dd 坐标链 + MC 精确噪声，输出 [hi,lo] 供 CPU 抽检比对后写入柱缓存。
                if (uTestMode == 21) {
                    int col = px.x + uColSlice * 128;       // 0..255 区块内柱号
                    int yl = px.y + uYSlice * 128;          // 0..383 高度层
                    float cx = float(col % 16);
                    float cz = float(col / 16);
                    dd x0 = mk_dd(uX0.x, uX0.y);
                    x0 = dd_add(x0, mk_dd(uOffX, 0.0));
                    x0 = dd_add(x0, mk_dd(cx, 0.0));
                    dd z0 = mk_dd(uY0.x, uY0.y);
                    z0 = dd_add(z0, mk_dd(uOffZ, 0.0));
                    z0 = dd_add(z0, mk_dd(cz, 0.0));
                    dd y0 = mk_dd(uMinY, 0.0);
                    y0 = dd_add(y0, mk_dd(uOffY, 0.0));    // + offsetY（MC get(DDD)F: y 世界坐标 = 传入 y + offsetY）
                    y0 = dd_add(y0, mk_dd(float(yl), 0.0));

                    int xi = int(floor(x0.hi));
                    if (fract(x0.hi) == 0.0 && x0.lo < 0.0) xi -= 1;
                    int yi = int(floor(y0.hi));
                    if (fract(y0.hi) == 0.0 && y0.lo < 0.0) yi -= 1;
                    int zi = int(floor(z0.hi));
                    if (fract(z0.hi) == 0.0 && z0.lo < 0.0) zi -= 1;
                    dd xf = dd_sub(x0, mk_dd(float(xi), 0.0));
                    dd yf = dd_sub(y0, mk_dd(float(yi), 0.0));
                    dd zf = dd_sub(z0, mk_dd(float(zi), 0.0));

                    int a = permAt(xi), b = permAt(xi + 1);
                    int ab = permAt(a + yi), aa = permAt(a + yi + 1);
                    int ba = permAt(b + yi), bb = permAt(b + yi + 1);
                    int c000 = permAt(ab + zi);
                    int c100 = permAt(ba + zi);
                    int c010 = permAt(aa + zi);
                    int c110 = permAt(bb + zi);
                    int c001 = permAt(ab + zi + 1);
                    int c101 = permAt(ba + zi + 1);
                    int c011 = permAt(aa + zi + 1);
                    int c111 = permAt(bb + zi + 1);

                    dd one = mk_dd(1.0, 0.0);
                    dd d000 = mc_grad(c000, xf, yf, zf);
                    dd d100 = mc_grad(c100, dd_sub(xf, one), yf, zf);
                    dd d010 = mc_grad(c010, xf, dd_sub(yf, one), zf);
                    dd d110 = mc_grad(c110, dd_sub(xf, one), dd_sub(yf, one), zf);
                    dd d001 = mc_grad(c001, xf, yf, dd_sub(zf, one));
                    dd d101 = mc_grad(c101, dd_sub(xf, one), yf, dd_sub(zf, one));
                    dd d011 = mc_grad(c011, xf, dd_sub(yf, one), dd_sub(zf, one));
                    dd d111 = mc_grad(c111, dd_sub(xf, one), dd_sub(yf, one), dd_sub(zf, one));

                    dd u21 = dd_fade(xf), v21 = dd_fade(yf), w21 = dd_fade(zf);
                    dd l00 = dd_lerp(u21, d000, d100);
                    dd l10 = dd_lerp(u21, d010, d110);
                    dd l01 = dd_lerp(u21, d001, d101);
                    dd l11 = dd_lerp(u21, d011, d111);
                    dd y0v = dd_lerp(v21, l00, l10);
                    dd y1v = dd_lerp(v21, l01, l11);
                    dd res21 = dd_lerp(w21, y0v, y1v);

                    fragColor = vec4(res21.hi, res21.lo, 0.0, 0.0);
                    return;
                }

                if (uTestMode == 1) {
                    fragColor = vec4(1.0, float(i) * 0.001, 0.0, 1.0);
                    return;
                }
                if (uTestMode == 4) {
                    // 输出 permU[0] 原始值，验证 uniform 数组上传
                    fragColor = permU[0];
                    return;
                }
                if (uTestMode == 5) {
                    // dd 探针：dd_add(100000, 0.00390625)，预期 R=100000 G=0.00390625（若 G=0 = dd 残差被丢）
                    dd t = dd_add(mk_dd(100000.0, 0.0), mk_dd(0.00390625, 0.0));
                    fragColor = vec4(t.hi, t.lo, 0.0, 0.0);
                    return;
                }
                if (uTestMode == 6) {
                    // 纯 FP32 探针：100000 + 0.00390625（FP32 会舍入到 100000），预期 R=100000 G=0.00390625 B=100000
                    float s = 100000.0 + 0.00390625;
                    fragColor = vec4(100000.0, 0.00390625, s, 0.0);
                    return;
                }
                if (uTestMode == 7) {
                    // 验证 uX0/uY0 uniform 传入值（v0.9.0 为 vec2 hi/lo，输出 hi 部分）
                    fragColor = vec4(uX0.x, uY0.x, 0.0, 0.0);
                    return;
                }
                if (uTestMode == 8) {
                    // 验证 uniform 路径的 dd_add(uv.x, uX0) 残差（对比 mode=5 常量路径）
                    vec2 uv8 = (vec2(px) + 0.5) * (1.0 / 128.0);
                    dd t8 = dd_add(mk_dd(uv8.x, 0.0), mk_dd(uX0.x, uX0.y));
                    fragColor = vec4(t8.hi, t8.lo, uv8.x, 0.0);
                    return;
                }
                if (uTestMode == 9) {
                    // 整数链探针：xi yi X Y（验证 floor/&255 与 CPU 一致）
                    int xi9, yi9, X9, Y9; dd xf9, yf9;
                    coords(xi9, yi9, xf9, yf9, X9, Y9);
                    fragColor = vec4(float(xi9), float(yi9), float(X9), float(Y9));
                    return;
                }
                if (uTestMode == 10) {
                    // xf/yf 探针：大坐标组应输出精确小数（px=89 → xf=0.796875, lo=0）
                    int xi0, yi0, X0, Y0; dd xf0, yf0;
                    coords(xi0, yi0, xf0, yf0, X0, Y0);
                    fragColor = vec4(xf0.hi, xf0.lo, yf0.hi, yf0.lo);
                    return;
                }
                if (uTestMode == 11) {
                    // hash 探针：h00 h10 h01 h11（验证 permAt 查表与 CPU 一致）
                    int xi1, yi1, X1, Y1; dd xf1, yf1;
                    coords(xi1, yi1, xf1, yf1, X1, Y1);
                    int h00 = permAt(permAt(X1) + Y1);
                    int h10 = permAt(permAt(X1 + 1) + Y1);
                    int h01 = permAt(permAt(X1) + Y1 + 1);
                    int h11 = permAt(permAt(X1 + 1) + Y1 + 1);
                    fragColor = vec4(float(h00), float(h10), float(h01), float(h11));
                    return;
                }
                if (uTestMode == 12) {
                    // fade 探针：u=dd_fade(xf) v=dd_fade(yf)（验证 fade 链精度）
                    int xi2, yi2, X2, Y2; dd xf2, yf2;
                    coords(xi2, yi2, xf2, yf2, X2, Y2);
                    dd u12 = dd_fade(xf2);
                    dd v12 = dd_fade(yf2);
                    fragColor = vec4(u12.hi, u12.lo, v12.hi, v12.lo);
                    return;
                }
                if (uTestMode == 13) {
                    // grad 探针：g00 g10 g01 g11（4 个 grad 结果，验证 grad 链）
                    int xi3, yi3, X3, Y3; dd xf3, yf3;
                    coords(xi3, yi3, xf3, yf3, X3, Y3);
                    dd xm3 = dd_sub(xf3, mk_dd(1.0, 0.0));
                    dd ym3 = dd_sub(yf3, mk_dd(1.0, 0.0));
                    int h00 = permAt(permAt(X3) + Y3);
                    int h10 = permAt(permAt(X3 + 1) + Y3);
                    int h01 = permAt(permAt(X3) + Y3 + 1);
                    int h11 = permAt(permAt(X3 + 1) + Y3 + 1);
                    dd g00 = dd_grad(h00, xf3, yf3);
                    dd g10 = dd_grad(h10, xm3, yf3);
                    dd g01 = dd_grad(h01, xf3, ym3);
                    dd g11 = dd_grad(h11, xm3, ym3);
                    fragColor = vec4(g00.hi + g00.lo, g10.hi + g10.lo, g01.hi + g01.lo, g11.hi + g11.lo);
                    return;
                }
                if (uTestMode == 14) {
                    // lerp 探针：x1.hi x1.lo x2.hi x2.lo（验证前两次 lerp）
                    int xi4, yi4, X4, Y4; dd xf4, yf4;
                    coords(xi4, yi4, xf4, yf4, X4, Y4);
                    dd u14 = dd_fade(xf4);
                    dd xm4 = dd_sub(xf4, mk_dd(1.0, 0.0));
                    dd ym4 = dd_sub(yf4, mk_dd(1.0, 0.0));
                    int h00 = permAt(permAt(X4) + Y4);
                    int h10 = permAt(permAt(X4 + 1) + Y4);
                    int h01 = permAt(permAt(X4) + Y4 + 1);
                    int h11 = permAt(permAt(X4 + 1) + Y4 + 1);
                    dd x1 = dd_lerp(u14, dd_grad(h00, xf4, yf4), dd_grad(h10, xm4, yf4));
                    dd x2 = dd_lerp(u14, dd_grad(h01, xf4, ym4), dd_grad(h11, xm4, ym4));
                    fragColor = vec4(x1.hi, x1.lo, x2.hi, x2.lo);
                    return;
                }

                // 坐标由 gl_FragCoord 直接生成（无需输入纹理）；uX0/uY0 用于大坐标偏移验证
                int xi, yi, X, Y; dd xf, yf;
                coords(xi, yi, xf, yf, X, Y);

                dd u = dd_fade(xf);
                dd vv = dd_fade(yf);

                int h00 = permAt(permAt(X) + Y);
                int h10 = permAt(permAt(X + 1) + Y);
                int h01 = permAt(permAt(X) + Y + 1);
                int h11 = permAt(permAt(X + 1) + Y + 1);

                dd xm = dd_sub(xf, mk_dd(1.0, 0.0));
                dd ym = dd_sub(yf, mk_dd(1.0, 0.0));

                dd x1 = dd_lerp(u, dd_grad(h00, xf, yf), dd_grad(h10, xm, yf));
                dd x2 = dd_lerp(u, dd_grad(h01, xf, ym), dd_grad(h11, xm, ym));
                dd res = dd_lerp(vv, x1, x2);

                // B/A 通道带 xf/yf 探针（应为 [0,1]）
                fragColor = vec4(res.hi, res.lo, xf.hi, yf.hi);
            }
            """;

    private final StringBuilder report;

    // ---------- FFM ----------
    private final Arena arena = Arena.ofShared();
    private final Linker linker = Linker.nativeLinker();

    // ---------- GLES 函数 ----------
    private MethodHandle glGetString, glCreateShader, glShaderSource, glCompileShader, glDeleteShader;
    private MethodHandle glGetShaderiv, glGetShaderInfoLog;
    private MethodHandle glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog, glDeleteProgram;
    private MethodHandle glUseProgram, glUniform1i, glUniform1f, glUniform2f, glUniform4fv, glGetUniformLocation;
    private MethodHandle glGenTextures, glBindTexture, glTexImage2D, glTexSubImage2D, glTexParameteri, glActiveTexture;
    private MethodHandle glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D, glCheckFramebufferStatus;
    private MethodHandle glGenVertexArrays, glBindVertexArray;
    private MethodHandle glDrawArrays, glViewport, glReadPixels, glGetError, glFinish, glGetIntegerv;
    private MethodHandle glDisable, glEnable, glIsEnabled, glGetFloatv;
    private MethodHandle glClearColor, glClear;
    private MethodHandle glPixelStorei;

    private int prog, texOut, fbo, vao;
    private int locPermU, locFreq, locTestMode, locX0, locY0;
    private int locOffX, locOffY, locOffZ;   // v0.7.0: MC offsetX/Y/Z uniform
    private int size = 128;

    // v0.6.9: 渲染前保存/恢复游戏全套 GL 状态（此前 v0.6.8 只恢复帧缓冲+viewport，
    // 纹理绑定/程序/VAO 泄漏导致 FFM 直连绕过 MC RenderSystem 缓存 → 主菜单 GUI 文字花屏）
    private static final int GL_FRAMEBUFFER_BINDING = 0x8CA6;
    private static final int GL_VIEWPORT = 0x0BA2;
    private int savedFbo;
    private final int[] savedViewport = new int[4];
    private int savedProgram;
    private int savedVao;
    private int savedActiveTexture;
    private int savedTexBinding;
    private final float[] savedClearColor = new float[4];
    private int savedBlend, savedDepth, savedScissor;
    private boolean waitLogged;

    /** 大坐标偏移（double-double 价值验证）：MC 世界坐标量级 */
    private static final float BIG_OFFSET = 100000.0f;

    private boolean initDone = false;
    private String initError;

    private int[] reqPerm;
    private float[] reqInput;
    private int reqFreq;
    private CountDownLatch done;
    private volatile float[][] diagResults;
    private volatile float[] diagBig;       // 大坐标偏移组的 mode=0 输出
    private volatile float[][] diagBigAll;  // 大坐标偏移组全部 mode 输出（诊断）
    private volatile long gpuNanos;

    // v0.7.0: MC 精确噪声对拍请求/结果（由 VerifyRunner 在 runDiagnostics 前设置）
    private volatile boolean mcRequested;
    private volatile int mcFreq;
    private volatile double mcX0, mcY0;
    private volatile double mcOx, mcOy, mcOz;
    private volatile float[] mcResult;       // mode 20 输出 [hi,lo,...]（MC 精确算法 GPU 复刻）

    // v0.8.0: 影子模式（验证完成后持续 GPU 比对真实 PerlinNoise，GPU 结果只记录丢弃，不改地形）
    private volatile boolean verifyDone;
    private volatile int shadowTick;
    private int shadowProg = -1;             // 影子比对专用 program（保留到游戏结束，不删）
    private int spPermU = -1, spFreq = -1, spTestMode = -1, spX0 = -1, spY0 = -1;
    private int spOffX = -1, spOffY = -1, spOffZ = -1;
    private int spColSlice = -1, spYSlice = -1;   // v0.10.0: mode21 柱组/高度层偏移
    private int spMinY = -1;                        // v0.10.0: mode21 柱起始高度
    private int pregenTickCount;                    // v0.10.0: 预生成节流计数
    private long pregenNanos;
    private long shadowSamples, shadowOver, shadowChecks, shadowSkipBig;
    private double shadowMaxErr, shadowMaxAbsCoord;
    /** v0.9.0: dd 坐标后 float 网格限制放宽到 1e7（整数部分 <2^24 仍精确，小数走 lo 通道）。
     * 超限仍降级到固定小坐标，只验证注入正确。真实世界坐标（<1000）远低于此。 */
    private static final double SAFE_LIMIT = 10_000_000.0;
    private static final int SHADOW_INTERVAL = 20;   // 约每 20 tick（~1s）比对一次
    /** v0.8.5: 影子比对做有限次即停（定位进世界 native 崩溃：持续 GL 读回疑似越界）。
     * 保留前几次真实坐标比对证明 GPU 复刻正确，之后不再触碰 GL，杜绝持续操作引发崩溃。 */
    private static final int SHADOW_MAX_CHECKS = 3;

    // v0.6.7: 多轮稳定性验证（一次启动连跑 STABILITY_ROUNDS 轮，每轮重新编译 shader）
    // 目的：MobileGlues GLSL 编译器非确定性（此前成功率约 50%），单次 PASS 不足，
    // 需确认每轮编译自检能否通过、以及编译成功后精度是否稳定复现。
    private static final int STABILITY_ROUNDS = 5;
    private int lastVs, lastFs;
    private volatile int[] roundAttempts;        // 每轮编译通过的尝试次数（1..5）；负值=该轮编译失败（绝对值=尝试次数）
    private volatile float[][] roundStdNoise;    // 每轮标准组 mode=0 最终噪声
    private volatile float[][] roundBigNoise;    // 每轮大坐标组 mode=0 最终噪声
    private volatile float[][] roundBigMode9;    // 每轮大坐标组 mode=9 整数链
    private volatile long[] roundNanos;          // 每轮 GPU 耗时

    public float[][] diagResults() { return diagResults; }
    public float[] diagBig() { return diagBig; }
    public float[][] diagBigAll() { return diagBigAll; }

    /** v0.7.0: 设置 MC 精确噪声对拍参数（在 runDiagnostics 前调用）。offsets 为 MC 实例的 offsetX/Y/Z。
     *  v0.9.0: 坐标参数改 double（GPU 端 dd 拆分保留小数精度）。 */
    public void requestMcCheck(int freq, double x0, double y0, double offX, double offY, double offZ) {
        this.mcRequested = true;
        this.mcFreq = freq;
        this.mcX0 = x0; this.mcY0 = y0;
        this.mcOx = offX; this.mcOy = offY; this.mcOz = offZ;
    }

    /** v0.7.0: MC 精确噪声 GPU 复刻结果（mode 20 输出 [hi,lo] 对） */
    public float[] mcResult() { return mcResult; }

    public GlComputeRunner(StringBuilder report) {
        this.report = report;
        ClientTickEvents.END_CLIENT_TICK.register(this::onTick);
    }

    public float[][] runDiagnostics(int[] perm, float[] input, int count, int freq) throws InterruptedException {
        this.reqPerm = perm;
        this.reqInput = input;
        this.reqFreq = freq;
        this.done = new CountDownLatch(1);
        done.await();
        return diagResults;
    }

    public long gpuNanos() { return gpuNanos; }
    public String error() { return initError; }

    // v0.6.7: 多轮稳定性结果
    public int stabilityRounds() { return STABILITY_ROUNDS; }
    public int[] roundAttempts() { return roundAttempts; }
    public float[][] roundStdNoise() { return roundStdNoise; }
    public float[][] roundBigNoise() { return roundBigNoise; }
    public float[][] roundBigMode9() { return roundBigMode9; }
    public long[] roundNanos() { return roundNanos; }

    // ================= 渲染线程执行 =================

    private void onTick(net.minecraft.client.Minecraft client) {
        try {
            if (done == null) {
                // v0.8.0: 验证已完成 → 持续影子比对；v0.10.0: 接管激活后改 GPU 预生成（真提供噪声值）
                if (verifyDone && initDone && initError == null) {
                    if (ShadowShared.takeoverActive) {
                        try { pregenTick(); }
                        catch (Throwable pt) { System.out.println("[GPUWorldGen] 预生成异常: " + pt); }
                    } else {
                        try { shadowTickOnce(); }
                        catch (Throwable st) { System.out.println("[GPUWorldGen] 影子比对异常: " + st); }
                    }
                }
                return;
            }
            // v0.8.5: 验证改到【进入世界后】再跑——主菜单阶段完全不碰 GL，杜绝 0.8.2-0.8.4
            // 在主菜单后验证导致的文字花屏；加载画面（splash）也完全不碰 GL，杜绝红屏。
            // 进入世界后游戏内渲染稳定，验证一次性跑完（~1s 卡顿可接受），由 0.6.9 的全套
            // saveState/restoreState 兜底恢复。主菜单文字不再花屏（不验证就不污染）。
            if (!client.isGameLoadFinished() || client.player == null) {
                return;   // splash / 主菜单期间不渲染，等进入世界再验证
            }
            if (!initDone) {
                initResources();   // 一次性资源（dlopen + bind + 输出纹理 + FBO + VAO）
                initDone = true;
            }
            if (initError != null) { release(); return; }

            saveState();   // 保存游戏当前帧缓冲 / viewport / 清屏色

            int n = reqInput.length / 4;
            int R = STABILITY_ROUNDS;
            roundAttempts = new int[R];
            roundStdNoise = new float[R][];
            roundBigNoise = new float[R][];
            roundBigMode9 = new float[R][];
            roundNanos = new long[R];
            long totalT0 = System.nanoTime();

            for (int r = 0; r < R; r++) {
                report.append("    [GL] ===== 稳定性第 ").append(r + 1).append(" / ").append(R)
                        .append(" 轮（重新编译 shader）=====\n");
                long t0 = System.nanoTime();
                int[] att = new int[1];
                int p = compileProgram(att);
                if (p < 0) {
                    roundAttempts[r] = -att[0];
                    report.append("    [GL] 第 ").append(r + 1).append(" 轮编译自检 5 次均失败\n");
                    roundNanos[r] = System.nanoTime() - t0;
                    continue;
                }
                roundAttempts[r] = att[0];
                prog = p;
                // 每轮 program 是新建的，重新查 uniform locations
                locPermU = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("permU"));
                locFreq = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("uFreq"));
                locTestMode = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("uTestMode"));
                locX0 = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("uX0"));
                locY0 = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("uY0"));
                locOffX = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("uOffX"));
                locOffY = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("uOffY"));
                locOffZ = (int) invoke(glGetUniformLocation, prog, arena.allocateFrom("uOffZ"));

                float[][] std = runAllModes(n, reqFreq, 0.0f, 0.0f);
                float[][] big = runAllModes(n, reqFreq, BIG_OFFSET, BIG_OFFSET);
                roundStdNoise[r] = std[0];
                roundBigNoise[r] = big[0];
                roundBigMode9[r] = big[9];
                roundNanos[r] = System.nanoTime() - t0;

                // v0.7.0: 第 1 轮编译成功后跑一次 MC 精确噪声对拍（mode 20）
                if (r == 0 && mcRequested) {
                    long mt0 = System.nanoTime();
                    mcResult = runMcMode(n, mcFreq, mcX0, mcY0, (float) mcOx, (float) mcOy, (float) mcOz);
                    report.append("    [GL] MC 精确噪声对拍（mode 20）耗时: ")
                            .append(String.format("%.0f", (System.nanoTime() - mt0) / 1e6)).append(" ms\n");
                }

                // 兼容旧字段（指向最后一轮成功结果）
                diagResults = std;
                diagBigAll = big;
                diagBig = big[0];

                // 删除本轮 program + shaders（下一轮重新编译，避免对象泄漏）
                invoke(glDeleteShader, lastVs);
                invoke(glDeleteShader, lastFs);
                invoke(glDeleteProgram, p);
            }
            gpuNanos = System.nanoTime() - totalT0;
            restoreState();
            verifyDone = true;
            release();
        } catch (Throwable t) {
            initError = t.toString();
            report.append("    [GL] tick 异常: ").append(t).append('\n');
            try { restoreState(); } catch (Throwable ignored) {}
            release();
        }
    }

    private void release() {
        CountDownLatch d = done;
        if (d != null) { done = null; d.countDown(); }    }

    /** v0.8.0: 持续影子比对一次（渲染线程 tick 内节流调用）。GPU 结果只记录丢弃，不改地形。 */
    private void shadowTickOnce() {
        if (shadowChecks >= SHADOW_MAX_CHECKS) return;   // v0.8.5: 有限次后停止持续 GL 操作
        if (!ShadowShared.ready) return;              // 真实实例数据未就绪（还没进入世界生成）
        if (++shadowTick % SHADOW_INTERVAL != 0) return;   // 节流：约每 ~1s 一次
        double[][] coords = ShadowShared.drain(1);
        saveState();
        try {
            ensureShadowProg();
            if (shadowProg < 0) return;           // 本轮编译失败，下轮重试（MobileGlues 非确定）
            double x, z;
            if (coords != null) {
                x = coords[0][0]; z = coords[0][1];
                double ax = Math.abs(x), az = Math.abs(z);
                if (ax > shadowMaxAbsCoord) shadowMaxAbsCoord = ax;
                if (az > shadowMaxAbsCoord) shadowMaxAbsCoord = az;
                if (ax > SAFE_LIMIT || az > SAFE_LIMIT) {
                    // 真实世界坐标超出 float 网格能力（wrap + px/128 网格小数丢失）→ 降级到固定小坐标，
                    // 仅验证"真实实例 perms/offsets 注入正确 + GPU 复刻精度"。大坐标处理留 v0.9.0 dd 坐标。
                    shadowSkipBig++;
                    x = 100.0; z = 100.0;
                }
            } else {
                x = 100.0; z = 100.0;   // 兜底：队列空时用固定小坐标，保证有比对数据（验证注入）
            }
            float offX = (float) ShadowShared.realOx;
            float offY = (float) ShadowShared.realOy;
            float offZ = (float) ShadowShared.realOz;
            // GPU: mode20 网格 128 点，uX0=原始坐标 x（v0.9.0 拆 dd hi/lo，shader 内部 +offsetX → 坐标 = x + offX + (px+0.5)/128）
            // CPU 参考用原始 double x（与 dd 拆分后的 shader 坐标一致）。比对 dd 坐标链精度。
            float[] out = runMcModeShadow(128, 128, x, z, offX, offY, offZ);
            CpuMcPerlin cpu = new CpuMcPerlin(ShadowShared.realPerms, (double) offX, (double) offY, (double) offZ);
            double cpuBaseX = x;
            double cpuBaseZ = z;
            double z0 = cpuBaseZ + 0.5 / 128.0;
            int mismatch = 0;
            double maxErr = 0;
            for (int i = 0; i < 128; i++) {
                double cpuV = cpu.get(cpuBaseX + (i + 0.5) / 128.0, z0);
                double gpuV = out[i * 4];
                double e = Math.abs(gpuV - cpuV);
                if (e > maxErr) maxErr = e;
                if (e > 1e-6) mismatch++;
            }
            shadowSamples += 128;
            shadowChecks++;
            if (maxErr > shadowMaxErr) shadowMaxErr = maxErr;
            shadowOver += mismatch;
            if (shadowChecks % 20 == 1 || mismatch > 0) {
                System.out.println(String.format("[GPUWorldGen] 影子比对#%d: 128点 maxErr=%.2e mismatch=%d | 累计%d点 maxErr=%.2e over=%d | 跳过大坐标=%d 最大|坐标|=%.1e",
                        shadowChecks, maxErr, mismatch, shadowSamples, shadowMaxErr, shadowOver, shadowSkipBig, shadowMaxAbsCoord));
            }
        } finally {
            restoreState();
        }
    }

    // ================= v0.10.0 GPU 接管：预生成 =================

    /** 懒编译影子/接管专用 program（含 mode21 uniforms）。编译失败返回负值，下轮重试。 */
    private void ensureShadowProg() {
        if (shadowProg >= 0) return;
        int[] att = new int[1];
        shadowProg = compileProgram(att);
        if (shadowProg < 0) return;
        spPermU = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("permU"));
        spFreq = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uFreq"));
        spTestMode = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uTestMode"));
        spX0 = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uX0"));
        spY0 = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uY0"));
        spOffX = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uOffX"));
        spOffY = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uOffY"));
        spOffZ = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uOffZ"));
        spColSlice = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uColSlice"));
        spYSlice = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uYSlice"));
        spMinY = (int) invoke(glGetUniformLocation, shadowProg, arena.allocateFrom("uMinY"));
    }

    /** v0.10.0: GPU 预生成一个区块（16x16 柱 x 384 层），6 次 mode21 draw + 回读。返回 cols[col*384+yl]。 */
    private float[] runMcColumns(int blockX, int blockZ) {
        invoke(glUseProgram, shadowProg);
        invoke(glBindVertexArray, vao);
        byte[] rp = ShadowShared.realPerms;
        float[] permU = new float[512];
        for (int i = 0; i < 512; i++) permU[i] = rp[i & 255];
        MemorySegment permUSeg = arena.allocate(512L * 4);
        MemorySegment.copy(permU, 0, permUSeg, ValueLayout.JAVA_FLOAT, 0, 512);
        for (int i = 0; i < 64; i++) {
            invoke(glUniform4fv, spPermU + i, 1, permUSeg.asSlice((long) i * 16, 16));
        }
        invoke(glUniform1i, spFreq, 128);
        float[] sX = ddSplit(blockX);
        float[] sZ = ddSplit(blockZ);
        invoke(glUniform2f, spX0, sX[0], sX[1]);
        invoke(glUniform2f, spY0, sZ[0], sZ[1]);
        if (spOffX >= 0) invoke(glUniform1f, spOffX, (float) ShadowShared.realOx);
        if (spOffY >= 0) invoke(glUniform1f, spOffY, (float) ShadowShared.realOy);   // uOffY = offsetY
        if (spOffZ >= 0) invoke(glUniform1f, spOffZ, (float) ShadowShared.realOz);
        if (spMinY >= 0) invoke(glUniform1f, spMinY, (float) ShadowShared.MIN_Y);    // uMinY = 柱起始高度
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, fbo);
        invoke(glViewport, 0, 0, size, size);
        float[] cols = new float[256 * 384];
        for (int cs = 0; cs < 2; cs++) {
            for (int ys = 0; ys < 3; ys++) {
                invoke(glUniform1i, spTestMode, 21);
                invoke(glUniform1i, spColSlice, cs);
                invoke(glUniform1i, spYSlice, ys);
                prepDraw();
                invoke(glDrawArrays, GL_TRIANGLES, 0, 3);
                invoke(glFinish);
                float[] out = readFbo(size * size);
                for (int py = 0; py < 128; py++) {
                    for (int pxx = 0; pxx < 128; pxx++) {
                        int col = cs * 128 + pxx;
                        int yl = ys * 128 + py;
                        cols[col * 384 + yl] = out[(py * 128 + pxx) * 4];
                    }
                }
            }
        }
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, 0);
        return cols;
    }

    /** v0.10.0: 渲染线程每 tick：从 pending 取 1 区块 GPU 预生成，抽检比对 CPU 参考（<1e-5）通过才写柱缓存。 */
    private void pregenTick() {
        if (ShadowShared.pending.isEmpty()) return;
        ensureShadowProg();
        if (shadowProg < 0) return;                 // 编译失败，下轮重试（MobileGlues 非确定）
        if (++pregenTickCount % 2 != 0) return;     // 每 2 tick 处理 1 区块，避免占帧过多
        Long ck = ShadowShared.pending.keySet().iterator().next();
        if (ck == null) return;
        ShadowShared.pending.remove(ck);
        int blockX = (int) (ck >> 32);
        int blockZ = (int) (ck.longValue());
        saveState();
        try {
            long t0 = System.nanoTime();
            float[] cols = runMcColumns(blockX, blockZ);
            // 抽检：每柱 24 层（yl 步长 16），CPU 参考用与 GPU 一致的 float 化 offsets
            CpuMcPerlin cpu = new CpuMcPerlin(ShadowShared.realPerms,
                    (double) (float) ShadowShared.realOx,
                    (double) (float) ShadowShared.realOy,
                    (double) (float) ShadowShared.realOz);
            int failedCols = 0;
            boolean[] ok = new boolean[256];
            for (int col = 0; col < 256; col++) {
                int cx = col % 16, cz = col / 16;
                double wx = blockX + cx, wz = blockZ + cz;
                boolean colOk = true;
                for (int yl = 0; yl < 384 && colOk; yl += 16) {
                    double ref = cpu.get3(wx, ShadowShared.MIN_Y + yl, wz);
                    double gpuV = cols[col * 384 + yl];
                    if (Math.abs(gpuV - ref) > 1e-5) colOk = false;
                }
                ok[col] = colOk;
                if (!colOk) failedCols++;
            }
            if (failedCols < 256) {                 // 至少 1 柱通过才写；全失败 = GPU 整体异常（不写 → 全 CPU fallback）
                float[] colData = new float[384];
                for (int col = 0; col < 256; col++) {
                    if (!ok[col]) continue;         // 失败柱不写 → mixin miss → CPU fallback
                    System.arraycopy(cols, col * 384, colData, 0, 384);
                    ShadowShared.columns.put(ShadowShared.colKey(blockX + col % 16, blockZ + col / 16), colData.clone());
                }
                ShadowShared.columnsPregen.incrementAndGet();
                if (failedCols > 0) ShadowShared.columnsFailed.addAndGet(failedCols);
            }
            pregenNanos += System.nanoTime() - t0;
            long total = ShadowShared.gpuHits.get() + ShadowShared.gpuMisses.get();
            if (ShadowShared.columnsPregen.get() % 8 == 1 || failedCols > 0) {
                System.out.println(String.format("[GPUWorldGen] GPU 接管: 提供噪声 %.1f%% (%d hit / %d 总调用) | 预生成区块=%d 失败柱=%d 队列=%d | 平均=%.1fms/区块",
                        100.0 * ShadowShared.gpuHits.get() / Math.max(1, total),
                        ShadowShared.gpuHits.get(), total,
                        ShadowShared.columnsPregen.get(), ShadowShared.columnsFailed.get(),
                        ShadowShared.pending.size(),
                        (double) pregenNanos / Math.max(1, ShadowShared.columnsPregen.get()) / 1e6));
            }
        } finally {
            restoreState();
        }
    }

    /** v0.8.0: 影子模式专用 mode20 网格采样（用 shadowProg/locations + 真实实例 perms）。返回 [hi,lo,...] 128 点 */
    private float[] runMcModeShadow(int count, int freq, double x0, double y0, float offX, float offY, float offZ) {
        invoke(glUseProgram, shadowProg);
        invoke(glBindVertexArray, vao);
        byte[] rp = ShadowShared.realPerms;
        float[] permU = new float[512];
        for (int i = 0; i < 512; i++) permU[i] = rp[i & 255];
        MemorySegment permUSeg = arena.allocate(512L * 4);
        MemorySegment.copy(permU, 0, permUSeg, ValueLayout.JAVA_FLOAT, 0, 512);
        for (int i = 0; i < 64; i++) {
            invoke(glUniform4fv, spPermU + i, 1, permUSeg.asSlice((long) i * 16, 16));
        }
        invoke(glUniform1i, spFreq, freq);
        // v0.9.0: 坐标拆 dd(hi,lo) 传 vec2 uniform（真实坐标小数不再丢）
        float[] sX = ddSplit(x0);
        float[] sY = ddSplit(y0);
        invoke(glUniform2f, spX0, sX[0], sX[1]);
        invoke(glUniform2f, spY0, sY[0], sY[1]);
        if (spOffX >= 0) invoke(glUniform1f, spOffX, offX);
        if (spOffY >= 0) invoke(glUniform1f, spOffY, offY);
        if (spOffZ >= 0) invoke(glUniform1f, spOffZ, offZ);
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, fbo);
        invoke(glViewport, 0, 0, size, size);
        prepDraw();
        invoke(glUniform1i, spTestMode, 20);
        invoke(glDrawArrays, GL_TRIANGLES, 0, 3);
        invoke(glFinish);
        // 读回全屏 size*size（readFbo 的 glReadPixels 固定读全屏，buf 须够大，不能只按 count 分配合）*/
        float[] out = readFbo(size * size);
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, 0);
        return out;
    }

    /** v0.6.9: 保存游戏当前全套 GL 状态（渲染前调用），渲染后精确恢复，杜绝 GUI 文字/画面花屏 */
    private void saveState() {
        MemorySegment oneInt = arena.allocate(4);
        invoke(glGetIntegerv, GL_FRAMEBUFFER_BINDING, oneInt);
        savedFbo = oneInt.get(ValueLayout.JAVA_INT, 0);
        MemorySegment vp = arena.allocate(16);
        invoke(glGetIntegerv, GL_VIEWPORT, vp);
        for (int i = 0; i < 4; i++) savedViewport[i] = vp.get(ValueLayout.JAVA_INT, (long) i * 4);

        invoke(glGetIntegerv, GL_CURRENT_PROGRAM, oneInt);
        savedProgram = oneInt.get(ValueLayout.JAVA_INT, 0);
        invoke(glGetIntegerv, GL_VERTEX_ARRAY_BINDING, oneInt);
        savedVao = oneInt.get(ValueLayout.JAVA_INT, 0);
        invoke(glGetIntegerv, GL_ACTIVE_TEXTURE, oneInt);
        savedActiveTexture = oneInt.get(ValueLayout.JAVA_INT, 0);
        // GL_TEXTURE_BINDING_2D 返回的是"当前激活纹理单元"上的绑定，必须先读 ACTIVE_TEXTURE 再读它
        invoke(glGetIntegerv, GL_TEXTURE_BINDING_2D, oneInt);
        savedTexBinding = oneInt.get(ValueLayout.JAVA_INT, 0);

        MemorySegment cc = arena.allocate(16);
        invoke(glGetFloatv, GL_COLOR_CLEAR_VALUE, cc);
        for (int i = 0; i < 4; i++) savedClearColor[i] = cc.get(ValueLayout.JAVA_FLOAT, (long) i * 4);

        savedBlend = isEnabled(GL_BLEND);
        savedDepth = isEnabled(GL_DEPTH_TEST);
        savedScissor = isEnabled(GL_SCISSOR_TEST);

        report.append("    [GL] 保存状态: 帧缓冲=").append(savedFbo)
                .append(" viewport=[").append(savedViewport[0]).append(',').append(savedViewport[1])
                .append(',').append(savedViewport[2]).append(',').append(savedViewport[3]).append(']')
                .append(" 程序=").append(savedProgram)
                .append(" VAO=").append(savedVao)
                .append(" 激活纹理单元=").append(savedActiveTexture)
                .append(" 纹理绑定=").append(savedTexBinding)
                .append(" blend/depth/scissor=").append(savedBlend).append('/').append(savedDepth).append('/').append(savedScissor)
                .append('\n');
    }

    /** v0.6.9: 精确恢复全套 GL 状态（渲染后调用）。此前只恢复帧缓冲+viewport，纹理绑定泄漏导致 GUI 文字乱码 */
    private void restoreState() {
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, savedFbo);
        invoke(glViewport, savedViewport[0], savedViewport[1], savedViewport[2], savedViewport[3]);
        invoke(glUseProgram, savedProgram);
        invoke(glBindVertexArray, savedVao);
        // 先切回原激活纹理单元，再恢复该单元上的 2D 纹理绑定（顺序不能反）
        invoke(glActiveTexture, savedActiveTexture);
        invoke(glBindTexture, GL_TEXTURE_2D, savedTexBinding);
        restoreEnabled(GL_BLEND, savedBlend);
        restoreEnabled(GL_DEPTH_TEST, savedDepth);
        restoreEnabled(GL_SCISSOR_TEST, savedScissor);
        invoke(glClearColor, savedClearColor[0], savedClearColor[1], savedClearColor[2], savedClearColor[3]);
    }

    private int isEnabled(int cap) {
        return ((int) invoke(glIsEnabled, cap)) & 0xFF;
    }

    private void restoreEnabled(int cap, int wasEnabled) {
        if (wasEnabled != 0) invoke(glEnable, cap);
        else invoke(glDisable, cap);
    }

    // ================= 初始化 =================

    /** v0.6.7: 一次性资源（dlopen + bind + 输出纹理 + FBO + VAO），shader 编译独立为 compileProgram */
    private void initResources() {
        report.append("    [GL] 渲染线程上下文就绪。\n");
        try {
            java.util.List<String> candidates = new java.util.ArrayList<>();
            candidates.add("libmobileglues.so");
            for (String p : findMobileGluesPaths()) candidates.add(p);

            SymbolLookup sym = null;
            String used = null;
            for (String c : candidates) {
                try {
                    sym = SymbolLookup.libraryLookup(c, arena);
                    used = c;
                    break;
                } catch (Throwable t) {
                    report.append("    [GL] 尝试打开 ").append(c).append(" 失败: ").append(t).append('\n');
                }
            }
            if (sym == null) {
                initError = "libmobileglues.so 所有候选路径均无法打开";
                report.append("    [GL] ").append(initError).append('\n');
                return;
            }
            report.append("    [GL] dlopen: ").append(used).append('\n');
            bind(sym);

            String ver = getString(glGetString, GL_VERSION);
            String ren = getString(glGetString, GL_RENDERER);
            report.append("    [GL] GL_VERSION: ").append(ver).append('\n');
            report.append("    [GL] GL_RENDERER: ").append(ren).append('\n');

            // ---- 输出纹理 texOut: 128x128 RGBA32F（FBO 颜色附件） ----
            MemorySegment texIds = arena.allocate(4);
            invoke(glGenTextures, 1, texIds);
            texOut = texIds.get(ValueLayout.JAVA_INT, 0);
            invoke(glBindTexture, GL_TEXTURE_2D, texOut);
            invoke(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            invoke(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            invoke(glTexImage2D, GL_TEXTURE_2D, 0, GL_RGBA32F, size, size, 0, GL_RGBA, GL_FLOAT, MemorySegment.NULL);

            // ---- FBO ----
            MemorySegment fboIds = arena.allocate(4);
            invoke(glGenFramebuffers, 1, fboIds);
            fbo = fboIds.get(ValueLayout.JAVA_INT, 0);
            invoke(glBindFramebuffer, GL_FRAMEBUFFER, fbo);
            invoke(glFramebufferTexture2D, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);
            int fbStatus = (int) invoke(glCheckFramebufferStatus, GL_FRAMEBUFFER);
            report.append("    [GL] FBO status: 0x").append(Integer.toHexString(fbStatus))
                    .append(fbStatus == GL_FRAMEBUFFER_COMPLETE ? " (COMPLETE)" : " (!! 非完成态)").append('\n');

            // ---- VAO ----
            MemorySegment vaoIds = arena.allocate(4);
            invoke(glGenVertexArrays, 1, vaoIds);
            vao = vaoIds.get(ValueLayout.JAVA_INT, 0);
            invoke(glBindVertexArray, vao);

        } catch (Throwable t) {
            initError = t.toString();
            report.append("    [GL] 初始化异常: ").append(t).append('\n');
        }
    }

    /**
     * v0.6.7: 编译 + 链接 + 自检（可重复调用，用于多轮稳定性验证）。
     * MobileGlues 编译器非确定性失败、glGetShaderiv 说谎，真实成败靠渲染自检：
     * 编译链接后立刻渲染 mode=1，读回 1 像素，R==1.0 且 G==0.0 才算成功。
     * @return 成功返回 program；失败返回 -1
     * outAttempts[0] 写入实际尝试次数（1..5）
     */
    private int compileProgram(int[] outAttempts) {
        int attempts = 0;
        for (int attempt = 0; attempt < 5; attempt++) {
            attempts = attempt + 1;
            int vs = compileShaderQuiet(GL_VERTEX_SHADER, VERT_SRC, attempt);
            int fs = compileShaderQuiet(GL_FRAGMENT_SHADER, FRAG_SRC, attempt);
            int p = (int) invoke(glCreateProgram);
            invoke(glAttachShader, p, vs);
            invoke(glAttachShader, p, fs);
            invoke(glLinkProgram, p);

            // 自检渲染
            invoke(glUseProgram, p);
            int ltm = (int) invoke(glGetUniformLocation, p, arena.allocateFrom("uTestMode"));
            if (ltm >= 0) invoke(glUniform1i, ltm, 1); // mode=1 写常量 R=1.0
            invoke(glBindFramebuffer, GL_FRAMEBUFFER, fbo);
            invoke(glViewport, 0, 0, size, size);
            invoke(glBindVertexArray, vao);
            prepDraw();
            invoke(glDrawArrays, GL_TRIANGLES, 0, 3);
            invoke(glFinish);
            MemorySegment one = arena.allocate(16);
            invoke(glReadPixels, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, one);
            float r = one.get(ValueLayout.JAVA_FLOAT, 0);
            float g = one.get(ValueLayout.JAVA_FLOAT, 4);
            if (r == 1.0f && g == 0.0f) {
                lastVs = vs;
                lastFs = fs;
                outAttempts[0] = attempts;
                report.append("    [GL] 编译自检通过（第 ").append(attempts).append(" 次尝试）\n");
                return p;
            }
            invoke(glDeleteProgram, p);
            invoke(glDeleteShader, vs);
            invoke(glDeleteShader, fs);
            report.append("    [GL] 编译自检失败（R=").append(r).append(" G=").append(g)
                    .append("），第 ").append(attempts).append(" 次尝试，重试\n");
        }
        outAttempts[0] = attempts;
        return -1;
    }

    private int compileShader(int type, String src) {
        int sh = (int) invoke(glCreateShader, type);
        MemorySegment s = arena.allocateFrom(src);
        MemorySegment ptrs = arena.allocate(ValueLayout.ADDRESS);
        ptrs.set(ValueLayout.ADDRESS, 0, s);
        invoke(glShaderSource, sh, 1, ptrs, MemorySegment.NULL);
        invoke(glCompileShader, sh);
        int ok = getShaderiv(sh, GL_COMPILE_STATUS);
        String log = infoLog(glGetShaderInfoLog, sh);
        String typeName = (type == GL_VERTEX_SHADER) ? "vertex" : "fragment";
        report.append("    [GL] ").append(typeName).append(" shader compile_status=").append(ok)
                .append(" log=").append(log.trim()).append('\n');
        if (ok == 0) {
            initError = typeName + " shader 编译失败";
            return 0;
        }
        return sh;
    }

    private int getShaderiv(int obj, int pname) {
        MemorySegment seg = arena.allocate(ValueLayout.JAVA_INT);
        invoke(glGetShaderiv, obj, pname, seg);
        return seg.get(ValueLayout.JAVA_INT, 0);
    }

    /** 编译 shader（不设 initError，供自检重试循环使用；真实成败由渲染自检判定） */
    private int compileShaderQuiet(int type, String src, int attempt) {
        int sh = (int) invoke(glCreateShader, type);
        MemorySegment s = arena.allocateFrom(src);
        MemorySegment ptrs = arena.allocate(ValueLayout.ADDRESS);
        ptrs.set(ValueLayout.ADDRESS, 0, s);
        invoke(glShaderSource, sh, 1, ptrs, MemorySegment.NULL);
        invoke(glCompileShader, sh);
        String typeName = (type == GL_VERTEX_SHADER) ? "vertex" : "fragment";
        report.append("    [GL] 编译 ").append(typeName).append("（尝试 ").append(attempt + 1).append("）\n");
        return sh;
    }

    private int getProgramiv(int obj, int pname) {
        MemorySegment seg = arena.allocate(ValueLayout.JAVA_INT);
        invoke(glGetProgramiv, obj, pname, seg);
        return seg.get(ValueLayout.JAVA_INT, 0);
    }

    // ================= 计算 =================

    /** v0.9.0: double 坐标拆成 dd(hi,lo) 两个 float（整数部分 float 精确到 2^24，小数残差放 lo）。
     * hi=(float)v 捕获大坐标整数，lo=(float)(v-hi) 捕获小数残差 → shader mk_dd(hi,lo) 恢复 double 精度。 */
    private static float[] ddSplit(double v) {
        float hi = (float) v;
        float lo = (float) (v - (double) hi);
        return new float[] { hi, lo };
    }

    private float[][] runAllModes(int count, int freq, float x0, float y0) {
        float[][] results = new float[15][];
        int n = count; // 128*128

        invoke(glUseProgram, prog);
        invoke(glBindVertexArray, vao);

        // ---- v0.5.9: perm 表逐元素上传（count=1，绕开 MobileGlues 批量 glUniform4fv 数组 bug） ----
        // shader 声明 uniform vec4 permU[64]（256 值），location 0..63（uFreq=64 uTestMode=65 已实证）
        float[] permU = new float[512];
        for (int i = 0; i < 512; i++) permU[i] = reqPerm[i];
        MemorySegment permUSeg = arena.allocate(512L * 4);
        MemorySegment.copy(permU, 0, permUSeg, ValueLayout.JAVA_FLOAT, 0, 512);
        for (int i = 0; i < 64; i++) {
            invoke(glUniform4fv, locPermU + i, 1, permUSeg.asSlice((long) i * 16, 16));
        }
        invoke(glUniform1i, locFreq, freq);
        // v0.9.0: 坐标拆 dd(hi,lo) 传 vec2 uniform
        float[] sX = ddSplit(x0);
        float[] sY = ddSplit(y0);
        invoke(glUniform2f, locX0, sX[0], sX[1]);
        invoke(glUniform2f, locY0, sY[0], sY[1]);

        // 绑 FBO + viewport；每次渲染前禁用游戏继承的状态 + 清屏（避免叠加污染）
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, fbo);
        invoke(glViewport, 0, 0, size, size);
        prepDraw();

        for (int mode = 1; mode <= 14; mode++) {
            invoke(glUniform1i, locTestMode, mode);
            invoke(glDrawArrays, GL_TRIANGLES, 0, 3);
            invoke(glFinish);
            results[mode] = readFbo(n);
            prepDraw(); // 清屏，避免下一次渲染叠加
        }
        invoke(glUniform1i, locTestMode, 0);
        invoke(glDrawArrays, GL_TRIANGLES, 0, 3);
        invoke(glFinish);
        results[0] = readFbo(n);

        // 恢复默认 FBO，不影响游戏
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, 0);

        int err = (int) invoke(glGetError);
        if (err != GL_NO_ERROR) {
            report.append("    [GL] 警告: glGetError = 0x").append(Integer.toHexString(err)).append('\n');
        }
        return results;
    }

    /** v0.7.0: 跑一次 MC 精确噪声对拍（mode 20）。复用 permU 上传 + FBO，只设 offsets。返回 [hi,lo] 每点 8 float */
    private float[] runMcMode(int count, int freq, double x0, double y0, float offX, float offY, float offZ) {
        int n = count;
        invoke(glUseProgram, prog);
        invoke(glBindVertexArray, vao);

        // permU 逐元素上传（与 runAllModes 相同）
        float[] permU = new float[512];
        for (int i = 0; i < 512; i++) permU[i] = reqPerm[i];
        MemorySegment permUSeg = arena.allocate(512L * 4);
        MemorySegment.copy(permU, 0, permUSeg, ValueLayout.JAVA_FLOAT, 0, 512);
        for (int i = 0; i < 64; i++) {
            invoke(glUniform4fv, locPermU + i, 1, permUSeg.asSlice((long) i * 16, 16));
        }
        invoke(glUniform1i, locFreq, freq);
        // v0.9.0: 坐标拆 dd(hi,lo) 传 vec2 uniform（大坐标小数不再丢）
        float[] sX = ddSplit(x0);
        float[] sY = ddSplit(y0);
        invoke(glUniform2f, locX0, sX[0], sX[1]);
        invoke(glUniform2f, locY0, sY[0], sY[1]);
        if (locOffX >= 0) invoke(glUniform1f, locOffX, offX);
        if (locOffY >= 0) invoke(glUniform1f, locOffY, offY);
        if (locOffZ >= 0) invoke(glUniform1f, locOffZ, offZ);

        invoke(glBindFramebuffer, GL_FRAMEBUFFER, fbo);
        invoke(glViewport, 0, 0, size, size);
        prepDraw();
        invoke(glUniform1i, locTestMode, 20);
        invoke(glDrawArrays, GL_TRIANGLES, 0, 3);
        invoke(glFinish);
        float[] out = readFbo(n);
        invoke(glBindFramebuffer, GL_FRAMEBUFFER, 0);
        return out;
    }

    /** 渲染前状态清理：禁用游戏继承的 blend/depth/scissor，清屏 */
    private void prepDraw() {
        invoke(glDisable, GL_BLEND);
        invoke(glDisable, GL_DEPTH_TEST);
        invoke(glDisable, GL_SCISSOR_TEST);
        invoke(glClearColor, 0.0f, 0.0f, 0.0f, 0.0f);
        invoke(glClear, GL_COLOR_BUFFER_BIT);
    }

    /** 从 FBO 读回 RGBA float（每像素 4 float：R=hi/探针A, G=lo/探针B, B=xf, A=yf） */
    private float[] readFbo(int n) {
        MemorySegment buf = arena.allocate((long) n * 4 * 4); // 128*128 RGBA float
        invoke(glReadPixels, 0, 0, size, size, GL_RGBA, GL_FLOAT, buf);
        float[] out = new float[n * 4];
        for (int i = 0; i < n; i++) {
            long off = (long) i * 16;
            out[i * 4] = buf.get(ValueLayout.JAVA_FLOAT, off);         // R
            out[i * 4 + 1] = buf.get(ValueLayout.JAVA_FLOAT, off + 4); // G
            out[i * 4 + 2] = buf.get(ValueLayout.JAVA_FLOAT, off + 8); // B
            out[i * 4 + 3] = buf.get(ValueLayout.JAVA_FLOAT, off + 12); // A
        }
        return out;
    }

    // ================= FFM 工具 =================

    private void bind(SymbolLookup sym) {
        glGetString = bind0(sym, "glGetString", FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        glCreateShader = bind0(sym, "glCreateShader", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glShaderSource = bind0(sym, "glShaderSource", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        glCompileShader = bind0(sym, "glCompileShader", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glDeleteShader = bind0(sym, "glDeleteShader", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glDeleteProgram = bind0(sym, "glDeleteProgram", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glGetShaderiv = bind0(sym, "glGetShaderiv", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glGetShaderInfoLog = bind0(sym, "glGetShaderInfoLog", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        glCreateProgram = bind0(sym, "glCreateProgram", FunctionDescriptor.of(ValueLayout.JAVA_INT));
        glAttachShader = bind0(sym, "glAttachShader", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glLinkProgram = bind0(sym, "glLinkProgram", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glGetProgramiv = bind0(sym, "glGetProgramiv", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glGetProgramInfoLog = bind0(sym, "glGetProgramInfoLog", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        glUseProgram = bind0(sym, "glUseProgram", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glUniform1i = bind0(sym, "glUniform1i", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glUniform1f = bind0(sym, "glUniform1f", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));
        glUniform2f = bind0(sym, "glUniform2f", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));
        glUniform4fv = bind0(sym, "glUniform4fv", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glGetUniformLocation = bind0(sym, "glGetUniformLocation", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glGenTextures = bind0(sym, "glGenTextures", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glBindTexture = bind0(sym, "glBindTexture", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glTexImage2D = bind0(sym, "glTexImage2D", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glTexSubImage2D = bind0(sym, "glTexSubImage2D", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glTexParameteri = bind0(sym, "glTexParameteri", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glActiveTexture = bind0(sym, "glActiveTexture", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glGenFramebuffers = bind0(sym, "glGenFramebuffers", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glBindFramebuffer = bind0(sym, "glBindFramebuffer", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glFramebufferTexture2D = bind0(sym, "glFramebufferTexture2D", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glCheckFramebufferStatus = bind0(sym, "glCheckFramebufferStatus", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));
        glGenVertexArrays = bind0(sym, "glGenVertexArrays", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glBindVertexArray = bind0(sym, "glBindVertexArray", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glDrawArrays = bind0(sym, "glDrawArrays", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glViewport = bind0(sym, "glViewport", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glReadPixels = bind0(sym, "glReadPixels", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glGetError = bind0(sym, "glGetError", FunctionDescriptor.of(ValueLayout.JAVA_INT));
        glFinish = bind0(sym, "glFinish", FunctionDescriptor.ofVoid());
        glGetIntegerv = bind0(sym, "glGetIntegerv", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glDisable = bind0(sym, "glDisable", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glEnable = bind0(sym, "glEnable", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        glIsEnabled = bind0(sym, "glIsEnabled", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        glGetFloatv = bind0(sym, "glGetFloatv", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        glClearColor = bind0(sym, "glClearColor", FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));
        glClear = bind0(sym, "glClear", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
    }

    private MethodHandle bind0(SymbolLookup sym, String name, FunctionDescriptor d) {
        try {
            return linker.downcallHandle(sym.find(name).orElseThrow(), d);
        } catch (Throwable t) {
            throw new IllegalStateException("symbol not found in libmobileglues.so: " + name, t);
        }
    }

    private Object invoke(MethodHandle fn, Object... args) {
        try {
            return fn.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new RuntimeException("GLES call failed: " + t, t);
        }
    }

    private String getString(MethodHandle fn, int pname) {
        try {
            MemorySegment s = (MemorySegment) fn.invokeWithArguments(pname);
            if (s == null || s.equals(MemorySegment.NULL)) return "(null)";
            try {
                s = s.reinterpret(256);
            } catch (Throwable t) {
                return "(reinterpret err)";
            }
            return s.getString(0);
        } catch (Throwable t) {
            return "(err " + t + ")";
        }
    }

    private String infoLog(MethodHandle fn, int obj) {
        try {
            MemorySegment buf = arena.allocate(4096);
            MemorySegment plen = arena.allocate(ValueLayout.JAVA_INT);
            fn.invokeWithArguments(obj, 4096, plen, buf);
            int len = plen.get(ValueLayout.JAVA_INT, 0);
            return buf.reinterpret(Math.max(len, 1)).getString(0);
        } catch (Throwable t) {
            return "(log err " + t + ")";
        }
    }

    /** 从 /proc/self/maps 找 libmobileglues.so 的完整路径（只收 /data/ 开头的真实路径） */
    private static java.util.List<String> findMobileGluesPaths() {
        java.util.List<String> out = new java.util.ArrayList<>();
        try {
            String maps = Files.readString(Path.of("/proc/self/maps"), StandardCharsets.UTF_8);
            for (String line : maps.split("\n")) {
                if (!line.contains("libmobileglues.so")) continue;
                int i = line.lastIndexOf('/');
                if (i < 0) continue;
                String path = line.substring(i).trim();
                int sp = path.indexOf(' ');
                if (sp > 0) path = path.substring(0, sp);
                if (path.endsWith("libmobileglues.so") && path.startsWith("/data/")) {
                    out.add(path);
                }
            }
        } catch (Throwable t) {
            // 忽略
        }
        return out;
    }
}
