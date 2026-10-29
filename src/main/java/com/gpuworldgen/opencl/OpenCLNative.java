package com.gpuworldgen.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * OpenCL 1.2 core binding via Java FFM (Java 25, stable API).
 *
 * 设计说明：
 * - MC 26.3 运行在 Java 25 上，FFM 是标准 API，无需任何额外 native 库。
 * - 通过 SymbolLookup.libraryLookup("libOpenCL.so", ...) 直接 dlopen Android 系统 OpenCL 驱动。
 * - 完全绕开 LWJGL（MC/Pojav 的 LWJGL 不包含 opencl 模块，且 Android 上没有对应 native）。
 *
 * 类型映射（arm64, LP64）：
 *   cl_int / cl_uint            -> int (32-bit)
 *   size_t                      -> long (64-bit)
 *   cl_mem_flags/cl_device_type -> long (64-bit)
 *   各种句柄 (cl_platform_id 等) -> MemorySegment (ADDRESS)
 */
public final class OpenCLNative implements AutoCloseable {

    // ---- 常量（Khronos cl.h / cl_platform.h, OpenCL 1.2）----
    public static final int CL_PLATFORM_PROFILE = 0x0900;
    public static final int CL_PLATFORM_VERSION = 0x0901;
    public static final int CL_PLATFORM_NAME = 0x0902;
    public static final int CL_PLATFORM_VENDOR = 0x0903;
    public static final int CL_PLATFORM_EXTENSIONS = 0x0904;

    public static final long CL_DEVICE_TYPE_CPU = 1L << 1;
    public static final long CL_DEVICE_TYPE_GPU = 1L << 2;
    public static final long CL_DEVICE_TYPE_ALL = 0xFFFFFFFFL;

    public static final int CL_DEVICE_TYPE = 0x1000;
    public static final int CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
    public static final int CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
    public static final int CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
    public static final int CL_DEVICE_NAME = 0x102B;
    public static final int CL_DEVICE_VENDOR = 0x102C;
    public static final int CL_DEVICE_DRIVER_VERSION = 0x102D;
    public static final int CL_DEVICE_PROFILE = 0x102E;
    public static final int CL_DEVICE_VERSION = 0x102F;
    public static final int CL_DEVICE_EXTENSIONS = 0x1030;
    public static final int CL_DEVICE_PLATFORM = 0x1031;
    public static final int CL_DEVICE_OPENCL_C_VERSION = 0x103D;

    public static final long CL_CONTEXT_PLATFORM = 0x1084L;

    public static final long CL_MEM_READ_WRITE = 1L << 0;
    public static final long CL_MEM_READ_ONLY = 1L << 1;
    public static final long CL_MEM_WRITE_ONLY = 1L << 2;

    public static final int CL_PROGRAM_BUILD_STATUS = 0x1181;
    public static final int CL_PROGRAM_BUILD_OPTIONS = 0x1182;
    public static final int CL_PROGRAM_BUILD_LOG = 0x1183;

    public static final int CL_SUCCESS = 0;

    private final Arena arena;
    private final Linker linker;
    private final SymbolLookup lookup;

    // ---- downcall handles ----
    private final MethodHandle getPlatformIDs;
    private final MethodHandle getPlatformInfo;
    private final MethodHandle getDeviceIDs;
    private final MethodHandle getDeviceInfo;
    private final MethodHandle createContext;
    private final MethodHandle createCommandQueue;
    private final MethodHandle createProgramWithSource;
    private final MethodHandle buildProgram;
    private final MethodHandle getProgramBuildInfo;
    private final MethodHandle createKernel;
    private final MethodHandle setKernelArg;
    private final MethodHandle createBuffer;
    private final MethodHandle enqueueWriteBuffer;
    private final MethodHandle enqueueReadBuffer;
    private final MethodHandle enqueueNDRangeKernel;
    private final MethodHandle finish;
    private final MethodHandle releaseMemObject;
    private final MethodHandle releaseKernel;
    private final MethodHandle releaseProgram;
    private final MethodHandle releaseCommandQueue;
    private final MethodHandle releaseContext;

    /** 共享 arena：mod 生命周期内有效 */
    public OpenCLNative(String libraryName) {
        this.arena = Arena.ofShared();
        this.linker = Linker.nativeLinker();
        this.lookup = SymbolLookup.libraryLookup(libraryName, arena);

        this.getPlatformIDs = bind("clGetPlatformIDs",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.getPlatformInfo = bind("clGetPlatformInfo",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.getDeviceIDs = bind("clGetDeviceIDs",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.getDeviceInfo = bind("clGetDeviceInfo",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.createContext = bind("clCreateContext",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.createCommandQueue = bind("clCreateCommandQueue",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
        this.createProgramWithSource = bind("clCreateProgramWithSource",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.buildProgram = bind("clBuildProgram",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.getProgramBuildInfo = bind("clGetProgramBuildInfo",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.createKernel = bind("clCreateKernel",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.setKernelArg = bind("clSetKernelArg",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
        this.createBuffer = bind("clCreateBuffer",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.enqueueWriteBuffer = bind("clEnqueueWriteBuffer",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.enqueueReadBuffer = bind("clEnqueueReadBuffer",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.enqueueNDRangeKernel = bind("clEnqueueNDRangeKernel",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.finish = bind("clFinish",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        this.releaseMemObject = bind("clReleaseMemObject",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        this.releaseKernel = bind("clReleaseKernel",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        this.releaseProgram = bind("clReleaseProgram",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        this.releaseCommandQueue = bind("clReleaseCommandQueue",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        this.releaseContext = bind("clReleaseContext",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
    }

    private MethodHandle bind(String symbol, FunctionDescriptor descriptor) {
        try {
            MemorySegment address = lookup.find(symbol)
                    .orElseThrow(() -> new IllegalStateException("OpenCL symbol not found: " + symbol));
            return linker.downcallHandle(address, descriptor);
        } catch (Throwable t) {
            throw new IllegalStateException("Failed to bind OpenCL symbol " + symbol + ": " + t, t);
        }
    }

    public Arena arena() { return arena; }

    // ================= 平台 =================

    public MemorySegment[] getPlatforms() {
        MemorySegment num = arena.allocate(ValueLayout.JAVA_INT);
        check(intResult(getPlatformIDs, 0, MemorySegment.NULL, num), "clGetPlatformIDs(discover)");
        int count = num.get(ValueLayout.JAVA_INT, 0);
        if (count == 0) return new MemorySegment[0];
        MemorySegment platforms = arena.allocate(ValueLayout.ADDRESS, count);
        check(intResult(getPlatformIDs, count, platforms, MemorySegment.NULL), "clGetPlatformIDs(list)");
        MemorySegment[] result = new MemorySegment[count];
        for (int i = 0; i < count; i++) {
            result[i] = platforms.get(ValueLayout.ADDRESS, i * ValueLayout.ADDRESS.byteSize());
        }
        return result;
    }

    // ================= 信息查询 =================

    public String getPlatformInfo(MemorySegment platform, int param) {
        return readInfoString(getPlatformInfo, platform, param);
    }

    public String getDeviceInfoString(MemorySegment device, int param) {
        return readInfoString(getDeviceInfo, device, param);
    }

    public long getDeviceInfoLong(MemorySegment device, int param) {
        MemorySegment buf = arena.allocate(ValueLayout.JAVA_LONG);
        check(intResult(getDeviceInfo, device, param, ValueLayout.JAVA_LONG.byteSize(), buf, MemorySegment.NULL), "clGetDeviceInfo(long)");
        return buf.get(ValueLayout.JAVA_LONG, 0);
    }

    private String readInfoString(MethodHandle fn, MemorySegment handle, int param) {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        check(intResult(fn, handle, param, 0L, MemorySegment.NULL, sizeBuf), "info size query");
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        if (size <= 0) return "";
        MemorySegment buf = arena.allocate(size);
        check(intResult(fn, handle, param, size, buf, MemorySegment.NULL), "info value query");
        return buf.getString(0);
    }

    // ================= 设备 =================

    /** 返回设备句柄数组 */
    public MemorySegment[] getDeviceIDs(MemorySegment platform, long deviceType) {
        MemorySegment num = arena.allocate(ValueLayout.JAVA_INT);
        check(intResult(getDeviceIDs, platform, deviceType, 0, MemorySegment.NULL, num), "clGetDeviceIDs(discover)");
        int count = num.get(ValueLayout.JAVA_INT, 0);
        if (count == 0) return new MemorySegment[0];
        MemorySegment devices = arena.allocate(ValueLayout.ADDRESS, count);
        check(intResult(getDeviceIDs, platform, deviceType, count, devices, MemorySegment.NULL), "clGetDeviceIDs(list)");
        MemorySegment[] result = new MemorySegment[count];
        for (int i = 0; i < count; i++) {
            result[i] = devices.get(ValueLayout.ADDRESS, i * ValueLayout.ADDRESS.byteSize());
        }
        return result;
    }

    // ================= 上下文 / 队列 =================

    public MemorySegment createContext(MemorySegment device, int[] errcode) {
        // properties 传 NULL：驱动按设备推断所属平台（避免句柄地址数值转换）
        MemorySegment devices = arena.allocate(ValueLayout.ADDRESS, 1);
        devices.set(ValueLayout.ADDRESS, 0, device);

        MemorySegment err = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment ctx = (MemorySegment) invoke(createContext, MemorySegment.NULL, 1, devices,
                MemorySegment.NULL, MemorySegment.NULL, err);
        if (errcode.length > 0) errcode[0] = err.get(ValueLayout.JAVA_INT, 0);
        return ctx;
    }

    public MemorySegment createCommandQueue(MemorySegment context, MemorySegment device, int[] errcode) {
        MemorySegment err = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment queue = (MemorySegment) invoke(createCommandQueue, context, device, 0L, err);
        if (errcode.length > 0) errcode[0] = err.get(ValueLayout.JAVA_INT, 0);
        return queue;
    }

    // ================= 程序 / kernel =================

    public MemorySegment createProgramWithSource(MemorySegment context, String source, int[] errcode) {
        MemorySegment src = arena.allocateFrom(source);
        MemorySegment strings = arena.allocate(ValueLayout.ADDRESS, 1);
        strings.set(ValueLayout.ADDRESS, 0, src);
        MemorySegment err = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment program = (MemorySegment) invoke(createProgramWithSource, context, 1, strings, MemorySegment.NULL, err);
        if (errcode.length > 0) errcode[0] = err.get(ValueLayout.JAVA_INT, 0);
        return program;
    }

    public int buildProgram(MemorySegment program, MemorySegment device, String options) {
        MemorySegment opt = (options == null || options.isEmpty()) ? MemorySegment.NULL : arena.allocateFrom(options);
        return intResult(buildProgram, program, 1, device, opt, MemorySegment.NULL, MemorySegment.NULL);
    }

    public String getProgramBuildInfo(MemorySegment program, MemorySegment device, int param) {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        check(intResult(getProgramBuildInfo, program, device, param, 0L, MemorySegment.NULL, sizeBuf), "program info size");
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        if (size <= 0) return "";
        MemorySegment buf = arena.allocate(size);
        check(intResult(getProgramBuildInfo, program, device, param, size, buf, MemorySegment.NULL), "program info value");
        return buf.getString(0);
    }

    public MemorySegment createKernel(MemorySegment program, String kernelName, int[] errcode) {
        MemorySegment name = arena.allocateFrom(kernelName);
        MemorySegment err = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment kernel = (MemorySegment) invoke(createKernel, program, name, err);
        if (errcode.length > 0) errcode[0] = err.get(ValueLayout.JAVA_INT, 0);
        return kernel;
    }

    public int setKernelArg(MemorySegment kernel, int index, long size, MemorySegment value) {
        return intResult(setKernelArg, kernel, index, size, value == null ? MemorySegment.NULL : value);
    }

    // ================= 内存 =================

    public MemorySegment createBuffer(MemorySegment context, long flags, long size, int[] errcode) {
        MemorySegment err = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment mem = (MemorySegment) invoke(createBuffer, context, flags, size, MemorySegment.NULL, err);
        if (errcode.length > 0) errcode[0] = err.get(ValueLayout.JAVA_INT, 0);
        return mem;
    }

    public int enqueueWriteBuffer(MemorySegment queue, MemorySegment mem, boolean blocking, long offset, MemorySegment src) {
        return intResult(enqueueWriteBuffer, queue, mem, blocking ? 1 : 0, offset, src.byteSize(), src, 0, MemorySegment.NULL, MemorySegment.NULL);
    }

    public int enqueueReadBuffer(MemorySegment queue, MemorySegment mem, boolean blocking, long offset, MemorySegment dst) {
        return intResult(enqueueReadBuffer, queue, mem, blocking ? 1 : 0, offset, dst.byteSize(), dst, 0, MemorySegment.NULL, MemorySegment.NULL);
    }

    public int enqueueNDRangeKernel(MemorySegment queue, MemorySegment kernel, int workDim,
                                    MemorySegment globalWorkSize, MemorySegment localWorkSize) {
        return intResult(enqueueNDRangeKernel, queue, kernel, workDim,
                MemorySegment.NULL, globalWorkSize,
                localWorkSize == null ? MemorySegment.NULL : localWorkSize,
                0, MemorySegment.NULL, MemorySegment.NULL);
    }

    public int finish(MemorySegment queue) {
        return intResult(finish, queue);
    }

    // ================= 释放 =================

    public void releaseMemObject(MemorySegment mem) { intResult(releaseMemObject, mem); }
    public void releaseKernel(MemorySegment kernel) { intResult(releaseKernel, kernel); }
    public void releaseProgram(MemorySegment program) { intResult(releaseProgram, program); }
    public void releaseCommandQueue(MemorySegment queue) { intResult(releaseCommandQueue, queue); }
    public void releaseContext(MemorySegment context) { intResult(releaseContext, context); }

    // ================= 工具 =================

    private Object invoke(MethodHandle fn, Object... args) {
        try {
            return fn.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenCL call failed: " + t, t);
        }
    }

    private int intResult(MethodHandle fn, Object... args) {
        return (int) invoke(fn, args);
    }

    private static void check(int err, String what) {
        if (err != CL_SUCCESS) {
            throw new IllegalStateException(what + " failed with OpenCL error " + err);
        }
    }

    /** 检测设备是否支持 FP64（用于诊断，本 mod 不依赖它） */
    public boolean supportsDoublePrecision(MemorySegment device) {
        String extensions = getDeviceInfoString(device, CL_DEVICE_EXTENSIONS);
        return extensions != null && extensions.contains("cl_khr_fp64");
    }

    @Override
    public void close() {
        // 句柄由调用方按需 release；arena 生命周期与 mod 一致。
    }
}
