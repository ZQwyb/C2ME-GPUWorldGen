package com.gpuworldgen.opencl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * OpenCL 高层封装：context / queue / program / kernel / buffer 生命周期管理。
 */
public final class OpenCLContext implements AutoCloseable {

    private final OpenCLNative cl;
    private final MemorySegment platform;
    private final MemorySegment device;
    private final MemorySegment context;
    private final MemorySegment queue;

    private MemorySegment program;
    private MemorySegment kernel;

    private OpenCLContext(OpenCLNative cl, MemorySegment platform, MemorySegment device,
                          MemorySegment context, MemorySegment queue) {
        this.cl = cl;
        this.platform = platform;
        this.device = device;
        this.context = context;
        this.queue = queue;
    }

    public static OpenCLContext create(OpenCLNative cl, MemorySegment platform, MemorySegment device) {
        int[] err = new int[1];
        MemorySegment context = cl.createContext(device, err);
        if (err[0] != OpenCLNative.CL_SUCCESS) {
            throw new IllegalStateException("clCreateContext failed: error " + err[0]);
        }
        MemorySegment queue = cl.createCommandQueue(context, device, err);
        if (err[0] != OpenCLNative.CL_SUCCESS) {
            cl.releaseContext(context);
            throw new IllegalStateException("clCreateCommandQueue failed: error " + err[0]);
        }
        return new OpenCLContext(cl, platform, device, context, queue);
    }

    public MemorySegment device() { return device; }
    public MemorySegment context() { return context; }
    public MemorySegment queue() { return queue; }

    /** 编译 kernel 源码；失败时抛异常并附带 build log */
    public void buildProgramFromSource(String source, String options) {
        int[] err = new int[1];
        MemorySegment prog = cl.createProgramWithSource(context, source, err);
        if (err[0] != OpenCLNative.CL_SUCCESS) {
            throw new IllegalStateException("clCreateProgramWithSource failed: error " + err[0]);
        }
        this.program = prog;

        int buildErr = cl.buildProgram(prog, device, options);
        if (buildErr != OpenCLNative.CL_SUCCESS) {
            String log = cl.getProgramBuildInfo(prog, device, OpenCLNative.CL_PROGRAM_BUILD_LOG);
            String opts = cl.getProgramBuildInfo(prog, device, OpenCLNative.CL_PROGRAM_BUILD_OPTIONS);
            cl.releaseProgram(prog);
            program = null;
            throw new IllegalStateException("clBuildProgram failed: error " + buildErr
                    + "\noptions: " + opts
                    + "\nbuild log:\n" + log);
        }
    }

    public void createKernel(String kernelName) {
        int[] err = new int[1];
        MemorySegment k = cl.createKernel(program, kernelName, err);
        if (err[0] != OpenCLNative.CL_SUCCESS) {
            throw new IllegalStateException("clCreateKernel(" + kernelName + ") failed: error " + err[0]);
        }
        this.kernel = k;
    }

    public void setKernelIntArg(int index, int value) {
        // 栈上分配 int 作为参数指针
        MemorySegment buf = cl.arena().allocate(ValueLayout.JAVA_INT);
        buf.set(ValueLayout.JAVA_INT, 0, value);
        check(cl.setKernelArg(kernel, index, ValueLayout.JAVA_INT.byteSize(), buf), "clSetKernelArg(int)");
    }

    public void setKernelBufferArg(int index, MemorySegment buffer) {
        // 传 buffer 句柄本身（cl_mem 是句柄值）
        MemorySegment h = cl.arena().allocate(ValueLayout.ADDRESS);
        h.set(ValueLayout.ADDRESS, 0, buffer);
        check(cl.setKernelArg(kernel, index, ValueLayout.ADDRESS.byteSize(), h), "clSetKernelArg(cl_mem)");
    }

    public MemorySegment createReadBuffer(long size) {
        int[] err = new int[1];
        MemorySegment mem = cl.createBuffer(context, OpenCLNative.CL_MEM_READ_ONLY, size, err);
        if (err[0] != OpenCLNative.CL_SUCCESS) {
            throw new IllegalStateException("clCreateBuffer(read) failed: error " + err[0]);
        }
        return mem;
    }

    public MemorySegment createWriteBuffer(long size) {
        int[] err = new int[1];
        MemorySegment mem = cl.createBuffer(context, OpenCLNative.CL_MEM_WRITE_ONLY, size, err);
        if (err[0] != OpenCLNative.CL_SUCCESS) {
            throw new IllegalStateException("clCreateBuffer(write) failed: error " + err[0]);
        }
        return mem;
    }

    public void writeBuffer(MemorySegment buffer, MemorySegment hostData) {
        check(cl.enqueueWriteBuffer(queue, buffer, true, 0L, hostData), "clEnqueueWriteBuffer");
    }

    public void readBuffer(MemorySegment buffer, MemorySegment hostData) {
        check(cl.enqueueReadBuffer(queue, buffer, true, 0L, hostData), "clEnqueueReadBuffer");
    }

    /** 一维 dispatch，globalSize 向上取整到工作组倍数 */
    public void dispatch(int globalSize, int workGroupSize) {
        long group = workGroupSize > 0 ? workGroupSize : 64L;
        long global = ((globalSize + group - 1) / group) * group;
        MemorySegment gs = cl.arena().allocate(ValueLayout.JAVA_LONG);
        MemorySegment ls = cl.arena().allocate(ValueLayout.JAVA_LONG);
        gs.set(ValueLayout.JAVA_LONG, 0, global);
        ls.set(ValueLayout.JAVA_LONG, 0, group);
        check(cl.enqueueNDRangeKernel(queue, kernel, 1, gs, ls), "clEnqueueNDRangeKernel");
        check(cl.finish(queue), "clFinish");
    }

    public void releaseBuffer(MemorySegment buffer) {
        cl.releaseMemObject(buffer);
    }

    private static void check(int err, String what) {
        if (err != OpenCLNative.CL_SUCCESS) {
            throw new IllegalStateException(what + " failed with OpenCL error " + err);
        }
    }

    @Override
    public void close() {
        if (kernel != null) { cl.releaseKernel(kernel); kernel = null; }
        if (program != null) { cl.releaseProgram(program); program = null; }
        cl.releaseCommandQueue(queue);
        cl.releaseContext(context);
    }
}
