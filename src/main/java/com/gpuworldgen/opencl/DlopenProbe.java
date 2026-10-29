package com.gpuworldgen.opencl;

/**
 * OpenCL 库可用性探测。
 * 用 System.load / loadLibrary：JVM 透传 dlopen 的真实错误原因，
 * 可精确区分「库不存在」与「linker namespace 拦截」。
 */
public final class DlopenProbe {

    public record Result(String path, boolean ok, String dlerror) {}

    public Result[] probe(String... paths) {
        Result[] out = new Result[paths.length];
        for (int i = 0; i < paths.length; i++) {
            out[i] = probeOne(paths[i]);
        }
        return out;
    }

    private Result probeOne(String path) {
        boolean absolute = path.startsWith("/");
        try {
            if (absolute) {
                System.load(path);
            } else {
                System.loadLibrary(path);
            }
            return new Result(path, true, "loaded ok");
        } catch (UnsatisfiedLinkError e) {
            String msg = e.getMessage();
            return new Result(path, false, msg == null ? "(no detail)" : msg);
        } catch (Throwable t) {
            return new Result(path, false, "exception: " + t);
        }
    }
}
