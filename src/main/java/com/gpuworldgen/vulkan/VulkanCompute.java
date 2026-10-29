package com.gpuworldgen.vulkan;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Vulkan Compute 管线（v0.2）：
 * 用 Adreno 830 的 Vulkan compute 队列跑 double-double 2D Perlin 噪声，
 * 取代被 Android linker namespace 拦截的 OpenCL 路线。
 *
 * 设计要点（针对 v0.1.2 真机问题）：
 * 1. 全程不调用任何 vkDestroy* —— 诊断进程生命周期短，销毁时机曾引发 JVM SIGSEGV，
 *    由进程退出统一回收资源。
 * 2. 字符串读取用受限 readCString —— 修复 v0.1.2 中 getString 越过 256 字节缓冲区的 UB。
 * 3. MemorySegment.ofAddress 属 restricted API：启动时先探测可用性，
 *    不可用则提示加 JVM 参数 --enable-native-access=ALL-UNNAMED 并降级。
 *
 * 布局参照 Vulkan 1.0 头文件（arm64 LP64：指针 8 字节、对齐 8）。
 */
public final class VulkanCompute {

    // ---------- VkStructureType ----------
    private static final int ST_APPLICATION_INFO = 0;
    private static final int ST_INSTANCE_CREATE_INFO = 1;
    private static final int ST_DEVICE_QUEUE_CREATE_INFO = 2;
    private static final int ST_DEVICE_CREATE_INFO = 3;
    private static final int ST_SHADER_MODULE_CREATE_INFO = 5;
    private static final int ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 6;
    private static final int ST_PIPELINE_LAYOUT_CREATE_INFO = 7;
    private static final int ST_COMPUTE_PIPELINE_CREATE_INFO = 10;
    private static final int ST_BUFFER_CREATE_INFO = 11;
    private static final int ST_MEMORY_ALLOCATE_INFO = 16;
    private static final int ST_DESCRIPTOR_POOL_CREATE_INFO = 20;
    private static final int ST_DESCRIPTOR_SET_ALLOCATE_INFO = 21;
    private static final int ST_WRITE_DESCRIPTOR_SET = 22;
    private static final int ST_COMMAND_POOL_CREATE_INFO = 30;
    private static final int ST_COMMAND_BUFFER_ALLOCATE_INFO = 31;
    private static final int ST_COMMAND_BUFFER_BEGIN_INFO = 32;
    private static final int ST_SUBMIT_INFO = 34;
    private static final int ST_PIPELINE_SHADER_STAGE_CREATE_INFO = 48;

    private static final int VK_SUCCESS = 0;
    private static final long VK_API_VERSION_1_0 = 0x400000L;
    private static final int VK_QUEUE_COMPUTE_BIT = 0x8;
    private static final int VK_SHADER_STAGE_COMPUTE_BIT = 0x20;
    private static final int VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 2;
    private static final int VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20;
    private static final int VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x1;
    private static final int VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x2;
    private static final int VK_PIPELINE_BIND_POINT_COMPUTE = 0;
    private static final int VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
    private static final int VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x1;
    private static final int LOCAL_SIZE = 64;

    /** restricted API（MemorySegment.ofAddress + reinterpret）是否可用 */
    private static final boolean RESTRICTED_OK;
    static {
        boolean ok = false;
        try {
            MemorySegment.ofAddress(8).reinterpret(1);
            ok = true;
        } catch (Throwable t) {
            ok = false;
        }
        RESTRICTED_OK = ok;
    }

    // ---------- FFM 基础设施 ----------
    private final Arena arena = Arena.ofShared();
    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup vk;

    // ---------- Vulkan 句柄 ----------
    private MemorySegment instance;
    private MemorySegment physicalDevice;
    private MemorySegment device;
    private MemorySegment queue;
    private MemorySegment shaderModule;
    private MemorySegment descriptorSetLayout;
    private MemorySegment pipelineLayout;
    private MemorySegment pipeline;
    private MemorySegment descriptorPool;
    private MemorySegment descriptorSet;
    private MemorySegment commandPool;
    private MemorySegment commandBuffer;
    private int queueFamilyIndex = -1;

    // ---------- Buffer 管理 ----------
    private static final class Buffer {
        final MemorySegment handle;   // VkBuffer
        final MemorySegment memory;   // VkDeviceMemory
        final long vkSize;            // vkCreateBuffer 的 size（描述符 range 用）
        final long mapSize;           // 内存对象大小（map 全量用）
        MemorySegment mapped;         // map 后的可读段
        Buffer(MemorySegment h, MemorySegment m, long vs, long ms) {
            handle = h; memory = m; vkSize = vs; mapSize = ms;
        }
    }
    private final Buffer[] buffers = new Buffer[3]; // 0=perm, 1=in, 2=out

    // ---------- 函数句柄 ----------
    private final MethodHandle vkCreateInstance;
    private final MethodHandle vkEnumeratePhysicalDevices;
    private final MethodHandle vkGetPhysicalDeviceProperties;
    private final MethodHandle vkGetPhysicalDeviceQueueFamilyProperties;
    private final MethodHandle vkGetPhysicalDeviceMemoryProperties;
    private final MethodHandle vkCreateDevice;
    private final MethodHandle vkGetDeviceQueue;
    private final MethodHandle vkCreateShaderModule;
    private final MethodHandle vkCreateDescriptorSetLayout;
    private final MethodHandle vkCreatePipelineLayout;
    private final MethodHandle vkCreateComputePipelines;
    private final MethodHandle vkCreateBuffer;
    private final MethodHandle vkGetBufferMemoryRequirements;
    private final MethodHandle vkAllocateMemory;
    private final MethodHandle vkBindBufferMemory;
    private final MethodHandle vkMapMemory;
    private final MethodHandle vkCreateDescriptorPool;
    private final MethodHandle vkAllocateDescriptorSets;
    private final MethodHandle vkUpdateDescriptorSets;
    private final MethodHandle vkCreateCommandPool;
    private final MethodHandle vkAllocateCommandBuffers;
    private final MethodHandle vkBeginCommandBuffer;
    private final MethodHandle vkCmdBindPipeline;
    private final MethodHandle vkCmdBindDescriptorSets;
    private final MethodHandle vkCmdPushConstants;
    private final MethodHandle vkCmdDispatch;
    private final MethodHandle vkEndCommandBuffer;
    private final MethodHandle vkQueueSubmit;
    private final MethodHandle vkQueueWaitIdle;

    private VulkanCompute() {
        this.vk = SymbolLookup.libraryLookup("libvulkan.so", arena);

        // VkResult vkCreateInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*)
        vkCreateInstance = bind("vkCreateInstance", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkEnumeratePhysicalDevices(VkInstance, uint32_t*, VkPhysicalDevice*)
        vkEnumeratePhysicalDevices = bind("vkEnumeratePhysicalDevices", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties*)
        vkGetPhysicalDeviceProperties = bind("vkGetPhysicalDeviceProperties", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*)
        vkGetPhysicalDeviceQueueFamilyProperties = bind("vkGetPhysicalDeviceQueueFamilyProperties",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*)
        vkGetPhysicalDeviceMemoryProperties = bind("vkGetPhysicalDeviceMemoryProperties", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo*, const VkAllocationCallbacks*, VkDevice*)
        vkCreateDevice = bind("vkCreateDevice", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue*)
        vkGetDeviceQueue = bind("vkGetDeviceQueue", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // VkResult vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*)
        vkCreateShaderModule = bind("vkCreateShaderModule", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkCreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout*)
        vkCreateDescriptorSetLayout = bind("vkCreateDescriptorSetLayout", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout*)
        vkCreatePipelineLayout = bind("vkCreatePipelineLayout", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkCreateComputePipelines(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline*)
        vkCreateComputePipelines = bind("vkCreateComputePipelines", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkCreateBuffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer*)
        vkCreateBuffer = bind("vkCreateBuffer", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkGetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements*)
        vkGetBufferMemoryRequirements = bind("vkGetBufferMemoryRequirements", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*)
        vkAllocateMemory = bind("vkAllocateMemory", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)
        vkBindBufferMemory = bind("vkBindBufferMemory", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        // VkResult vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void**)
        vkMapMemory = bind("vkMapMemory", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // VkResult vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool*)
        vkCreateDescriptorPool = bind("vkCreateDescriptorPool", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*)
        vkAllocateDescriptorSets = bind("vkAllocateDescriptorSets", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkUpdateDescriptorSets(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const VkCopyDescriptorSet*)
        vkUpdateDescriptorSets = bind("vkUpdateDescriptorSets", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // VkResult vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool*)
        vkCreateCommandPool = bind("vkCreateCommandPool", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*)
        vkAllocateCommandBuffers = bind("vkAllocateCommandBuffers", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
        vkBeginCommandBuffer = bind("vkBeginCommandBuffer", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline)
        vkCmdBindPipeline = bind("vkCmdBindPipeline", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // void vkCmdBindDescriptorSets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*)
        vkCmdBindDescriptorSets = bind("vkCmdBindDescriptorSets", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // void vkCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t, uint32_t, const void*)
        vkCmdPushConstants = bind("vkCmdPushConstants", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // void vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t)
        vkCmdDispatch = bind("vkCmdDispatch", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        // VkResult vkEndCommandBuffer(VkCommandBuffer)
        vkEndCommandBuffer = bind("vkEndCommandBuffer", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS));
        // VkResult vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence)
        vkQueueSubmit = bind("vkQueueSubmit", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkQueueWaitIdle(VkQueue)
        vkQueueWaitIdle = bind("vkQueueWaitIdle", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS));
    }

    private MethodHandle bind(String symbol, FunctionDescriptor d) {
        try {
            return linker.downcallHandle(vk.find(symbol).orElseThrow(), d);
        } catch (Throwable t) {
            throw new IllegalStateException("vulkan symbol not found: " + symbol, t);
        }
    }

    private static Object invoke(MethodHandle fn, Object... args) {
        try {
            return fn.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new IllegalStateException("vulkan call failed: " + t, t);
        }
    }

    private static void check(String what, Object rc) {
        if (rc != null && ((int) rc) != VK_SUCCESS) {
            throw new IllegalStateException(what + " 失败 VkResult=" + rc);
        }
    }

    /** 受限读字符串：最多 maxLen 字节，绝不越界（修复 getString 越界 UB） */
    private static String readCString(MemorySegment seg, long offset, int maxLen) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < maxLen; i++) {
            byte b = seg.get(ValueLayout.JAVA_BYTE, offset + i);
            if (b == 0) break;
            sb.append((char) (b & 0xFF));
        }
        return sb.toString();
    }

    public static boolean restrictedAvailable() {
        return RESTRICTED_OK;
    }

    /** 完整初始化（instance → device → queue → pipeline → buffers），任何一步失败抛异常 */
    public static VulkanCompute create(StringBuilder report, byte[] spirv, int n) {
        VulkanCompute v = new VulkanCompute();
        v.init(report, spirv, n);
        return v;
    }

    // ==================== 初始化 ====================

    private void init(StringBuilder report, byte[] spirv, int n) {
        // --- VkApplicationInfo (48B) ---
        MemorySegment appInfo = arena.allocate(48);
        appInfo.set(ValueLayout.JAVA_INT, 0, ST_APPLICATION_INFO);
        appInfo.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        appInfo.set(ValueLayout.ADDRESS, 16, arena.allocateFrom("gpuworldgen"));
        appInfo.set(ValueLayout.JAVA_INT, 24, 0);
        appInfo.set(ValueLayout.ADDRESS, 32, arena.allocateFrom("gpuworldgen"));
        appInfo.set(ValueLayout.JAVA_INT, 40, 0);
        appInfo.set(ValueLayout.JAVA_INT, 44, (int) VK_API_VERSION_1_0);

        // --- VkInstanceCreateInfo (64B) ---
        MemorySegment createInfo = arena.allocate(64);
        createInfo.set(ValueLayout.JAVA_INT, 0, ST_INSTANCE_CREATE_INFO);
        createInfo.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        createInfo.set(ValueLayout.JAVA_INT, 16, 0);
        createInfo.set(ValueLayout.ADDRESS, 24, appInfo);
        createInfo.set(ValueLayout.JAVA_INT, 32, 0);
        createInfo.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
        createInfo.set(ValueLayout.JAVA_INT, 48, 0);
        createInfo.set(ValueLayout.ADDRESS, 56, MemorySegment.NULL);

        MemorySegment pInstance = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateInstance", invoke(vkCreateInstance, createInfo, MemorySegment.NULL, pInstance));
        instance = pInstance.get(ValueLayout.ADDRESS, 0);
        report.append("    vkCreateInstance 成功。\n");

        // --- 枚举物理设备，取第一个 ---
        MemorySegment pCount = arena.allocate(ValueLayout.JAVA_INT);
        check("vkEnumeratePhysicalDevices", invoke(vkEnumeratePhysicalDevices, instance, pCount, MemorySegment.NULL));
        int count = pCount.get(ValueLayout.JAVA_INT, 0);
        if (count == 0) throw new IllegalStateException("没有 Vulkan 物理设备");
        report.append("    物理设备数: " + count + "\n");
        MemorySegment devices = arena.allocate(ValueLayout.ADDRESS, count);
        check("vkEnumeratePhysicalDevices", invoke(vkEnumeratePhysicalDevices, instance, pCount, devices));
        physicalDevice = devices.get(ValueLayout.ADDRESS, 0);

        // 设备属性：apiVersion@0 driverVersion@4 vendorID@8 deviceID@12 deviceType@16 deviceName@20..276
        MemorySegment props = arena.allocate(320);
        invoke(vkGetPhysicalDeviceProperties, physicalDevice, props);
        String name = readCString(props, 20, 255);
        int apiVersion = props.get(ValueLayout.JAVA_INT, 0);
        int vendorID = props.get(ValueLayout.JAVA_INT, 8);
        int deviceID = props.get(ValueLayout.JAVA_INT, 12);
        report.append("    [设备] 名称: " + name + "\n");
        report.append("        API v" + (apiVersion >> 22) + "." + ((apiVersion >> 12) & 0x3FF) + "."
                + (apiVersion & 0xFFF) + " vendorID=0x" + Integer.toHexString(vendorID)
                + " deviceID=0x" + Integer.toHexString(deviceID) + "\n");

        // --- 队列族：VkQueueFamilyProperties = 24B {queueFlags@0, queueCount@4, timestampValidBits@8, minImageTransferGranularity@12..24}
        MemorySegment pQCount = arena.allocate(ValueLayout.JAVA_INT);
        invoke(vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice, pQCount, MemorySegment.NULL);
        int qCount = pQCount.get(ValueLayout.JAVA_INT, 0);
        MemorySegment qProps = arena.allocate(qCount * 24L);
        invoke(vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice, pQCount, qProps);
        for (int q = 0; q < qCount; q++) {
            int flags = qProps.get(ValueLayout.JAVA_INT, q * 24L);
            int qq = qProps.get(ValueLayout.JAVA_INT, q * 24L + 4);
            report.append("        队列族 #" + q + ": flags=0x" + Integer.toHexString(flags)
                    + " 队列数=" + qq + (qq > 0 ? "" : " (跳过)") + "\n");
            if (queueFamilyIndex < 0 && (flags & VK_QUEUE_COMPUTE_BIT) != 0 && qq > 0) {
                queueFamilyIndex = q;
                report.append("        → 选用 compute 队列族 #" + q + "\n");
            }
        }
        if (queueFamilyIndex < 0) throw new IllegalStateException("未找到 compute 队列族");

        // --- 内存属性（HOST_VISIBLE|HOST_COHERENT 类型选择用）---
        // VkPhysicalDeviceMemoryProperties: memoryTypeCount@0, memoryTypes[32]@4(每项8B: propertyFlags@0,heapIndex@4)
        MemorySegment mp = arena.allocate(520);
        invoke(vkGetPhysicalDeviceMemoryProperties, physicalDevice, mp);

        // --- VkDeviceQueueCreateInfo (40B) ---
        MemorySegment prio = arena.allocate(4);
        prio.set(ValueLayout.JAVA_FLOAT, 0, 1.0f);
        MemorySegment qci = arena.allocate(40);
        qci.set(ValueLayout.JAVA_INT, 0, ST_DEVICE_QUEUE_CREATE_INFO);
        qci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        qci.set(ValueLayout.JAVA_INT, 16, 0);
        qci.set(ValueLayout.JAVA_INT, 20, queueFamilyIndex);
        qci.set(ValueLayout.JAVA_INT, 24, 1);
        qci.set(ValueLayout.ADDRESS, 32, prio);

        // --- VkDeviceCreateInfo (72B) ---
        MemorySegment devCI = arena.allocate(72);
        devCI.set(ValueLayout.JAVA_INT, 0, ST_DEVICE_CREATE_INFO);
        devCI.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        devCI.set(ValueLayout.JAVA_INT, 16, 0);
        devCI.set(ValueLayout.JAVA_INT, 20, 1);
        devCI.set(ValueLayout.ADDRESS, 24, qci);
        devCI.set(ValueLayout.JAVA_INT, 32, 0);
        devCI.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
        devCI.set(ValueLayout.JAVA_INT, 48, 0);
        devCI.set(ValueLayout.ADDRESS, 56, MemorySegment.NULL);
        devCI.set(ValueLayout.ADDRESS, 64, MemorySegment.NULL);

        MemorySegment pDev = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateDevice", invoke(vkCreateDevice, physicalDevice, devCI, MemorySegment.NULL, pDev));
        device = pDev.get(ValueLayout.ADDRESS, 0);
        report.append("    vkCreateDevice 成功。\n");

        // --- 队列 ---
        MemorySegment pQueue = arena.allocate(ValueLayout.ADDRESS);
        invoke(vkGetDeviceQueue, device, queueFamilyIndex, 0, pQueue);
        queue = pQueue.get(ValueLayout.ADDRESS, 0);

        createPipeline(spirv);
        createBuffers(mp, n);
        createDescriptorSet();
        createCommandResources();

        report.append("    Vulkan compute 管线就绪（shader + pipeline + 3 buffers + cmd）。\n");
    }

    // ==================== 管线 ====================

    private void createPipeline(byte[] spirv) {
        // --- VkShaderModuleCreateInfo (40B)：pCode 需 4 字节对齐 ---
        int codeLen = (spirv.length + 3) & ~3;
        MemorySegment code = arena.allocate(codeLen);
        MemorySegment.copy(spirv, 0, code, ValueLayout.JAVA_BYTE, 0, spirv.length);
        MemorySegment smci = arena.allocate(40);
        smci.set(ValueLayout.JAVA_INT, 0, ST_SHADER_MODULE_CREATE_INFO);
        smci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        smci.set(ValueLayout.JAVA_INT, 16, 0);
        smci.set(ValueLayout.JAVA_LONG, 24, spirv.length);
        smci.set(ValueLayout.ADDRESS, 32, code);
        MemorySegment pMod = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateShaderModule", invoke(vkCreateShaderModule, device, smci, MemorySegment.NULL, pMod));
        shaderModule = pMod.get(ValueLayout.ADDRESS, 0);

        // --- VkDescriptorSetLayoutBinding (24B) x3 ---
        MemorySegment bindings = arena.allocate(24L * 3);
        for (int b = 0; b < 3; b++) {
            MemorySegment bd = bindings.asSlice(b * 24L, 24);
            bd.set(ValueLayout.JAVA_INT, 0, b);                // binding
            bd.set(ValueLayout.JAVA_INT, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            bd.set(ValueLayout.JAVA_INT, 8, 1);                // descriptorCount
            bd.set(ValueLayout.JAVA_INT, 12, VK_SHADER_STAGE_COMPUTE_BIT);
            bd.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL);
        }
        // --- VkDescriptorSetLayoutCreateInfo (32B) ---
        MemorySegment dslci = arena.allocate(32);
        dslci.set(ValueLayout.JAVA_INT, 0, ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
        dslci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        dslci.set(ValueLayout.JAVA_INT, 16, 0);
        dslci.set(ValueLayout.JAVA_INT, 20, 3);
        dslci.set(ValueLayout.ADDRESS, 24, bindings);
        MemorySegment pDsl = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateDescriptorSetLayout", invoke(vkCreateDescriptorSetLayout, device, dslci, MemorySegment.NULL, pDsl));
        descriptorSetLayout = pDsl.get(ValueLayout.ADDRESS, 0);

        // --- VkPushConstantRange (12B) + VkPipelineLayoutCreateInfo (48B) ---
        MemorySegment pcr = arena.allocate(12);
        pcr.set(ValueLayout.JAVA_INT, 0, VK_SHADER_STAGE_COMPUTE_BIT);
        pcr.set(ValueLayout.JAVA_INT, 4, 0);
        pcr.set(ValueLayout.JAVA_INT, 8, 8);                   // freq + count
        MemorySegment layoutPtr = arena.allocate(ValueLayout.ADDRESS);
        layoutPtr.set(ValueLayout.ADDRESS, 0, descriptorSetLayout);
        MemorySegment plci = arena.allocate(48);
        plci.set(ValueLayout.JAVA_INT, 0, ST_PIPELINE_LAYOUT_CREATE_INFO);
        plci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        plci.set(ValueLayout.JAVA_INT, 16, 0);
        plci.set(ValueLayout.JAVA_INT, 20, 1);
        plci.set(ValueLayout.ADDRESS, 24, layoutPtr);
        plci.set(ValueLayout.JAVA_INT, 32, 1);
        plci.set(ValueLayout.ADDRESS, 40, pcr);
        MemorySegment pPl = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreatePipelineLayout", invoke(vkCreatePipelineLayout, device, plci, MemorySegment.NULL, pPl));
        pipelineLayout = pPl.get(ValueLayout.ADDRESS, 0);

        // --- VkComputePipelineCreateInfo (96B)，stage 内嵌 48B@24 ---
        MemorySegment pName = arena.allocateFrom("main");
        MemorySegment cpci = arena.allocate(96);
        cpci.set(ValueLayout.JAVA_INT, 0, ST_COMPUTE_PIPELINE_CREATE_INFO);
        cpci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        cpci.set(ValueLayout.JAVA_INT, 16, 0);                 // flags
        cpci.set(ValueLayout.JAVA_INT, 24, ST_PIPELINE_SHADER_STAGE_CREATE_INFO); // stage.sType
        cpci.set(ValueLayout.ADDRESS, 32, MemorySegment.NULL); // stage.pNext
        cpci.set(ValueLayout.JAVA_INT, 40, 0);                 // stage.flags
        cpci.set(ValueLayout.JAVA_INT, 44, VK_SHADER_STAGE_COMPUTE_BIT);
        cpci.set(ValueLayout.ADDRESS, 48, shaderModule);
        cpci.set(ValueLayout.ADDRESS, 56, pName);
        cpci.set(ValueLayout.ADDRESS, 64, MemorySegment.NULL); // pSpecializationInfo
        cpci.set(ValueLayout.ADDRESS, 72, pipelineLayout);
        cpci.set(ValueLayout.ADDRESS, 80, MemorySegment.NULL); // basePipelineHandle
        cpci.set(ValueLayout.JAVA_INT, 88, -1);                // basePipelineIndex
        MemorySegment pPipe = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateComputePipelines",
                invoke(vkCreateComputePipelines, device, MemorySegment.NULL, 1, cpci, MemorySegment.NULL, pPipe));
        pipeline = pPipe.get(ValueLayout.ADDRESS, 0);
    }

    // ==================== Buffers ====================

    private void createBuffers(MemorySegment memProps, int n) {
        buffers[0] = createBuffer(memProps, 512L * 4);       // perm
        buffers[1] = createBuffer(memProps, n * 16L);        // in
        buffers[2] = createBuffer(memProps, n * 8L);         // out
    }

    /** 创建 buffer + 内存 + map */
    private Buffer createBuffer(MemorySegment memProps, long size) {
        // --- VkBufferCreateInfo (56B) ---
        MemorySegment bci = arena.allocate(56);
        bci.set(ValueLayout.JAVA_INT, 0, ST_BUFFER_CREATE_INFO);
        bci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        bci.set(ValueLayout.JAVA_INT, 16, 0);                  // flags
        bci.set(ValueLayout.JAVA_LONG, 24, size);
        bci.set(ValueLayout.JAVA_INT, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        bci.set(ValueLayout.JAVA_INT, 36, 0);                  // sharingMode EXCLUSIVE
        bci.set(ValueLayout.JAVA_INT, 40, 0);                  // queueFamilyIndexCount
        bci.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL);
        MemorySegment pBuf = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateBuffer", invoke(vkCreateBuffer, device, bci, MemorySegment.NULL, pBuf));
        MemorySegment handle = pBuf.get(ValueLayout.ADDRESS, 0);

        // --- VkMemoryRequirements (24B) ---
        MemorySegment req = arena.allocate(24);
        invoke(vkGetBufferMemoryRequirements, device, handle, req);
        long reqSize = req.get(ValueLayout.JAVA_LONG, 0);
        int memoryTypeBits = req.get(ValueLayout.JAVA_INT, 16);

        // --- 选 HOST_VISIBLE | HOST_COHERENT 内存类型 ---
        int nTypes = memProps.get(ValueLayout.JAVA_INT, 0);
        int chosenType = -1;
        for (int i = 0; i < nTypes; i++) {
            int flags = memProps.get(ValueLayout.JAVA_INT, 4 + i * 8L);
            if ((memoryTypeBits & (1L << i)) != 0
                    && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0
                    && (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
                chosenType = i;
                break;
            }
        }
        if (chosenType < 0) throw new IllegalStateException("没有 HOST_VISIBLE|HOST_COHERENT 内存类型");

        // --- VkMemoryAllocateInfo (32B) ---
        MemorySegment ma = arena.allocate(32);
        ma.set(ValueLayout.JAVA_INT, 0, ST_MEMORY_ALLOCATE_INFO);
        ma.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        ma.set(ValueLayout.JAVA_LONG, 16, reqSize);
        ma.set(ValueLayout.JAVA_INT, 24, chosenType);
        MemorySegment pMem = arena.allocate(ValueLayout.ADDRESS);
        check("vkAllocateMemory", invoke(vkAllocateMemory, device, ma, MemorySegment.NULL, pMem));
        MemorySegment mem = pMem.get(ValueLayout.ADDRESS, 0);

        check("vkBindBufferMemory", invoke(vkBindBufferMemory, device, handle, mem, 0L));

        // --- map 全量（保持到进程结束，不 unmap）---
        MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
        check("vkMapMemory", invoke(vkMapMemory, device, mem, 0L, reqSize, 0, pp));
        long ptr = pp.get(ValueLayout.ADDRESS, 0).address();
        MemorySegment mapped = MemorySegment.ofAddress(ptr).reinterpret(reqSize);

        Buffer b = new Buffer(handle, mem, size, reqSize);
        b.mapped = mapped;
        return b;
    }

    /** run 前按 n 确保 in/out buffer 容量（默认按启动时 n 已建好，通常直接返回） */
    private void resizeDataBuffers(int n) {
        long inSize = n * 16L;
        long outSize = n * 8L;
        if (buffers[1] != null && buffers[1].vkSize >= inSize
                && buffers[2] != null && buffers[2].vkSize >= outSize) {
            return; // 已够大
        }
        if (buffers[1] != null) buffers[1] = null;
        if (buffers[2] != null) buffers[2] = null;
        MemorySegment mp = arena.allocate(520);
        invoke(vkGetPhysicalDeviceMemoryProperties, physicalDevice, mp);
        buffers[1] = createBuffer(mp, inSize);
        buffers[2] = createBuffer(mp, outSize);
        updateDescriptorSetWrites();
    }

    // ==================== Descriptor ====================

    private void createDescriptorSet() {
        // --- VkDescriptorPoolSize (8B) + VkDescriptorPoolCreateInfo (40B) ---
        MemorySegment ps = arena.allocate(8);
        ps.set(ValueLayout.JAVA_INT, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        ps.set(ValueLayout.JAVA_INT, 4, 3);
        MemorySegment dpci = arena.allocate(40);
        dpci.set(ValueLayout.JAVA_INT, 0, ST_DESCRIPTOR_POOL_CREATE_INFO);
        dpci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        dpci.set(ValueLayout.JAVA_INT, 16, 0);
        dpci.set(ValueLayout.JAVA_INT, 20, 1);                 // maxSets
        dpci.set(ValueLayout.JAVA_INT, 24, 1);                 // poolSizeCount
        dpci.set(ValueLayout.ADDRESS, 32, ps);
        MemorySegment pPool = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateDescriptorPool", invoke(vkCreateDescriptorPool, device, dpci, MemorySegment.NULL, pPool));
        descriptorPool = pPool.get(ValueLayout.ADDRESS, 0);

        // --- VkDescriptorSetAllocateInfo (40B) ---
        MemorySegment layoutPtr = arena.allocate(ValueLayout.ADDRESS);
        layoutPtr.set(ValueLayout.ADDRESS, 0, descriptorSetLayout);
        MemorySegment dsai = arena.allocate(40);
        dsai.set(ValueLayout.JAVA_INT, 0, ST_DESCRIPTOR_SET_ALLOCATE_INFO);
        dsai.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        dsai.set(ValueLayout.ADDRESS, 16, descriptorPool);
        dsai.set(ValueLayout.JAVA_INT, 24, 1);
        dsai.set(ValueLayout.ADDRESS, 32, layoutPtr);
        MemorySegment pSet = arena.allocate(ValueLayout.ADDRESS);
        check("vkAllocateDescriptorSets", invoke(vkAllocateDescriptorSets, device, dsai, pSet));
        descriptorSet = pSet.get(ValueLayout.ADDRESS, 0);

        updateDescriptorSetWrites();
    }

    private void updateDescriptorSetWrites() {
        // --- VkWriteDescriptorSet (64B) x3 + VkDescriptorBufferInfo (24B) x3 ---
        MemorySegment writes = arena.allocate(64L * 3);
        MemorySegment dbis = arena.allocate(24L * 3);
        for (int b = 0; b < 3; b++) {
            MemorySegment w = writes.asSlice(b * 64L, 64);
            w.set(ValueLayout.JAVA_INT, 0, ST_WRITE_DESCRIPTOR_SET);
            w.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
            w.set(ValueLayout.ADDRESS, 16, descriptorSet);
            w.set(ValueLayout.JAVA_INT, 24, b);
            w.set(ValueLayout.JAVA_INT, 28, 0);
            w.set(ValueLayout.JAVA_INT, 32, 1);
            w.set(ValueLayout.JAVA_INT, 36, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
            MemorySegment dbi = dbis.asSlice(b * 24L, 24);
            dbi.set(ValueLayout.ADDRESS, 0, buffers[b].handle);
            dbi.set(ValueLayout.JAVA_LONG, 8, 0);
            dbi.set(ValueLayout.JAVA_LONG, 16, buffers[b].vkSize);
            w.set(ValueLayout.ADDRESS, 48, dbi);
            w.set(ValueLayout.ADDRESS, 56, MemorySegment.NULL);
        }
        invoke(vkUpdateDescriptorSets, device, 3, writes, 0, MemorySegment.NULL);
    }

    // ==================== Command ====================

    private void createCommandResources() {
        // --- VkCommandPoolCreateInfo (24B) ---
        MemorySegment cpci = arena.allocate(24);
        cpci.set(ValueLayout.JAVA_INT, 0, ST_COMMAND_POOL_CREATE_INFO);
        cpci.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        cpci.set(ValueLayout.JAVA_INT, 16, 0);
        cpci.set(ValueLayout.JAVA_INT, 20, queueFamilyIndex);
        MemorySegment pPool = arena.allocate(ValueLayout.ADDRESS);
        check("vkCreateCommandPool", invoke(vkCreateCommandPool, device, cpci, MemorySegment.NULL, pPool));
        commandPool = pPool.get(ValueLayout.ADDRESS, 0);

        // --- VkCommandBufferAllocateInfo (32B) ---
        MemorySegment cbai = arena.allocate(32);
        cbai.set(ValueLayout.JAVA_INT, 0, ST_COMMAND_BUFFER_ALLOCATE_INFO);
        cbai.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        cbai.set(ValueLayout.ADDRESS, 16, commandPool);
        cbai.set(ValueLayout.JAVA_INT, 24, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        cbai.set(ValueLayout.JAVA_INT, 28, 1);
        MemorySegment pCb = arena.allocate(ValueLayout.ADDRESS);
        check("vkAllocateCommandBuffers", invoke(vkAllocateCommandBuffers, device, cbai, pCb));
        commandBuffer = pCb.get(ValueLayout.ADDRESS, 0);
    }

    // ==================== 执行 ====================

    /**
     * 跑一次完整计算：
     * @param perm  512 个 int 的排列表
     * @param in    n*4 个 float（x_hi x_lo y_hi y_lo）
     * @param out   n*2 个 float（result_hi result_lo）写回
     */
    public void run(int n, int freq, MemorySegment perm, MemorySegment in, MemorySegment out) {
        resizeDataBuffers(n);

        // 1) host 写数据（coherent 内存，无需 flush）
        MemorySegment.copy(perm, 0, buffers[0].mapped, 0, 512L * 4);
        MemorySegment.copy(in, 0, buffers[1].mapped, 0, n * 16L);

        // 2) record 命令
        MemorySegment cbbi = arena.allocate(32);
        cbbi.set(ValueLayout.JAVA_INT, 0, ST_COMMAND_BUFFER_BEGIN_INFO);
        cbbi.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        cbbi.set(ValueLayout.JAVA_INT, 16, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        cbbi.set(ValueLayout.ADDRESS, 24, MemorySegment.NULL);
        check("vkBeginCommandBuffer", invoke(vkBeginCommandBuffer, commandBuffer, cbbi));

        invoke(vkCmdBindPipeline, commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        MemorySegment setPtr = arena.allocate(ValueLayout.ADDRESS);
        setPtr.set(ValueLayout.ADDRESS, 0, descriptorSet);
        invoke(vkCmdBindDescriptorSets, commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                0, 1, setPtr, 0, MemorySegment.NULL);

        MemorySegment pc = arena.allocate(8);
        pc.set(ValueLayout.JAVA_INT, 0, freq);
        pc.set(ValueLayout.JAVA_INT, 4, n);
        invoke(vkCmdPushConstants, commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);

        int groups = (n + LOCAL_SIZE - 1) / LOCAL_SIZE;
        invoke(vkCmdDispatch, commandBuffer, groups, 1, 1);
        check("vkEndCommandBuffer", invoke(vkEndCommandBuffer, commandBuffer));

        // 3) submit + wait
        MemorySegment cmdPtr = arena.allocate(ValueLayout.ADDRESS);
        cmdPtr.set(ValueLayout.ADDRESS, 0, commandBuffer);
        MemorySegment sub = arena.allocate(72);
        sub.set(ValueLayout.JAVA_INT, 0, ST_SUBMIT_INFO);
        sub.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        sub.set(ValueLayout.JAVA_INT, 16, 0);
        sub.set(ValueLayout.ADDRESS, 24, MemorySegment.NULL);
        sub.set(ValueLayout.ADDRESS, 32, MemorySegment.NULL);
        sub.set(ValueLayout.JAVA_INT, 40, 1);
        sub.set(ValueLayout.ADDRESS, 48, cmdPtr);
        sub.set(ValueLayout.JAVA_INT, 56, 0);
        sub.set(ValueLayout.ADDRESS, 64, MemorySegment.NULL);
        check("vkQueueSubmit", invoke(vkQueueSubmit, queue, 1, sub, MemorySegment.NULL));
        check("vkQueueWaitIdle", invoke(vkQueueWaitIdle, queue));

        // 4) 读回（coherent，wait 后直接可见）
        MemorySegment.copy(buffers[2].mapped, 0, out, 0, n * 8L);
    }

    /** perm buffer 访问（供诊断使用） */
    public MemorySegment permMapped() {
        return buffers[0].mapped;
    }
}
