package com.gpuworldgen.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Vulkan 能力探测（v0.1.2，修正函数签名与结构体布局）：
 * 1. dlopen libvulkan.so（公开 NDK 库，Android 上必然可加载）
 * 2. vkCreateInstance（Vulkan 1.0）
 * 3. 枚举物理设备 → 名称 / vendorID / deviceID / 类型 / compute 队列族 / 扩展
 *
 * 这是 v0.2 切换 Vulkan Compute 路线的依据。
 */
public final class VulkanDetector {

    private static final int VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
    private static final int VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
    private static final int VK_SUCCESS = 0;
    private static final long VK_API_VERSION_1_0 = 0x400000L; // VK_MAKE_API_VERSION(0,1,0,0)
    private static final int VK_QUEUE_COMPUTE_BIT = 0x8;

    private final Arena arena;
    private final Linker linker;
    private final SymbolLookup vulkan;

    private final MethodHandle vkCreateInstance;
    private final MethodHandle vkDestroyInstance;
    private final MethodHandle vkEnumeratePhysicalDevices;
    private final MethodHandle vkGetPhysicalDeviceProperties;
    private final MethodHandle vkEnumerateDeviceExtensionProperties;
    private final MethodHandle vkGetPhysicalDeviceQueueFamilyProperties;

    public VulkanDetector(String libraryName) {
        this.arena = Arena.ofShared();
        this.linker = Linker.nativeLinker();
        this.vulkan = SymbolLookup.libraryLookup(libraryName, arena);

        // VkResult vkCreateInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*)
        this.vkCreateInstance = bind("vkCreateInstance",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkDestroyInstance(VkInstance, const VkAllocationCallbacks*)
        this.vkDestroyInstance = bind("vkDestroyInstance",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkEnumeratePhysicalDevices(VkInstance, uint32_t*, VkPhysicalDevice*)
        this.vkEnumeratePhysicalDevices = bind("vkEnumeratePhysicalDevices",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties*)
        this.vkGetPhysicalDeviceProperties = bind("vkGetPhysicalDeviceProperties",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // VkResult vkEnumerateDeviceExtensionProperties(VkPhysicalDevice, const char*, uint32_t*, VkExtensionProperties*)
        this.vkEnumerateDeviceExtensionProperties = bind("vkEnumerateDeviceExtensionProperties",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // void vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*)
        this.vkGetPhysicalDeviceQueueFamilyProperties = bind("vkGetPhysicalDeviceQueueFamilyProperties",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    }

    private MethodHandle bind(String symbol, FunctionDescriptor d) {
        try {
            return linker.downcallHandle(vulkan.find(symbol).orElseThrow(), d);
        } catch (Throwable t) {
            throw new IllegalStateException("vulkan symbol not found: " + symbol, t);
        }
    }

    private Object invoke(MethodHandle fn, Object... args) {
        try {
            return fn.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new IllegalStateException("vulkan call failed: " + t, t);
        }
    }

    /** 跑完整探测，结果追加到 report */
    public void detect(StringBuilder report) {
        // --- VkApplicationInfo (48 bytes) ---
        MemorySegment appInfo = arena.allocate(48);
        appInfo.set(ValueLayout.JAVA_INT, 0, VK_STRUCTURE_TYPE_APPLICATION_INFO);
        appInfo.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        appInfo.set(ValueLayout.ADDRESS, 16, arena.allocateFrom("gpuworldgen"));
        appInfo.set(ValueLayout.JAVA_INT, 24, 0);
        appInfo.set(ValueLayout.ADDRESS, 32, arena.allocateFrom("gpuworldgen"));
        appInfo.set(ValueLayout.JAVA_INT, 40, 0);
        appInfo.set(ValueLayout.JAVA_INT, 44, (int) VK_API_VERSION_1_0);

        // --- VkInstanceCreateInfo (64 bytes) ---
        MemorySegment createInfo = arena.allocate(64);
        createInfo.set(ValueLayout.JAVA_INT, 0, VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
        createInfo.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL);
        createInfo.set(ValueLayout.JAVA_INT, 16, 0);
        createInfo.set(ValueLayout.ADDRESS, 24, appInfo);
        createInfo.set(ValueLayout.JAVA_INT, 32, 0);
        createInfo.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
        createInfo.set(ValueLayout.JAVA_INT, 48, 0);
        createInfo.set(ValueLayout.ADDRESS, 56, MemorySegment.NULL);

        MemorySegment pInstance = arena.allocate(ValueLayout.ADDRESS);
        int rc = (int) invoke(vkCreateInstance, createInfo, MemorySegment.NULL, pInstance);
        if (rc != VK_SUCCESS) {
            report.append("    vkCreateInstance 失败: VkResult=" + rc + "（Vulkan 1.0 instance 建不起来，路线不可行）\n");
            return;
        }
        MemorySegment instance = pInstance.get(ValueLayout.ADDRESS, 0);
        report.append("    vkCreateInstance 成功。\n");

        try {
            // --- 枚举物理设备 ---
            MemorySegment pCount = arena.allocate(ValueLayout.JAVA_INT);
            invoke(vkEnumeratePhysicalDevices, instance, pCount, MemorySegment.NULL);
            int count = pCount.get(ValueLayout.JAVA_INT, 0);
            report.append("    物理设备数: " + count + "\n");

            MemorySegment devices = arena.allocate(ValueLayout.ADDRESS, count);
            invoke(vkEnumeratePhysicalDevices, instance, pCount, devices);

            for (int i = 0; i < count; i++) {
                MemorySegment dev = devices.get(ValueLayout.ADDRESS, (long) i * ValueLayout.ADDRESS.byteSize());
                report.append("    [设备 " + i + "]\n");

                // VkPhysicalDeviceProperties：apiVersion@0, driverVersion@4, vendorID@8,
                // deviceID@12, deviceType@16, deviceName@20..276
                MemorySegment props = arena.allocate(320);
                invoke(vkGetPhysicalDeviceProperties, dev, props);
                String name = props.getString(20);
                int vendorID = props.get(ValueLayout.JAVA_INT, 8);
                int deviceID = props.get(ValueLayout.JAVA_INT, 12);
                int deviceType = props.get(ValueLayout.JAVA_INT, 16);
                int apiVersion = props.get(ValueLayout.JAVA_INT, 0);
                String typeStr = switch (deviceType) {
                    case 1 -> "集成显卡";
                    case 2 -> "独立显卡";
                    case 3 -> "虚拟GPU";
                    case 4 -> "CPU";
                    default -> "未知(" + deviceType + ")";
                };
                String versionStr = "v" + (apiVersion >> 22) + "." + ((apiVersion >> 12) & 0x3FF) + "." + (apiVersion & 0xFFF);
                report.append("        名称: " + name + "\n");
                report.append("        API " + versionStr + " vendorID=0x" + Integer.toHexString(vendorID)
                        + " deviceID=0x" + Integer.toHexString(deviceID) + " 类型: " + typeStr + "\n");

                // --- 队列族：VkQueueFamilyProperties { VkQueueFlags queueFlags@0; uint32_t queueCount@4; ... }
                MemorySegment pQCount = arena.allocate(ValueLayout.JAVA_INT);
                invoke(vkGetPhysicalDeviceQueueFamilyProperties, dev, pQCount, MemorySegment.NULL);
                int qCount = pQCount.get(ValueLayout.JAVA_INT, 0);
                MemorySegment qProps = arena.allocate((long) qCount * 16);
                invoke(vkGetPhysicalDeviceQueueFamilyProperties, dev, pQCount, qProps);
                boolean hasCompute = false;
                for (int q = 0; q < qCount; q++) {
                    int flags = qProps.get(ValueLayout.JAVA_INT, (long) q * 16);
                    int queueCount = qProps.get(ValueLayout.JAVA_INT, (long) q * 16 + 4);
                    if ((flags & VK_QUEUE_COMPUTE_BIT) != 0) {
                        hasCompute = true;
                        report.append("        compute 队列族 #" + q + ": " + queueCount + " 个队列\n");
                    }
                }
                if (!hasCompute) {
                    report.append("        警告: 未发现 compute 队列族\n");
                }

                // --- 设备扩展（前 20 个）---
                MemorySegment pExtCount = arena.allocate(ValueLayout.JAVA_INT);
                invoke(vkEnumerateDeviceExtensionProperties, dev, MemorySegment.NULL, pExtCount, MemorySegment.NULL);
                int extCount = pExtCount.get(ValueLayout.JAVA_INT, 0);
                MemorySegment exts = arena.allocate((long) extCount * 260);
                invoke(vkEnumerateDeviceExtensionProperties, dev, MemorySegment.NULL, pExtCount, exts);
                report.append("        扩展数: " + extCount + "（前20个）:\n          ");
                for (int e = 0; e < Math.min(20, extCount); e++) {
                    report.append(exts.getString((long) e * 260)).append(", ");
                }
                report.append("\n");
            }
        } finally {
            invoke(vkDestroyInstance, instance, MemorySegment.NULL);
        }
    }
}
