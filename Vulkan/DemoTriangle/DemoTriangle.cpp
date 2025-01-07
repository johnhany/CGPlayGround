#define NOMINMAX // for fixing `std::max - expected an identifier` error
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <vector>
#include <optional>
#include <set>


// the resolution {WIDTH, HEIGHT} that we specified earlier when creating the window is measured in screen coordinates
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// validation layers are used for:
//   Checking the values of parameters against the specification to detect misuse
//   Tracking creationand destruction of objects to find resource leaks
//   Checking thread safety by tracking the threads that calls originate from
//   Logging every calland its parameters to the standard output
//   Tracing Vulkan calls for profilingand replaying
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Vulkan does not have the concept of a "default framebuffer", 
// hence it requires an infrastructure that will own the buffers we will render to before we visualize them on the screen. 
// This infrastructure is known as the swap chain and must be created explicitly in Vulkan. 
// The swap chain is essentially a queue of images that are waiting to be presented to the screen
// You have to enable the VK_KHR_swapchain device extension after querying for its support.
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


struct QueueFamilyIndices {
    // `std::optional` is a wrapper that contains no value until you assign something to it
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// There are basically three kinds of properties we need to check in swap chain:
//   Basic surface capabilities(min / max number of images in swap chain, min / max width and height of images)
//   Surface formats(pixel format, color space)
//   Available presentation modes
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


static std::vector<char> readFile(const std::string& filename) {
    // `ate`: Start reading at the end of the file
    // `binary`: Read the file as binary file(avoid text transformations)
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkSurfaceKHR surface;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;


    void initWindow() {
        std::cout << ">>> initWindow" << std::endl;

        // initializes the GLFW library
        glfwInit();

        // Because GLFW was originally designed to create an OpenGL context, we need to tell it to not create an OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // disable window resizing
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // The first three parameters specify the width, height and title of the window.
        // The fourth parameter allows you to optionally specify a monitor to open the window on 
        // and the last parameter is only relevant to OpenGL.
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

        std::cout << "<<< initWindow" << std::endl;
    }

    // checks if all of the requested layers are available
    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    std::cout << "Found validation layer: " << layerName << std::endl;
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    // return the required list of extensions based on whether validation layers are enabled or not
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    void createInstance() {
        std::cout << ">>> createInstance" << std::endl;

        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // global validation layers to enable
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        // 1st arg: Pointer to struct with creation info
        // 2nd arg: Pointer to custom allocator callbacks
        // 3rd arg: Pointer to the variable that stores the handle to the new object
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        // First get the number of extensions
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        // Then get extension details
        std::vector<VkExtensionProperties> extensionsAll(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionsAll.data());

        // Each `VkExtensionProperties` struct contains the name and version of an extension
        std::cout << "available extensions:\n";
        for (const auto& extension : extensionsAll) {
            std::cout << "\t" << extension.extensionName << " (version: " << extension.specVersion << ")" << std::endl;
        }

        std::cout << "<<< createInstance" << std::endl;
    }

    // a debug callback function
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {

        // The `messageSeverity` parameter specifies the severity of the message, which is one of the following flags:
        //   VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: Diagnostic message
        //   VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT : Informational message like the creation of a resource
        //   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT : Message about behavior that is not necessarily an error, but very likely a bug in your application
        //   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT : Message about behavior that is invalidand may cause crashes
        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            // Message is important enough to show
        }

        // The `messageType` parameter can have the following values:
        //   VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: Some event has happened that is unrelated to the specification or performance
        //   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT : Something has happened that violates the specification or indicates a possible mistake
        //   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT : Potential non-optimal use of Vulkan

        // The `pCallbackData` parameter refers to a VkDebugUtilsMessengerCallbackDataEXT struct containing the details of the message itself, with the most important members being:
        //   pMessage: The debug message as a null-terminated string
        //   pObjects : Array of Vulkan object handles related to the message
        //   objectCount : Number of objects in array

        // The `pUserData` parameter contains a pointer that was specified during the setup of the callback and allows you to pass your own data to it

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        // If the callback returns true, then the call is aborted with the VK_ERROR_VALIDATION_FAILED_EXT error
        return VK_FALSE;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr; // Optional
    }

    // Find `vkCreateDebugUtilsMessengerEXT` function's address
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) {
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        }
        else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    void setupDebugMessenger() {
        std::cout << ">>> setupDebugMessenger" << std::endl;

        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        // This struct should be passed to the `vkCreateDebugUtilsMessengerEXT` function to create the `VkDebugUtilsMessengerEXT` object. 
        // Unfortunately, because this function is an extension function, it is not automatically loaded. 
        // We have to look up its address ourselves using `vkGetInstanceProcAddr`
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }

        std::cout << "<<< setupDebugMessenger" << std::endl;
    }

    // To establish the connection between Vulkan and the window system to present results to the screen, we need to use the WSI (Window System Integration) extensions
    // The `VK_KHR_surface` extension is an instance level extension, it needs the HWND and HMODULE handles on Windows,
    // which on Windows is called `VK_KHR_win32_surface` and is also automatically included in the list from `glfwGetRequiredInstanceExtensions`
    void createSurface() {
        std::cout << ">>> createSurface" << std::endl;

        // `glfwCreateWindowSurface` function does the following job:
        //VkWin32SurfaceCreateInfoKHR createInfo{};
        //createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        //createInfo.hwnd = glfwGetWin32Window(window);
        //createInfo.hinstance = GetModuleHandle(nullptr);
        //vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface)

        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        std::cout << "<<< createSurface" << std::endl;
    }

    // almost every operation in Vulkan, anything from drawing to uploading textures, requires commands to be submitted to a queue
    // There are different types of queues that originate from different queue families and each family of queues allows only a subset of commands
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        // The `VkQueueFamilyProperties` struct contains some details about the queue family, 
        // including the type of operations that are supported and the number of queues that can be created based on that family
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        std::cout << "Found " << queueFamilies.size() << " queue families" << std::endl;

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            // ensure that a device can present images to the surface we created
            // the presentation is a queue-specific feature
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                std::cout << "Found graphics support in queue family: " << i << std::endl;
                indices.graphicsFamily = i;
            }
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                std::cout << "Found surface support in queue family: " << i << std::endl;
                indices.presentFamily = i;
            }
            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            std::cout << "Checking swap chain extension: " << extension.extensionName << std::endl;
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        // the basic surface capabilities
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        // query the supported surface formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        std::cout << "Found " << formatCount << " supported surface formats" << std::endl;

        // query the supported presentation modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        std::cout << "Found " << presentModeCount << " supported present modes" << std::endl;

        return details;
    }

    // check if any of the physical devices meet the requirements
    bool isDeviceSuitable(VkPhysicalDevice device) {
        bool foundDevice = false;
        bool foundQueueFamily = false;

        // Basic device properties like the name, typeand supported Vulkan version
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        // optional features like texture compression, 64 bit floats and multi viewport rendering (useful for VR)
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            deviceFeatures.geometryShader) {
            std::cout << "Found physical device: " << deviceProperties.deviceName << std::endl;
            foundDevice = true;
        }

        // Find supported queue family on device
        QueueFamilyIndices indices = findQueueFamilies(device);
        if (indices.isComplete()) {
            std::cout << "Found available queue family" << std::endl;
            foundQueueFamily = true;
        }

        // Find swap chain extension on device
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        // Check requirements of swap chain
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            std::cout << "Found available swap chain" << std::endl;
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            if (swapChainAdequate) {
                std::cout << "Found supported swap chain" << std::endl;
            }
        }

        return foundDevice && foundQueueFamily && swapChainAdequate;
    }

    void pickPhysicalDevice() {
        std::cout << ">>> pickPhysicalDevice" << std::endl;

        physicalDevice = VK_NULL_HANDLE;

        // Listing the graphics cards
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        // an array to hold all of the `VkPhysicalDevice` handles
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        std::cout << "<<< pickPhysicalDevice" << std::endl;
    }

    void createLogicalDevice() {
        std::cout << ">>> createLogicalDevice" << std::endl;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            // Vulkan lets you assign priorities to queues to influence the scheduling of command buffer execution using floating point numbers between 0.0 and 1.0. 
            // This is required even if there is only a single queue
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // the set of device features that we'll be using
        // An example of a device specific extension is `VK_KHR_swapchain`, which allows you to present rendered images from that device to windows
        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        // Enable device extensions (such as swap chain)
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        // Previous implementations of Vulkan made a distinction between instance and device specific validation layers, but this is no longer the case.
        // That means that the `enabledLayerCount` and `ppEnabledLayerNames` fields of `VkDeviceCreateInfo` are ignored by up-to-date implementations
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // The queues are automatically created along with the logical device
        // Device queues are implicitly cleaned up when the device is destroyed
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

        std::cout << "<<< createLogicalDevice" << std::endl;
    }

    // Surface format (color depth)
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // Each VkSurfaceFormatKHR entry contains a format and a colorSpace member
        // Prefer SRGB color space
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                std::cout << "Found preferred surface format" << std::endl;
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    // The presentation mode represents the actual conditions for showing images to the screen. 
    // There are four possible modes available in Vulkan:
    //   VK_PRESENT_MODE_IMMEDIATE_KHR: Images submitted by your application are transferred to the screen right away, which may result in tearing.
    //   VK_PRESENT_MODE_FIFO_KHR : The swap chain is a queue where the display takes an image from the front of the queue when the display is refreshed
    //     and the program inserts rendered images at the back of the queue.If the queue is full then the program has to wait.
    //     This is most similar to vertical sync as found in modern games.The moment that the display is refreshed is known as "vertical blank".
    //   VK_PRESENT_MODE_FIFO_RELAXED_KHR : This mode only differs from the previous one if the application is lateand the queue was empty at the last vertical blank.
    //     Instead of waiting for the next vertical blank, the image is transferred right away when it finally arrives.This may result in visible tearing.
    //   VK_PRESENT_MODE_MAILBOX_KHR : This is another variation of the second mode.Instead of blocking the application when the queue is full, 
    //     the images that are already queued are simply replaced with the newer ones.
    //     This mode can be used to render frames as fast as possible while still avoiding tearing, resulting in fewer latency issues than standard vertical sync.
    //     This is commonly known as "triple buffering", although the existence of three buffers alone does not necessarily mean that the framerate is unlocked.
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // On PC platform, prefer VK_PRESENT_MODE_MAILBOX_KHR
        // On mobile platform, prefer VK_PRESENT_MODE_FIFO_KHR, due to energy usage
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                std::cout << "Found preferred present mode" << std::endl;
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // The swap extent is the resolution of the swap chain images 
    // and it's almost always exactly equal to the resolution of the window that we're drawing to in pixels
    // On a high DPI display (like Apple's Retina display), screen coordinates don't correspond to pixels. 
    // Instead, due to the higher pixel density, the resolution of the window in pixel will be larger than the resolution in screen coordinates
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            std::cout << "Size of swap extent: (" << actualExtent.width << ", " << actualExtent.height << ")" << std::endl;
            return actualExtent;
        }
    }

    void createSwapChain() {
        std::cout << ">>> createSwapChain" << std::endl;

        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // simply sticking to this minimum means that we may sometimes have to wait on the driver to complete internal operations before we can acquire another image to render
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        // 0 is a special value that means that there is no maximum
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        std::cout << "Swap chain image count: " << imageCount << std::endl;

        // specify which surface the swap chain should be tied to
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        // The `imageArrayLayers` specifies the amount of layers each image consists of.This is always 1 unless you are developing a stereoscopic 3D application
        createInfo.imageArrayLayers = 1;
        // The `imageUsage` bit field specifies what kind of operations we'll use the images in the swap chain for
        //   `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` means rendering directly, which means that they're used as color attachment
        //   `VK_IMAGE_USAGE_TRANSFER_DST_BIT` means rendering images to a separate image first to perform operations like post-processing
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        // There are two ways to handle images that are accessed from multiple queues:
        //   VK_SHARING_MODE_EXCLUSIVE: An image is owned by one queue family at a timeand ownership must be explicitly transferred before using it in another queue family.
        //     This option offers the best performance.
        //   VK_SHARING_MODE_CONCURRENT : Images can be used across multiple queue families without explicit ownership transfers.
        // If the queue families differ, then we'll be using the concurrent mode in this tutorial to avoid having to do the ownership chapters
        // If the graphics queue family and presentation queue family are the same, which will be the case on most hardware, then we should stick to exclusive mode
        if (indices.graphicsFamily != indices.presentFamily) {
            std::cout << "Swap chain image sharing mode: VK_SHARING_MODE_CONCURRENT" << std::endl;
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            std::cout << "Swap chain image sharing mode: VK_SHARING_MODE_EXCLUSIVE" << std::endl;
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        // a certain transform can be applied to images in the swap chain, like a 90 degree clockwise rotation or horizontal flip
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        // The `compositeAlpha` field specifies if the alpha channel should be used for blending with other windows in the window system
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;
        // means that we don't care about the color of pixels that are obscured, for example because another window is in front of them
        // enabling for better performance
        createInfo.clipped = VK_TRUE;
        // it's possible that your swap chain becomes invalid or unoptimized while your application is running, for example because the window was resized. 
        // In that case the swap chain actually needs to be recreated from scratch and a reference to the old one must be specified in this field
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // Create a set of images that can be drawn onto and can be presented to the window
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        std::cout << "<<< createSwapChain" << std::endl;
    }

    // An image view is quite literally a view into an image. It describes how to access the image and which part of the image to access, 
    // for example if it should be treated as a 2D texture depth texture without any mipmapping levels.
    void createImageViews() {
        std::cout << ">>> createImageViews" << std::endl;

        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            // The `viewType` parameter allows you to treat images as 1D textures, 2D textures, 3D textures and cube maps.
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            // The `components` field allows you to swizzle the color channels around
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            // The `subresourceRange` field describes what the image's purpose is and which part of the image should be accessed
            // If you were working on a stereographic 3D application, then you would create a swap chain with multiple layers. 
            // You could then create multiple image views for each image representing the views for the left and right eyes by accessing different layers.
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }

        std::cout << "<<< createImageViews" << std::endl;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // we only need to specify a pointer to the buffer with the bytecode and the length of it
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        // The one catch is that the size of the bytecode is specified in bytes, but the bytecode pointer is a uint32_t pointer rather than a char pointer
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    // We need to specify how many color and depth buffers there will be, how many samples to use for each of them and how their contents should be handled throughout the rendering operations. 
    // All of this information is wrapped in a render pass object
    void createRenderPass() {
        std::cout << ">>> createRenderPass" << std::endl;

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        // no multisampling 
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        // The `loadOp` and `storeOp` determine what to do with the data in the attachment before rendering and after rendering. 
        // We have the following choices for loadOp:
        //   VK_ATTACHMENT_LOAD_OP_LOAD: Preserve the existing contents of the attachment
        //   VK_ATTACHMENT_LOAD_OP_CLEAR : Clear the values to a constant at the start
        //   VK_ATTACHMENT_LOAD_OP_DONT_CARE : Existing contents are undefined; we don't care about them
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // There are only two possibilities for the storeOp:
        //   VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memory and can be read later
        //   VK_ATTACHMENT_STORE_OP_DONT_CARE : Contents of the framebuffer will be undefined after the rendering operation
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // The loadOp and storeOp apply to color and depth data, and stencilLoadOp / stencilStoreOp apply to stencil data
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // Textures and framebuffers in Vulkan are represented by VkImage objects with a certain pixel format, 
        // however the layout of the pixels in memory can change based on what you're trying to do with an image.
        // Some of the most common layouts are :
        //   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment
        //   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : Images to be presented in the swap chain
        //   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : Images to be used as destination for a memory copy operation
        // The initialLayout specifies which layout the image will have before the render pass begins. 
        // The finalLayout specifies the layout to automatically transition to when the render pass finishes.
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // A single render pass can consist of multiple subpasses. 
        // Subpasses are subsequent rendering operations that depend on the contents of framebuffers in previous passes, 
        // for example a sequence of post-processing effects that are applied one after another. 
        // If you group these rendering operations into one render pass, then Vulkan is able to reorder the operations and conserve memory bandwidth for possibly better performance.
        VkAttachmentReference colorAttachmentRef{};
        // The `attachment` parameter specifies which attachment to reference by its index in the attachment descriptions array
        colorAttachmentRef.attachment = 0;
        // The `layout` specifies which layout we would like the attachment to have during a subpass that uses this reference
        // We intend to use the attachment to function as a color buffer and the VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL layout will give us the best performance
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        // The index of the attachment in this array is directly referenced from the fragment shader with the `layout(location = 0) out vec4 outColor` directive
        subpass.pColorAttachments = &colorAttachmentRef;
        // The following other types of attachments can be referenced by a subpass:
        //   pInputAttachments: Attachments that are read from a shader
        //   pResolveAttachments : Attachments used for multisampling color attachments
        //   pDepthStencilAttachment : Attachment for depthand stencil data
        //   pPreserveAttachments : Attachments that are not used by this subpass, but for which the data must be preserved

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        // the subpasses in a render pass automatically take care of image layout transitions. 
        // These transitions are controlled by subpass dependencies, which specify memory and execution dependencies between subpasses
        // We have only a single subpass right now, but the operations right before and right after this subpass also count as implicit "subpasses".
        VkSubpassDependency dependency{};
        // The special value VK_SUBPASS_EXTERNAL refers to the implicit subpass before or after the render pass depending on whether it is specified in srcSubpass or dstSubpass. 
        // The index 0 refers to our subpass, which is the first and only one. 
        // The dstSubpass must always be higher than srcSubpass to prevent cycles in the dependency graph (unless one of the subpasses is VK_SUBPASS_EXTERNAL).
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        std::cout << "<<< createRenderPass" << std::endl;
    }

    // Unlike earlier APIs, shader code in Vulkan has to be specified in a bytecode format as opposed to human-readable syntax like GLSL and HLSL. 
    // This bytecode format is called SPIR-V and is designed to be used with both Vulkan and OpenCL
    // Khronos has released their own vendor-independent compiler that compiles GLSL to SPIR-V.
    void createGraphicsPipeline() {
        std::cout << ">>> createGraphicsPipeline" << std::endl;

        // The vertex shader processes each incoming vertex. It takes its attributes, like world position, color, normal and texture coordinates as input. 
        // The output is the final position in clip coordinates and the attributes that need to be passed on to the fragment shader, like color and texture coordinates. 
        // These values will then be interpolated over the fragments by the rasterizer to produce a smooth gradient.
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        // Shader modules are just a thin wrapper around the shader bytecode that we've previously loaded from a file and the functions defined in it. 
        // The compilation and linking of the SPIR-V bytecode to machine code for execution by the GPU doesn't happen until the graphics pipeline is created. 
        // That means that we're allowed to destroy the shader modules again as soon as pipeline creation is finished
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        // There is one more (optional) member, pSpecializationInfo, which we won't be using here, but is worth discussing. 
        // It allows you to specify values for shader constants. You can use a single shader module where its behavior can be 
        // configured at pipeline creation by specifying different values for the constants used in it

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // describes the format of the vertex data that will be passed to the vertex shader. It describes this in roughly two ways:
        //   Bindings descriptions: spacing between data and whether the data is per-vertex or per-instance(see instancing)
        //   Attribute descriptions: type of the attributes passed to the vertex shader, which binding to load them from and at which offset
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

        // describes two things: what kind of geometry will be drawn from the vertices and if primitive restart should be enabled. 
        // The former is specified in the `topology` member and can have values like:
        //   VK_PRIMITIVE_TOPOLOGY_POINT_LIST: points from vertices
        //   VK_PRIMITIVE_TOPOLOGY_LINE_LIST : line from every 2 vertices without reuse
        //   VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : the end vertex of every line is used as start vertex for the next line
        //   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : triangle from every 3 vertices without reuse
        //   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : the second and third vertex of every triangle are used as first two vertices of the next triangle
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        // If you set the primitiveRestartEnable member to VK_TRUE, then it's possible to break up lines and triangles 
        // in the _STRIP topology modes by using a special index of 0xFFFF or 0xFFFFFFFF.
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // While viewports define the transformation from the image to the framebuffer, scissor rectangles define in which regions pixels will actually be stored
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        // viewport and scissor rectangle need to be combined into a viewport state
        // It is possible to use multiple viewports and scissor rectangles on some graphics cards, so its members reference an array of them. 
        // Using multiple requires enabling a GPU feature
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // The rasterizer takes the geometry that is shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader. 
        // It also performs depth testing, face culling and the scissor test, and it can be configured to output fragments that fill entire polygons or just the edges (wireframe rendering)
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        // If `depthClampEnable` is set to VK_TRUE, then fragments that are beyond the near and far planes are clamped to them as opposed to discarding them. 
        // This is useful in some special cases like shadow maps. Using this requires enabling a GPU feature.
        rasterizer.depthClampEnable = VK_FALSE;
        // If `rasterizerDiscardEnable` is set to VK_TRUE, then geometry never passes through the rasterizer stage. This basically disables any output to the framebuffer
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        // The `polygonMode` determines how fragments are generated for geometry. The following modes are available:
        //   VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
        //   VK_POLYGON_MODE_LINE : polygon edges are drawn as lines
        //   VK_POLYGON_MODE_POINT : polygon vertices are drawn as points
        // Using any mode other than fill requires enabling a GPU feature.
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        // The maximum line width that is supported depends on the hardware and any line thicker than 1.0f requires you to enable the wideLines GPU feature.
        rasterizer.lineWidth = 1.0f;
        // The `cullMode` variable determines the type of face culling to use. You can disable culling, cull the front faces, cull the back faces or both
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        // The `frontFace` variable specifies the vertex order for faces to be considered front-facing and can be clockwise or counterclockwise.
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        // The rasterizer can alter the depth values by adding a constant value or biasing them based on a fragment's slope. This is sometimes used for shadow mapping
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

        // configures multisampling, which is one of the ways to perform anti-aliasing
        // This mainly occurs along edges, which is also where the most noticeable aliasing artifacts occur. 
        // Because it doesn't need to run the fragment shader multiple times if only one polygon maps to a pixel, 
        // it is significantly less expensive than simply rendering to a higher resolution and then downscaling. 
        // Enabling it requires enabling a GPU feature.
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional

        // After a fragment shader has returned a color, it needs to be combined with the color that is already in the framebuffer. 
        // This transformation is known as color blending and there are two ways to do it:
        //   Mix the oldand new value to produce a final color
        //   Combine the oldand new value using a bitwise operation
        // There are two types of structs to configure color blending.
        //   `VkPipelineColorBlendAttachmentState` contains the configuration per attached framebuffer
        //   `VkPipelineColorBlendStateCreateInfo` contains the global color blending settings
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        // If you want to use the second method of blending (bitwise combination), then you should set logicOpEnable to VK_TRUE. 
        // The bitwise operation can then be specified in the logicOp field. Note that this will automatically disable the first method, 
        // as if you had set blendEnable to VK_FALSE for every attached framebuffer
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional


        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0; // Optional
        pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // Optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr; // Optional
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        // Vulkan allows you to create a new graphics pipeline by deriving from an existing pipeline. 
        // The idea of pipeline derivatives is that it is less expensive to set up pipelines when they have much functionality in common 
        // with an existing pipeline and switching between pipelines from the same parent can also be done quicker. 
        // You can either specify the handle of an existing pipeline with basePipelineHandle or reference another pipeline that is about to be created by index with basePipelineIndex
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional

        // A pipeline cache can be used to store and reuse data relevant to pipeline creation across multiple calls to vkCreateGraphicsPipelines and even across program executions if the cache is stored to a file
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);

        std::cout << "<<< createGraphicsPipeline" << std::endl;
    }

    void createFramebuffers() {
        std::cout << ">>> createFramebuffers" << std::endl;

        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }

        std::cout << "<<< createFramebuffers" << std::endl;
    }

    // Commands in Vulkan, like drawing operations and memory transfers, are not executed directly using function calls. 
    // You have to record all of the operations you want to perform in command buffer objects
    void createCommandPool() {
        std::cout << ">>> createCommandPool" << std::endl;

        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // There are two possible flags for command pools:
        //   VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers are rerecorded with new commands very often(may change memory allocation behavior)
        //   VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : Allow command buffers to be rerecorded individually, without this flag they all have to be reset together
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        // Command buffers are executed by submitting them on one of the device queues, like the graphics and presentation queues we retrieved. 
        // Each command pool can only allocate command buffers that are submitted on a single type of queue. 
        // We're going to record commands for drawing, which is why we've chosen the graphics queue family.
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

        std::cout << "<<< createCommandPool" << std::endl;
    }

    void createCommandBuffer() {
        std::cout << ">>> createCommandBuffer" << std::endl;

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        // The level parameter specifies if the allocated command buffers are primary or secondary command buffers.
        //   VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be submitted to a queue for execution, but cannot be called from other command buffers.
        //   VK_COMMAND_BUFFER_LEVEL_SECONDARY : Cannot be submitted directly, but can be called from primary command buffers.
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        std::cout << "<<< createCommandBuffer" << std::endl;
    }

    // A core design philosophy in Vulkan is that synchronization of execution on the GPU is explicit. 
    // The order of operations is up to us to define using various synchronization primitives which tell the driver the order we want things to run in. 
    // This means that many Vulkan API calls which start executing work on the GPU are asynchronous, the functions will return before the operation has finished.
    void createSyncObjects() {
        std::cout << ">>> createSyncObjects" << std::endl;

        // In summary, semaphores are used to specify the execution order of operations on the GPU while fences are used to keep the CPU and GPU in sync with each-other.

        // A semaphore is used to add order between queue operations
        // There happens to be two kinds of semaphores in Vulkan, binary and timeline
        // A semaphore is either unsignaled or signaled. It begins life as unsignaled. 
        // The way we use a semaphore to order queue operations is by providing the same semaphore as a 'signal' semaphore in one queue operation 
        // and as a 'wait' semaphore in another queue operation
        // We want to use semaphores for swapchain operations because they happen on the GPU
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // A fence has a similar purpose, in that it is used to synchronize execution, but it is for ordering the execution on the CPU, otherwise known as the host
        // Similar to semaphores, fences are either in a signaled or unsignaled state. Whenever we submit work to execute, we can attach a fence to that work. 
        // When the work is finished, the fence will be signaled. Then we can make the host wait for the fence to be signaled, 
        // guaranteeing that the work has finished before the host continues.
        // For waiting on the previous frame to finish, we want to use fences for the opposite reason, because we need the host to wait
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // `inFlightFence` is only signaled after a frame has finished rendering, yet since this is the first frame, there are no previous frames in which to signal the fence
        // Create the fence in the signaled state, so that the first call to vkWaitForFences() returns immediately since the fence is already signaled.
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }

        std::cout << "<<< createSyncObjects" << std::endl;
    }

    void initVulkan() {
        std::cout << ">>> initVulkan" << std::endl;

        createInstance();

        // The `vkCreateDebugUtilsMessengerEXT` call requires a valid instance to have been created
        setupDebugMessenger();

        // The window surface needs to be created right after the instance creation, because it can actually influence the physical device selection
        createSurface();

        pickPhysicalDevice();

        // You can even create multiple logical devices from the same physical device
        createLogicalDevice();

        createSwapChain();

        createImageViews();

        createRenderPass();

        createGraphicsPipeline();

        createFramebuffers();

        createCommandPool();

        createCommandBuffer();

        createSyncObjects();

        std::cout << "<<< initVulkan" << std::endl;
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        //std::cout << ">>> recordCommandBuffer" << std::endl;

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // The flags parameter specifies how we're going to use the command buffer. The following values are available:
        //   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.
        //   VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : This is a secondary command buffer that will be entirely within a single render pass.
        //   VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : The command buffer can be resubmitted while it is also already pending execution.
        beginInfo.flags = 0; // Optional
        // The pInheritanceInfo parameter is only relevant for secondary command buffers. 
        // It specifies which state to inherit from the calling primary command buffers.
        beginInfo.pInheritanceInfo = nullptr; // Optional

        // If the command buffer was already recorded once, then a call to vkBeginCommandBuffer will implicitly reset it. 
        // It's not possible to append commands to a buffer at a later time.
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        // We created a framebuffer for each swap chain image where it is specified as a color attachment. 
        // Thus we need to bind the framebuffer for the swapchain image we want to draw to. 
        // Using the imageIndex parameter which was passed in, we can pick the right framebuffer for the current swapchain image.
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        // The render area defines where shader loads and stores will take place. 
        // The pixels outside this region will have undefined values. It should match the size of the attachments for best performance.
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        // define the clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR, which we used as load operation for the color attachment
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        // All of the functions that record commands can be recognized by their vkCmd prefix. 
        // They all return void, so there will be no error handling until we've finished recording
        // The final parameter controls how the drawing commands within the render pass will be provided. It can have one of two values:
        //   VK_SUBPASS_CONTENTS_INLINE: The render pass commands will be embedded in the primary command buffer itself and no secondary command buffers will be executed.
        //   VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : The render pass commands will be executed from secondary command buffers.
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // The second parameter specifies if the pipeline object is a graphics or compute pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // It has the following parameters, aside from the command buffer:
        //   vertexCount: Even though we don't have a vertex buffer, we technically still have 3 vertices to draw.
        //   instanceCount : Used for instanced rendering, use 1 if you're not doing that.
        //   firstVertex : Used as an offset into the vertex buffer, defines the lowest value of `gl_VertexIndex`.
        //   firstInstance : Used as an offset for instanced rendering, defines the lowest value of `gl_InstanceIndex`.
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        //std::cout << "<<< recordCommandBuffer" << std::endl;
    }

    // At a high level, rendering a frame in Vulkan consists of a common set of steps:
    //   Wait for the previous frame to finish
    //   Acquire an image from the swap chain
    //   Record a command buffer which draws the scene onto that image
    //   Submit the recorded command buffer
    //   Present the swap chain image
    void drawFrame() {
        // The VK_TRUE we pass here indicates that we want to wait for all fences
        // This function also has a timeout parameter that we set to the maximum value of a 64 bit unsigned integer, UINT64_MAX, which effectively disables the timeout.
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        // The third parameter specifies a timeout in nanoseconds for an image to become available. 
        // Using the maximum value of a 64 bit unsigned integer means we effectively disable the timeout.
        // The last parameter specifies a variable to output the index of the swap chain image that has become available. 
        // The index refers to the VkImage in our swapChainImages array. We're going to use that index to pick the VkFrameBuffer
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        vkResetCommandBuffer(commandBuffer, 0);

        recordCommandBuffer(commandBuffer, imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // The last step of drawing a frame is submitting the result back to the swap chain to have it eventually show up on the screen
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        // It allows you to specify an array of VkResult values to check for every individual swap chain if presentation was successful. 
        // It's not necessary if you're only using a single swap chain, because you can simply use the return value of the present function.
        presentInfo.pResults = nullptr; // Optional

        vkQueuePresentKHR(presentQueue, &presentInfo);
    }

    void mainLoop() {
        std::cout << ">>> mainLoop" << std::endl;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        // Remember that all of the operations in drawFrame are asynchronous. 
        // That means that when we exit the loop in mainLoop, drawing and presentation operations may still be going on. 
        // Cleaning up resources while that is happening is a bad idea.
        // To fix that problem, we should wait for the logical device to finish operations before exiting mainLoopand destroying the window
        vkDeviceWaitIdle(device);

        std::cout << "<<< mainLoop" << std::endl;
    }

    // Find `vkDestroyDebugUtilsMessengerEXT` function's address
    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance, debugMessenger, pAllocator);
        }
    }

    void cleanup() {
        std::cout << ">>> cleanup" << std::endl;

        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        // The graphics pipeline is required for all common drawing operations, so it should also only be destroyed at the end of the program
        vkDestroyPipeline(device, graphicsPipeline, nullptr);

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        // Destroy the swap chain before device
        vkDestroySwapchainKHR(device, swapChain, nullptr);

        vkDestroyDevice(device, nullptr);

        // `vkDestroyDebugUtilsMessengerEXT` must be called before the instance is destroyed
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        // Make sure that the surface is destroyed before the instance.
        vkDestroySurfaceKHR(instance, surface, nullptr);

        // The `VkInstance` should be only destroyed right before the program exits
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();

        std::cout << "<<< cleanup" << std::endl;
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
