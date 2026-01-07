#pragma once

#include <CL/opencl.hpp>
#include <string>
#include <map>
#include <span>


class CLManager {
public:
    static inline CLManager& Instance() {
        static CLManager instance;
        return instance;
    }

    template <typename T>
    static inline cl::Buffer AllocateWriteOnly(size_t size) {
        return cl::Buffer(GetContext(), CL_MEM_WRITE_ONLY, sizeof(T) * size);
    }

    template <typename T>
    static inline cl::Buffer AllocateFromReadOnly(std::span<const T> data) {
        return cl::Buffer(GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * data.size(), const_cast<T*>(data.data()));
    }

    template <typename T, typename Allocator>
    static inline cl::Buffer AllocateFromReadOnly(const std::vector<T, Allocator>& v) {
        return AllocateFromReadOnly(std::span<const T>(v));
    }

    template <typename... Args>
    static inline cl::Event Launch1D(std::string kernelName, size_t globalSize, Args&&... args) {
        cl::Kernel &kernel = GetKernel(kernelName);
        
        size_t argIndex = 0;
        (kernel.setArg(argIndex++, args), ...);
        
        cl::Event event;
        GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange, nullptr, &event);
        return event;
    }

    template <typename T>
    static inline void ReadbackBlocking(const cl::Buffer& buffer, std::span<T> out) {
        GetCommandQueue().enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T) * out.size(), out.data());
    }

private:

    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _commandQueue;
    std::map<std::string, cl::Kernel> _kernels;
    
    static inline cl::Context& GetContext() {
        return Instance()._context;
    }
    static inline cl::Device& GetDevice() {
        return Instance()._device;
    }
    static inline cl::CommandQueue& GetCommandQueue() {
        return Instance()._commandQueue;
    }
    static inline cl::Kernel& GetKernel(const std::string &name) {
        const auto it = Instance()._kernels.find(name);
        if (it == Instance()._kernels.end()) {
            throw std::runtime_error("Kernel Not Found: " + name);
        }
        return it->second;
    }

    CLManager();
    void Initialize();
    void LoadKernels();
    void BuildProgram(const std::string &filePath);
    std::string ReadFile(const std::string &path);

    CLManager(const CLManager&) = delete;
    CLManager& operator=(const CLManager&) = delete;
};
