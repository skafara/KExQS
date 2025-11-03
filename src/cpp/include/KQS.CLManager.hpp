#pragma once

#include <CL/opencl.hpp>
#include <string>
#include <map>


class CLManager {
public:
    static CLManager& Instance();

    cl::Context& GetContext();
    cl::Device& GetDevice();
    cl::CommandQueue& GetCommandQueue();
    cl::Kernel& GetKernel(const std::string &name);

private:
    CLManager();
    void Initialize();
    void LoadKernels();
    void BuildProgram(const std::string &filePath);
    std::string ReadFile(const std::string &path);

    CLManager(const CLManager&) = delete;
    CLManager& operator=(const CLManager&) = delete;

    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _commandQueue;
    std::map<std::string, cl::Kernel> _kernels;
};
