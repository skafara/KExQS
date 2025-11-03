#include "KQS.CLManager.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>


#ifndef OPENCL_KERNELS_PATH
#define OPENCL_KERNELS_PATH "."
#endif


#ifdef _WIN32
#include <windows.h>
#endif

static std::filesystem::path GetExecutableDir() {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    return exePath.parent_path();
#else
    return std::filesystem::current_path();
#endif
}


CLManager& CLManager::Instance() {
    static CLManager instance;
    return instance;
}

CLManager::CLManager() {
    Initialize();
    LoadKernels();
}

void CLManager::Initialize() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    _device = devices.front();

    _context = cl::Context(_device);
    _commandQueue = cl::CommandQueue(_context, _device);
}

void CLManager::LoadKernels() {
    const std::filesystem::path path = GetExecutableDir() / OPENCL_KERNELS_PATH;
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        BuildProgram(entry.path().string());
    }
}

void CLManager::BuildProgram(const std::string &filePath) {
    const std::string source = ReadFile(filePath);
    cl::Program program(_context, source);
    program.build({_device});

    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
    for (auto &k : kernels) {
        std::string name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
        _kernels[name] = k;
    }
}

std::string CLManager::ReadFile(const std::string &path) {
    std::ifstream file(path);
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

cl::Context& CLManager::GetContext() {
    return _context;
}

cl::Device& CLManager::GetDevice() {
    return _device;
}

cl::CommandQueue& CLManager::GetCommandQueue() {
    return _commandQueue;
}

cl::Kernel& CLManager::GetKernel(const std::string &name) {
    const auto it = _kernels.find(name);
    if (it == _kernels.end())
        throw std::runtime_error("Kernel Not Found: " + name);
    return it->second;
}
