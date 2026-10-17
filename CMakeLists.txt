cmake_minimum_required(VERSION 3.29)
project(SWE-Vulkan VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan 1.4 REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Stb REQUIRED)

include(cmake/SlangShaders.cmake)

# ---- shaders --------------------------------------------------------------
add_slang_shader(selftest "${CMAKE_SOURCE_DIR}/shaders/selftest.slang")
add_slang_shader(solver   "${CMAKE_SOURCE_DIR}/shaders/solver/solver.slang")
add_slang_shader(terrain  "${CMAKE_SOURCE_DIR}/shaders/render/terrain.slang")
add_slang_shader(water    "${CMAKE_SOURCE_DIR}/shaders/render/water.slang")
add_slang_shader(skirt    "${CMAKE_SOURCE_DIR}/shaders/render/skirt.slang")
add_slang_shader(streamlines "${CMAKE_SOURCE_DIR}/shaders/render/streamlines.slang")
add_slang_shader(effects  "${CMAKE_SOURCE_DIR}/shaders/effects/effects.slang")

# ---- application ----------------------------------------------------------
file(GLOB_RECURSE SWE_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
add_executable(swe ${SWE_SOURCES})
target_include_directories(swe PRIVATE "${CMAKE_SOURCE_DIR}/src" ${Stb_INCLUDE_DIR})
target_link_libraries(swe PRIVATE Vulkan::Vulkan glfw glm::glm imgui::imgui implot::implot nlohmann_json::nlohmann_json)
target_compile_definitions(swe PRIVATE
  VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
  GLFW_INCLUDE_VULKAN
  GLM_FORCE_RADIANS
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  NOMINMAX
  WIN32_LEAN_AND_MEAN
  SWE_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
  SWE_SHADER_DIR="${SWE_SHADER_OUTPUT_DIR}")
target_compile_options(swe PRIVATE /W4 /permissive- /Zc:preprocessor /EHsc /utf-8 /bigobj)
target_precompile_headers(swe PRIVATE src/pch.h)
set_target_properties(swe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_dependencies(swe swe_shaders)

enable_testing()
add_subdirectory(tests)
