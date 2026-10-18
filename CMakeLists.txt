cmake_minimum_required(VERSION 3.29)
project(SWE-Vulkan VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ninja tracks the headers of each object through cl's /showIncludes lines. A French Visual Studio prints them as
# "Remarque : inclusion du fichier :" with non-breaking spaces in UTF-8 (C2 A0), while CMake detects the prefix and
# writes rules.ninja in the console code page (cp850: A0 -> FF): nothing matched, no header was tracked, and editing
# a .h left stale objects (crashes). Use the UTF-8 form; build.bat switches the console to UTF-8 (chcp 65001) so
# that CMake writes it unchanged.
if(MSVC AND CMAKE_CXX_CL_SHOWINCLUDES_PREFIX MATCHES "^Remarque")
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "Remarque : inclusion du fichier :  ")
  set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "${CMAKE_CL_SHOWINCLUDES_PREFIX}")
  set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "${CMAKE_CL_SHOWINCLUDES_PREFIX}")
endif()

find_package(Vulkan 1.4 REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(GDAL CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)

include(cmake/SlangShaders.cmake)

# ---- shaders --------------------------------------------------------------
add_slang_shader(selftest "${CMAKE_SOURCE_DIR}/shaders/selftest.slang")
add_slang_shader(solver   "${CMAKE_SOURCE_DIR}/shaders/solver/solver.slang")
add_slang_shader(terrain  "${CMAKE_SOURCE_DIR}/shaders/render/terrain.slang")
add_slang_shader(water    "${CMAKE_SOURCE_DIR}/shaders/render/water.slang")
add_slang_shader(skirt    "${CMAKE_SOURCE_DIR}/shaders/render/skirt.slang")
add_slang_shader(streamlines "${CMAKE_SOURCE_DIR}/shaders/render/streamlines.slang")
add_slang_shader(effects  "${CMAKE_SOURCE_DIR}/shaders/effects/effects.slang")

# ---- common build settings ------------------------------------------------
add_library(swe_common INTERFACE)
target_include_directories(swe_common INTERFACE "${CMAKE_SOURCE_DIR}/src" ${Stb_INCLUDE_DIR})
target_link_libraries(swe_common INTERFACE Vulkan::Vulkan glfw glm::glm nlohmann_json::nlohmann_json)
target_compile_definitions(swe_common INTERFACE
  VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
  GLFW_INCLUDE_VULKAN
  GLM_FORCE_RADIANS
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  NOMINMAX
  WIN32_LEAN_AND_MEAN
  SWE_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
  SWE_SHADER_DIR="${SWE_SHADER_OUTPUT_DIR}"
  # PROJ / GDAL resource files (proj.db, ...) of the vcpkg install
  SWE_PROJ_DATA="${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/proj"
  SWE_GDAL_DATA="${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/gdal")
target_compile_options(swe_common INTERFACE /W4 /permissive- /Zc:preprocessor /EHsc /utf-8 /bigobj)

# ---- modules ----------------------------------------------------------------
# One static library per src/<dir>, linked only to the layers below it: an include that crosses the layering
# (e.g. the solver reaching into the application) does not link.
#   util <- vk, cases <- geo, solver <- render, sim <- swe (app)
function(swe_module NAME DIR)
  file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/${DIR}/*.cpp")
  add_library(${NAME} STATIC ${sources})
  target_link_libraries(${NAME} PUBLIC swe_common ${ARGN})
  target_precompile_headers(${NAME} PRIVATE "${CMAKE_SOURCE_DIR}/src/pch.h")
endfunction()

swe_module(swe_util   util)
swe_module(swe_vk     vk     swe_util)
swe_module(swe_cases  cases  swe_util)
swe_module(swe_geo    geo    swe_cases GDAL::GDAL CURL::libcurl)
swe_module(swe_solver solver swe_vk swe_cases)
swe_module(swe_render render swe_solver)
swe_module(swe_sim    sim    swe_solver swe_geo)

# ---- application ------------------------------------------------------------
file(GLOB_RECURSE SWE_APP_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/app/*.cpp")
add_executable(swe "${CMAKE_SOURCE_DIR}/src/main.cpp" ${SWE_APP_SOURCES} "${CMAKE_SOURCE_DIR}/packaging/app.rc")
target_link_libraries(swe PRIVATE swe_sim swe_render imgui::imgui implot::implot)
target_precompile_headers(swe PRIVATE src/pch.h)
set_target_properties(swe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# Windows application (no console window when double-clicked); main() attaches to the parent console when there
# is one, so command-line runs and the Python tools still see the logs
if(MSVC)
  target_link_options(swe PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
endif()
add_dependencies(swe swe_shaders)

# ---- distribution (packaging/make_installer.bat) ---------------------------
# dist/: everything the packaged application needs next to the executable
set(SWE_DIST_DIR "${CMAKE_BINARY_DIR}/dist" CACHE PATH "staging folder of the packaged application")
add_custom_target(dist
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${SWE_DIST_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SWE_DIST_DIR}/assets/cases" "${SWE_DIST_DIR}/assets/data"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:swe>" "${SWE_DIST_DIR}/SWE-Vulkan.exe"
  COMMAND ${CMAKE_COMMAND} -DSRC="${CMAKE_BINARY_DIR}/bin" -DDST="${SWE_DIST_DIR}" -DGLOB=*.dll -P "${CMAKE_SOURCE_DIR}/cmake/CopyGlob.cmake"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${SWE_SHADER_OUTPUT_DIR}" "${SWE_DIST_DIR}/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/proj" "${SWE_DIST_DIR}/proj"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/gdal" "${SWE_DIST_DIR}/gdal"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/assets/icon.png" "${SWE_DIST_DIR}/assets/icon.png"
  # demonstration cases: the analytic ones and the imported area with its orthophoto (not the large surveys)
  COMMAND ${CMAKE_COMMAND} -DSRC="${CMAKE_SOURCE_DIR}/assets/cases" -DDST="${SWE_DIST_DIR}/assets/cases" "-DGLOB=dam_break.json;lake_at_rest.json;drain_test.json;inflow_test.json;gw_*.json;rain_infiltration.json;zone_importee.json" -P "${CMAKE_SOURCE_DIR}/cmake/CopyGlob.cmake"
  COMMAND ${CMAKE_COMMAND} -DSRC="${CMAKE_SOURCE_DIR}/assets/data" -DDST="${SWE_DIST_DIR}/assets/data" "-DGLOB=zone_importee*" -P "${CMAKE_SOURCE_DIR}/cmake/CopyGlob.cmake"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/README.md" "${SWE_DIST_DIR}/README.md"
  DEPENDS swe
  COMMENT "Staging the packaged application in ${SWE_DIST_DIR}")

enable_testing()
add_subdirectory(tests)
