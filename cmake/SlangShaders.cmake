# add_slang_shader(<name> <source.slang>)
#   Compiles one Slang file (and every module it imports) into
#   ${SWE_SHADER_OUTPUT_DIR}/<name>.spv with all its [shader(...)] entry points.
#   Entry points keep their Slang names (-fvk-use-entrypoint-name), so the
#   C++ side creates each pipeline with pName = the Slang function name.
find_program(SLANGC_EXECUTABLE slangc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(SWE_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/shaders")
file(MAKE_DIRECTORY "${SWE_SHADER_OUTPUT_DIR}")

# Any .slang file is a potential dependency (imports); good enough until -depfile is wired.
file(GLOB_RECURSE SWE_SLANG_ALL CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/shaders/*.slang")

add_custom_target(swe_shaders ALL)

function(add_slang_shader NAME SOURCE)
  set(out "${SWE_SHADER_OUTPUT_DIR}/${NAME}.spv")
  add_custom_command(
    OUTPUT "${out}"
    COMMAND "${SLANGC_EXECUTABLE}" "${SOURCE}"
            -target spirv -profile spirv_1_4 -emit-spirv-directly
            -capability spvGroupNonUniform -capability spvGroupNonUniformArithmetic
            -capability spvGroupNonUniformBallot -capability spvGroupNonUniformVote
            -fvk-use-entrypoint-name
            -I "${CMAKE_SOURCE_DIR}/shaders/common"
            -I "${CMAKE_SOURCE_DIR}/shaders/solver"
            -I "${CMAKE_SOURCE_DIR}/shaders/render"
            -I "${CMAKE_SOURCE_DIR}/shaders/effects"
            "$<$<CONFIG:Debug>:-g2>" "$<$<NOT:$<CONFIG:Debug>>:-O2>"
            -o "${out}"
    DEPENDS "${SOURCE}" ${SWE_SLANG_ALL}
    COMMENT "slangc ${NAME}.spv"
    VERBATIM COMMAND_EXPAND_LISTS)
  set_property(TARGET swe_shaders APPEND PROPERTY SOURCES "${out}")
endfunction()
