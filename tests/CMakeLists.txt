add_test(NAME selftest COMMAND swe --headless --selftest)

# Hydraulic tests (tests/README.md): pytest on the built executable, when a Python with pytest is found
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" _swe_preset)
  if(_swe_preset STREQUAL "relwithdebinfo")
    set(_swe_preset release)
  endif()
  add_test(NAME pytest
           COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests -q -p no:cacheprovider --preset ${_swe_preset}
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
