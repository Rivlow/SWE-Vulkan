# cmake -DSRC=<dir> -DDST=<dir> -DGLOB=<pattern;pattern...> -P CopyGlob.cmake : copies the matching files of SRC into DST
file(MAKE_DIRECTORY "${DST}")
foreach(pattern IN LISTS GLOB)
  file(GLOB files "${SRC}/${pattern}")
  foreach(f IN LISTS files)
    file(COPY "${f}" DESTINATION "${DST}")
  endforeach()
endforeach()
