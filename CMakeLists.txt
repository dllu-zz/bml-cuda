#Specifies minimum version required
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(bmlcuda)

FIND_PACKAGE(CUDA)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
)

SET(EXECUTABLE_OUTPUT_PATH ../bin 
CACHE INTERNAL "Single output directory for building all executables.")

SET(LIBRARY_OUTPUT_PATH ../bin 
CACHE INTERNAL "Single output directory for building all libraries.")

ADD_SUBDIRECTORY( src )