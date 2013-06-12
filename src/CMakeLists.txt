#
# List of source files


INCLUDE(FindCUDA)
SET(bmlcuda_SRC 
	main.cpp 
	bml.cpp
	bml.cu
)


CUDA_ADD_EXECUTABLE(bmlcuda ${bmlcuda_SRC})
 
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
)

TARGET_LINK_LIBRARIES(bmlcuda)