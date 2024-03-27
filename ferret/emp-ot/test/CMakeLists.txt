#Testing macro
macro (add_test_executable_with_lib _name libs)
	add_executable(test_${_name} ../../../gpu/gpu.o ${_name}.cpp ../emp-ot/ferret/dev_layer.cu)
	target_include_directories(test_${_name} PUBLIC ../../../gpu)
	target_link_libraries(test_${_name} ${EMP-TOOL_LIBRARIES} "cuda" "cudart" "curand")
endmacro()

macro (add_test_case _name)
	add_test_executable_with_lib(${_name} "")
  	add_test(NAME ${_name} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_test_case_with_run _name)
	add_test_executable_with_lib(${_name} "")
	add_test(NAME ${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")

endmacro()

# Test cases
add_test_case_with_run(main)
