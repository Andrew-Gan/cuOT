#!/bin/bash

./ferret/emp-ot/run ./ferret/emp-ot/bin/test_ferret 24
# valgrind --leak-check=full ./ferret/emp-ot/run ./ferret/emp-ot/bin/test_ferret 28
# compute-sanitizer --tool memcheck --target-processes all ./ferret/emp-ot/run ./ferret/emp-ot/bin/test_ferret 28
# nsys profile --stats=true ./ferret/emp-ot/run ./ferret/emp-ot/bin/test_ferret 28
