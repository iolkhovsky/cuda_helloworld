nvcc -o sm-optimized-vector-add 01-vector-add/01-vector-add.cu -run
nsys profile --stats=true ./sm-optimized-vector-add
