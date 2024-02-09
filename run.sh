#/bin/bash

echo "compiling matmul.c"
clang matmul.c -O3 
echo "running matmul"
./a.out
rm a.out
