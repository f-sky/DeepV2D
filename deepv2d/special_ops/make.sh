#TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_CFLAGS='-I/home/xieyiming/anaconda3/envs/deepv2d/lib/python3.6/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0'
#TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LFLAGS='-L/home/xieyiming/anaconda3/envs/deepv2d/lib/python3.6/site-packages/tensorflow -ltensorflow_framework'


CUDALIB=/usr/local/cuda-10.0/lib64/

nvcc -std=c++11 -c -o backproject_op_gpu.cu.o backproject_op_gpu.cu.cc \
  -I/home/xieyiming/anaconda3/envs/deepv2d/lib/python3.6/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o backproject.so backproject_op.cc \
  backproject_op_gpu.cu.o -I/home/xieyiming/anaconda3/envs/deepv2d/lib/python3.6/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -fPIC -lcudart  -L${CUDALIB} -L/home/xieyiming/anaconda3/envs/deepv2d/lib/python3.6/site-packages/tensorflow -ltensorflow_framework

