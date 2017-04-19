all: 00-rel

00-deb:
	nvcc -g 00-config.cu -o 00-config
00-rel:
	nvcc -O2 00-config.cu -o 00-config
	strip 00-config

ctags:
	ctags -R . /usr/local/cuda-6.5/extras/CUPTI/include /usr/local/cuda-6.5/extras/Debugger/include /usr/local/cuda-6.5/nvvm/libnvvm-samples/common/include /usr/local/cuda-6.5/nvvm/include /usr/local/cuda-6.5/samples/7_CUDALibraries/common/FreeImage/include /usr/local/cuda-6.5/targets/armv7-linux-gnueabihf/include /usr/local/cuda-6.5/include

