rel: 00-rel 01-rel 02-rel

deb: 00-deb 01-deb 02-deb

00-deb:
	nvcc -g -G 00-config.cu -o 00-config
00-rel:
	nvcc -O2 00-config.cu -o 00-config
	strip 00-config

01-deb:
	nvcc -g -G 01-add-vector.cu -o 01-add-vector
01-rel:
	nvcc -O2 01-add-vector.cu -o 01-add-vector
	strip 01-add-vector

02-deb:
	nvcc -g -G 02-recursion.cu -o 02-recursion
02-rel:
	nvcc -O2 02-recursion.cu -o 02-recursion
	strip 02-recursion


clean:
	rm -f 00-config 01-add-vector 02-recursion

ctags:
	ctags -R . /usr/local/cuda-6.5/extras/CUPTI/include /usr/local/cuda-6.5/extras/Debugger/include /usr/local/cuda-6.5/nvvm/libnvvm-samples/common/include /usr/local/cuda-6.5/nvvm/include /usr/local/cuda-6.5/samples/7_CUDALibraries/common/FreeImage/include /usr/local/cuda-6.5/targets/armv7-linux-gnueabihf/include /usr/local/cuda-6.5/include

