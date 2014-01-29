all: *.cpp
	g++ -g -O0 -Wall -o hesaff pyramid.cpp affine.cpp sim.cpp siftdesc.cpp helpers.cpp hesaff.cpp `pkg-config opencv --cflags --libs` -lrt
	g++ -g -O0 -Wall -o hes pyramid.cpp sim.cpp siftdesc.cpp helpers.cpp hes.cpp `pkg-config opencv --cflags --libs` -lrt
