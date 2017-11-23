CPP_FILES := pyramid.cpp siftdesc.cpp helpers.cpp
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
FLAGS := -O3 -Wall -lrt `pkg-config --cflags --libs opencv` 

all: hesaff

hesaff: $(CPP_FILES) hesaff.cpp affine.cpp
	g++ $(FLAGS) -o hesaff $(CPP_FILES) affine.cpp hesaff.cpp

clean:
	rm hesaff
