CPP_FILES := pyramid.cpp siftdesc.cpp helpers.cpp
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

MATLABPATH := /home/karel/bin/MATLAB/R2017a
OPENCV_LIB_PATH := $(MATLABPATH)/bin/glnxa64
OPENCV_INC_PATH := $(MATLABPATH)/toolbox/vision/builtins/src/ocvcg/opencv/include


FLAGS := -O3 -std=c++11 -Wall -lrt -I$(OPENCV_INC_PATH) -I$(OPENCV_INC_PATH)/opencv -L$(OPENCV_LIB_PATH)  -Wl,-rpath=$(OPENCV_LIB_PATH) -lopencv_highgui -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core

all: hesaff

hesaff: $(CPP_FILES) hesaff.cpp affine.cpp
	g++ $(FLAGS) -o hesaff $(CPP_FILES) affine.cpp hesaff.cpp

clean:
	rm hesaff
