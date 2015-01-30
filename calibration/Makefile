PROJECTNAME :=calibrate

CC := g++-4.9
CFLAGS := -fdiagnostics-color=always -std=c++11 -Wall $(ARGS) `pkg-config --cflags opencv`

CUDAPATH := /usr/local/cuda
CUDALIBPATH := -L$(CUDAPATH)/lib64


LIBS := `pkg-config --libs opencv`
INCLUDES := 

all: $(PROJECTNAME)

$(PROJECTNAME): calibrate.o
	$(CC) $(CFLAGS) -o $@ $^ $(CUDALIBPATH) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $(INCLUDES) $< 

clean:
	sudo rm *.o
	sudo rm $(PROJECTNAME)