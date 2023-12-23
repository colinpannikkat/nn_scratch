OBJS	= main.o nn.o
SOURCE	= main.cpp nn.cpp
HEADER	= nn.h
OUT	= nn
CC	 = g++
FLAGS	 = -std=c++17 -g -c -Wall
LFLAGS	 = 

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

main.o: main.cpp
	$(CC) $(FLAGS) main.cpp 

nn.o: nn.cpp
	$(CC) $(FLAGS) nn.cpp

clean:
	rm -f $(OBJS) $(OUT)