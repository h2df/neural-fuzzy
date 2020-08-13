#g++ or direct path to it
CC = g++
#Mingw or Unix
CompilerVersion = Mingw

main.exe : main.o training-data-reader.o nf-system.o
	$(CC) -std=c++11 -s -o main.exe main.o training-data-reader.o nf-system.o 
main.o : main.cpp training-data-reader.h nf-system.h
	$(CC) -c main.cpp
training-data-reader.o : training-data-reader.cpp training-data-reader.h pendulum-data.h
	$(CC) -c training-data-reader.cpp
nf-system.o : nf-system.cpp nf-system.h pendulum-data.h
	$(CC) -c nf-system.cpp
clean:
	del *.o
	del *.exe
