#g++ or direct path to it
CC = g++ -g -Wall
#Mingw or Unix
CompilerVersion = Mingw

demo.exe : demo.o training-data-reader.o
	$(CC) -std=c++11 -s -o demo.exe demo.o training-data-reader.o  
demo.o : demo.cpp training-data-reader.h
	$(CC) -c demo.cpp
training-data-reader.o : training-data-reader.cpp training-data-reader.h
	$(CC) -c training-data-reader.cpp
clean:
	del *.o
	del *.exe
