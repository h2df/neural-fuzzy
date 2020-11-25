main.exe		: 	main.o algorithm.o 
	g++ -g -std=c++11 -W -o main.exe main.o algorithm.o -l gdi32 -static-libgcc -static-libstdc++
			
main.o		:	main.cpp algorithm.h
	g++ -g -std=c++11 -c -fpermissive -fconserve-space  -Wno-write-strings main.cpp
	
algorithm.o		:	 algorithm.cpp algorithm.h
	g++ -g -std=c++11 -c -fpermissive -fconserve-space  -Wno-write-strings algorithm.cpp		
	
clean:
	del *.o
	del *.exe