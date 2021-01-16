//SETTING THE EXTRA LIBRARY TO USE
#define CUDA 1
#define OPENCV 0
#define TENSORRT 1
#define NUMCPP 0

#include "main.hpp"

int main(void) {

	CUDA ? interface2() : interface2();

	std::cout << "test success ! \n" << std::endl;

	do {
		std::cout << '\n' << "Press the Enter key to continue.";
	} while (std::cin.get() != '\n');
}