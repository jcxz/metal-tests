#include <iostream>

extern bool test1();
extern bool test2();
extern bool test3();


int main()
{
	std::cout << "========== Test 1 ==========" << std::endl;
	test1();

	std::cout << "========== Test 2 ==========" << std::endl;
	test2();

	std::cout << "========== Test 3 ==========" << std::endl;
	test3();

	return 0;
}
