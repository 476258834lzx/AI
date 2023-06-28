#include"fun_temp.h"

using namespace std;

template <typename T>

T add(T& a, T& b) {
	return a + b;
}


void fun_temp_show() {
	int x = 3, y = 4;
	cout << add(x, y) << endl;
}