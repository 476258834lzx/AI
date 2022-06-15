#include"constant.h"
using namespace std;

#define PI 3.1415926

void constant_show() {
	cout << PI << endl;

	const double e = 2.71;
	cout << e << endl;

	const float b = 1.2f;
	cout << b << endl;

	const int c = 0x10;
	cout << c << endl;

	const int d = 0b10;
	cout << d << endl;
}