#include"array.h"

using namespace std;

void array_show() {
	int a[10];
	cout << sizeof(a) << endl;

	double b[5] = {1000.2,2.3,3.4,7.1,50.7};
	cout << b[3] << endl;
	cout << b[7] << endl;

	cout << "------------------------------" << endl;

	cout << b << endl;
	cout << *b << endl;
	cout << *(b + 1) << endl;
	cout << *(b + 7) << endl;

	cout << "------------------------------" << endl;
	int c[3][4] = {
		{0,1,2,3},
		{4,5,6,7},
		{8,9,10,11}
	};
	cout << c[1][2] << endl;
	cout << c[1] << endl;
}