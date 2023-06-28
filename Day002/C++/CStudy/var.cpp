#include"var.h"
using namespace std;

void var_show() {
	int i = 0;
	cout << i << endl;

	float j = 2.2;
	cout << j << endl;

	unsigned int k = -2;
	cout << k << endl;

	char c = 'a', d = 0x35;
	cout << c << " " << d << endl;

	string str1 = "hello";
	cout << str1 << endl;

	char str2[] = "world";
	cout << str2 << endl;

	char str3[] = { 'L','i','e','W','e','i','\0'};
	cout << str3 << endl;
}