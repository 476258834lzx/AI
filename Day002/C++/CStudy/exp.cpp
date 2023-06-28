#include"exp.h"

using namespace std;

void arithmetic() {
	int a = 21;
	int b = 10;
	int c;
	
	c = a + b;
	cout << "Line 1-c 的值是" << c << endl;
	c = a - b;
	cout << "Line 2-c 的值是" << c << endl;
	c = a * b;
	cout << "Line 3-c 的值是" << c << endl;
	c = a / b;
	cout << "Line 4-c 的值是" << c << endl;
	c = a % b;
	cout << "Line 5-c 的值是" << c << endl;

	int d = 10;
	c = d++;
	cout << "Line 6-c 的值是" << c << endl;

	d = 10;
	c = d--;
	cout << "Line 7-c 的值是" << c << endl;
}

void compare() {
	cout << (1 < 2) << endl;
	cout << (1 > 2) << endl;
	cout << (1 == 2) << endl;
	cout << (1 != 2) << endl;
	cout << (1 >= 2) << endl;
	cout << (1 <= 2) << endl;
}

void logic() {
	cout << (1 < 2 && 3 > 2) << endl;
	cout << (1 < 2 || 3 < 2) << endl;
	cout << !(1 < 2) << endl;
}

void assign() {
	int a = 21;
	int c;
	
	c = a;
	cout << "Line 1 - =运算符实例，c的值=：" << c << endl;
	c += a;
	cout << "Line 2 - +=运算符实例，c的值=:" << c << endl;
	c -= a;
	cout << "Line 3 - +=运算符实例，c的值=:" << c << endl;
	c *= a;
	cout << "Line 4 - *=运算符实例，c的值=:" << c << endl;
	c /= a;
	cout << "Line 5 - /=运算符实例，c的值=:" << c << endl;
	
	c = 200;
	c %= a;
	cout << "Line 6 - %=运算符实例，c的值=:" << c << endl;
}

void exp_show() {
	arithmetic();
	compare();
	logic();
	assign();
}