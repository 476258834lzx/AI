#include"memory.h"
#include<Windows.h>
using namespace std;

void stack() {
	int* ptr_a = NULL;
	{
		int a = 3;
		ptr_a = &a;
		cout << *ptr_a << endl;
	}
	cout << *ptr_a << endl;//error ptr_a has been delete
}

void heap() {
	int* ptr_a = NULL;
	{
		ptr_a = new int(4);
	}
	cout << *ptr_a << endl;
	delete ptr_a;
}

void leek() {
	for (int i = 1; i < 1000000; i++) {
		//int* a = new int[100000];
		int a[100000];
		delete [] a;
		Sleep(1);
	}
}

void memory_show() {
	stack();
	//heap();
	//leek();
}