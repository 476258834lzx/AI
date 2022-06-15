#include"interrupt.h"

using namespace std;

void interrupt_show() {
	int a[] = { 0,1,2,3,4,5,6,7,8,9 };
	for (int i = 0; i < 10; i++) {
		if (a[i] == 5) {
			continue;
		}
		cout << a[i];
	}

	cout << "----------------------------" << endl;

	for (int i = 0; i < 10; i++) {
		if (a[i] == 5) {
			break;
		}
		cout << a[i];
	}
}