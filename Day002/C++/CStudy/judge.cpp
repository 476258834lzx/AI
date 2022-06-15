#include"judge.h"

using namespace std;

void judge_show() {
	int a = 1;

	if (a == 1) {
		cout << "1" << endl;
	}
	else if (a == 2) {
		cout << "2" << endl;
	}
	else {
		cout << "other" << endl;
	}

	bool b = true;
	string c = b ? "true" : "false";
	cout << c << endl;
}