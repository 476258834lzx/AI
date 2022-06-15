#include"stl_vector.h"

using namespace std;



void stl_vector_show() {
	vector <int> vec;
	int i;

	cout << "vector size=" << vec.size() << endl;

	for (i = 0; i < 5; i++) {
		vec.push_back(i);
	}

	cout << "extended vector size=" << vec.size() << endl;

	for (i = 0; i < 5; i++) {
		cout << "value of vec [" << i << "]=" << vec[i] << endl;
	}

	vector <int>::iterator v = vec.begin();
	while (v != vec.end()) {
		cout << "Value of v=" << *v << endl;
		v++;
	}
}