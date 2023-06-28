#include"clazz.h"

using namespace std;

struct Person {
	string name;
	int age;

	//默认构造函数
	Person() {
		this->name = "lingluoyu";
		this->age = 32;
	}

	//自定义构造函数
	Person(string name,int age) {
		this->name = name;
		this->age = age;
	}

	void say() {
		cout << "my name is " << this->name << ",i am" << this->age << "year old." << endl;
	}

	//析构函数
	~Person() {
		cout << "person已被释放" << endl;
	}
};

void clazz_show() {
	{Person tom;
	tom.say(); }
	cout << "---------------------------" << endl;
	{
		Person* ling = new Person("lingyu", 25);
		ling->say();
		(*ling).say();
	}
	cout << "***************************" << endl;
}