#include"clazz.h"

using namespace std;

struct Person {
	string name;
	int age;

	//Ĭ�Ϲ��캯��
	Person() {
		this->name = "lingluoyu";
		this->age = 32;
	}

	//�Զ��幹�캯��
	Person(string name,int age) {
		this->name = name;
		this->age = age;
	}

	void say() {
		cout << "my name is " << this->name << ",i am" << this->age << "year old." << endl;
	}

	//��������
	~Person() {
		cout << "person�ѱ��ͷ�" << endl;
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