#include"inherit.h"

using namespace std;

class Animal {
protected :
	string m_sex;
public:
	//classĬ����˽�еģ����캯�������ڹ������򣬷����޷�ʵ����
	Animal(string sex) :m_sex(sex) {}
	//����һ�����鷽��
	//virtual void fly()=0;
	//����һ���鷽��
	virtual void fly() {
		cout << "What is the sex of the Animal ?" << endl;
	}
	/*void fly() {
		cout << "What is the sex of the Animal?" << endl;
	}*/
};

class Cat :public Animal {
public:
	Cat(string sex) :Animal(sex) {}
	void fly() {
		cout << "This cat~s sex is " << this->m_sex << "." << endl;
	}
};

void inherit_show() {
	Animal* mycat = new Cat("��");
	mycat->fly();
	delete mycat;
}