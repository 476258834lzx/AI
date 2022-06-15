#include"inherit.h"

using namespace std;

class Animal {
protected :
	string m_sex;
public:
	//class默认是私有的，构造函数必须在公共区域，否则无法实例化
	Animal(string sex) :m_sex(sex) {}
	//定义一个纯虚方法
	//virtual void fly()=0;
	//定义一个虚方法
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
	Animal* mycat = new Cat("公");
	mycat->fly();
	delete mycat;
}