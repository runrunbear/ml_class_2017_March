#include "stdafx.h"
#include <math.h>
using namespace std;

double f(double x)	//y
{
	return x*x*x*x;
	return x*x;
	return -cos(x);
}

double g(double x)	//y'
{
	return 4*x*x*x;
	return 2*x;
	return sin(x);
}

double g2(double x)	//y''
{
	return 12*x*x;
	return 2;
	return cos(x);
}

double GetA_Armijo(double x, double d, double a)
{
	double c1 = 0.3;
	double now = f(x);
	double next = f(x - a*d);

	int count = 30;
	while(next < now)
	{
		a *= 2;
		next = f(x - a*d);
		count--;
		if(count == 0)
			break;
	}

	count = 50;
	while(next > now - c1*a*d*d)
	{
		a /= 2;
		next = f(x - a*d);
		count--;
		if(count == 0)
			break;
	}
	return a;
}

double GetA_Quad(double x, double d, double a)
{
	double c1 = 0.3;
	double now = f(x);
	double next = f(x - a*d);

	int count = 30;
	while(next < now)
	{
		a *= 2;
		next = f(x - a*d);
		count--;
		if(count == 0)
			break;
	}

	count = 50;
	double b;
	while(next > now - c1*a*d*d)
	{
		b = d * a * a / (now + d * a - next);
		b /= 2;
		if(b < 0)
			a /= 2;
		else
			a = b;
		next = f(x - a*d);
		count--;
		if(count == 0)
			break;
	}
	return a;
}


int _tmain(int argc, _TCHAR* argv[])
{
	double x = 1.5;
	double d;			//一阶导
	double a = 0.01;	//学习率
	for(int i = 0; i < 100; i++)
	{
		d = g(x);
		a = 1/g2(x);//GetA_Armijo(x, d, 10);
		x -= d * a;
		cout << i << '\t' << a << '\t' << x << '\n';
	}
	return 0;
}

