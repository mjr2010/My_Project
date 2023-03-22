//the Cohen–Sutherland algorithm is an algorithm used for line clipping. 
//The algorithm divides a two-dimensional space into 9 regions and then efficiently determines the lines and portions of lines that are visible in the central region of interest (the viewport).


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<graphics.h>
#include<dos.h>
 
struct coordinate
{
	int x,y;
	char code[4];
}Q;
 
void drawwindow();
void drawline(Q p1,Q p2);
Q setcode(Q p);
int visibility(Q p1,Q p2);
Q resetQ(Q p1,Q p2);
 
void main()
{
	int gd=DETECT,v,gm;
	Q p1,p2,p3,p4,Qtemp;
	
	printf("\nEnter x1 and y1: ");
	scanf("%d %d",&p1.x,&p1.y);
	printf("\nEnter x2 and y2: ");
	scanf("%d %d",&p2.x,&p2.y);

	initgraph(&gd,&gm,"c:\\turboc3\\bgi");
	drawwindow();
	delay(1000);

	drawline(p1,p2);
	delay(1000);
	cleardevice();

	delay(5000);
	p1=setcode(p1);
	p2=setcode(p2);
	v=visibility(p1,p2);
	delay(500);
	
	switch(v)
	{
	case 0: drawwindow();
			delay(500);
			drawline(p1,p2);
			break;
	case 1:	drawwindow();
			delay(500);
			break;
	case 2:	p3=resetQ(p1,p2);
			p4=resetQ(p2,p1);
			drawwindow();
			delay(500);
			drawline(p3,p4);
			break;
	}
	
	delay(5000);
	closegraph();
}
 
void drawwindow()
{
	line(150,100,450,100);
	line(450,100,450,350);
	line(450,350,150,350);
	line(150,350,150,100);
}
 
void drawline(Q p1,Q p2)
{
	line(p1.x,p1.y,p2.x,p2.y);
}
 
Q setcode(Q p)	//for setting the 4 bit code
{
	Q Qtemp;
	
	if(p.y<100)
		Qtemp.code[0]='1';	//Top
	else
		Qtemp.code[0]='0';
	
	if(p.y>350)
		Qtemp.code[1]='1';	//Bottom
	else
		Qtemp.code[1]='0';
		
	if(p.x>450)
		Qtemp.code[2]='1';	//Right
	else
		Qtemp.code[2]='0';
		
	if(p.x<150)
		Qtemp.code[3]='1';	//Left
	else
		Qtemp.code[3]='0';
	
	Qtemp.x=p.x;
	Qtemp.y=p.y;
	
	return(Qtemp);
}
 
int visibility(Q p1,Q p2)
{
	int i,flag=0;
	
	for(i=0;i<4;i++)
	{
		if((p1.code[i]!='0') || (p2.code[i]!='0'))
			flag=1;
	}
	
	if(flag==0)
		return(0);
	
	for(i=0;i<4;i++)
	{
		if((p1.code[i]==p2.code[i]) && (p1.code[i]=='1'))
			flag='0';
	}
	
	if(flag==0)
		return(1);
	
	return(2);
}
 
Q resetQ(Q p1,Q p2)
{
	Q temp;
	int x,y,i;
	float m,k;
	
	if(p1.code[3]=='1')
		x=150;
	
	if(p1.code[2]=='1')
		x=450;
	
	if((p1.code[3]=='1') || (p1.code[2]=='1'))
	{
		m=(float)(p2.y-p1.y)/(p2.x-p1.x);
		k=(p1.y+(m*(x-p1.x)));
		temp.y=k;
		temp.x=x;
		
		for(i=0;i<4;i++)
			temp.code[i]=p1.code[i];
		
		if(temp.y<=350 && temp.y>=100)
			return (temp);
	}
	
	if(p1.code[0]=='1')
		y=100;
	
	if(p1.code[1]=='1')
		y=350;
		
	if((p1.code[0]=='1') || (p1.code[1]=='1'))
	{
		m=(float)(p2.y-p1.y)/(p2.x-p1.x);
		k=(float)p1.x+(float)(y-p1.y)/m;
		temp.x=k;
		temp.y=y;
		
		for(i=0;i<4;i++)
			temp.code[i]=p1.code[i];
		
		return(temp);
	}
	else
		return(p1);
}

