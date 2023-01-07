#include<graphics.h>
#include<conio.h>
#include<stdio.h>
#include<time.h>
#include<dos.h>
int gd=0,gm;
int num1=1,num2=1,X1=52,Y1=412,X2=52,Y2=392,turn=1;

void Ladder()
{
    int x1,y1,x2,y2;
    
setlinestyle(0,0,3);
    setcolor(10);

    //ladder from 61 to 97

    line(32,152,152,32);
    line(48,168,168,48);

    x1=44; y1=140; x2=60;y2=156;
    while(x1<152)
    {
        line(x1,y1,x2,y2);
        x1=x1+16;
        y1=y1-16;
        x2=x2+16;
        y2=y2-16;
    }

    //ladder from 54 to 71

    line(272,192,396,112);
    line(288,208,416,128);

    x1=288; y1=180; x2=304; y2=196;
    while(x1<392)
    {
        line(x1,y1,x2,y2);
        x1=x1+20;
        y1=y1-12;
        x2=x2+20;
        y2=y2-12;
    }

    //ladder from 35 to 76

    line(248,272,168,112);
    line(232,288,152,128);

    x1=232; y1=240; x2=216; y2=256;
    while(x1>168)
    {
        line(x1,y1,x2,y2);
        x1=x1-16;
        y1=y1-32;
        x2=x2-16;
        y2=y2-32;
    }

    // ladder from 13 to 27

    line(328,352,128,312);
    line(312,368,112,328);

    x1=308; y1=348; x2=292; y2=364;
    while(x1>128)
    {
        line(x1,y1,x2,y2);
        x1=x1-20;
        y1=y1-4;
        x2=x2-20;
        y2=y2-4;
    }
}

void Snakes()
{
    setcolor(7);
    setlinestyle(0,0,2);
        /* Snake from 99 to 78*/
    circle(80,50,10);
    circle(85,70,10);
    circle(90,90,10);
    circle(98,108,10);
    circle(112,120,10);

    /*snake from 86 to 38*/
    circle(240,90,10);
    circle(240,110,10);
    circle(235,129,10);
    circle(225,145,10);
    circle(210,158,10);
    circle(195,170,10);
    circle(180,180,10);
    circle(163,190,10);
    circle(153,207,10);
    circle(150,227,10);
    circle(145,245,10);
    circle(132,260,10);
    circle(120,276,10);

    /*snake from 51 to 12*/
    circle(405,210,10);
    circle(395,228,10);
    circle(390,248,10);
    circle(385,268,10);
    circle(385,288,10);
    circle(375,305,10);
    circle(365,323,10);

    /*snake from 41 to 3*/
    circle(50,250,10);
    circle(50,270,10);
    circle(50,290,10);
    circle(60,308,10);
    circle(70,326,10);
    circle(85,341,10);
    circle(90,362,10);
    circle(100,381,10);
    circle(115,396,10);
}

void position()
{
    setfillstyle(1,15);
    setlinestyle(0,0,3);
    //player 1
    setcolor(5);
    fillellipse(X1,Y1,6,6);

    //player 2
    setcolor(3);
    fillellipse(X2,Y2,6,6);
}

void board()
{
    int i,x1=20,x2=20,y1=20,y2=420;
    setcolor(9);
    outtextxy(148,0,"Snakes And Ladders");

    outtextxy(22,405, "1        2    3   4    5    6    7    8    9   10");
    outtextxy(25,365, "20     19   18   17   16  15   14   13   12   11");
    outtextxy(25,325, "21     22   23   24   25  26   27   28   29   30");
    outtextxy(25,285, "40     39   38   37   36  35   34   33   32   31");
    outtextxy(25,245, "41     42   43   44   45  46   47   48   49   50");
    outtextxy(25,205, "60     59   58   57   56  55   54   53   52   51");
    outtextxy(25,165, "61     62   63   64   65  66   67   68   69   70");
    outtextxy(25,125, "80     79   78   77   76  75   74   73   72   71");
    outtextxy(25,85,  "81     82   83   84   85  86   87   88   89   90");
    outtextxy(25,45,  "100   99   98   97   96  95   94    93   92   91");



    setcolor(14);
    setlinestyle(0,0,1);

        /*Draws vertical line of board*/

    for(i=0;i<=10;i++)
    {
       line(x1,y1,x2,y2);
       x1=x1+40;
       x2=x2+40;
    }

        /* Draws horizontal ;ines of board*/

    x1=20; y1=20;
    x2=420;y2=20;
    for(i=0;i<=10;i++)
    {
        line(x1,y1,x2,y2);
        y1=y1+40;
        y2=y2+40;
    }

    Ladder();
    Snakes();

    setcolor(15);
    outtextxy(450,40,"Press Enter To Roll Dice");
    setcolor(15);
	rectangle(490,60,570,140);
	floodfill(510,100,15);

}

void checkposition(int *n,int *x,int *y)
{
    switch(*n)
    {
        /* check for Ladder*/
        case 13 : *x=132;
                  if(turn==1)
                    *y=332;
                  else
                    *y=312;
                  *n=23;
                  break;

        case 35 : *x=172;
                  if(turn==1)
                    *y=132;
                  else
                    *y=112;
                  *n=77;
                  break;

        case 54 : *x=412;
                  if(turn==1)
                    *y=132;
                  else
                    *y=112;
                  *n=71;
                  break;

        case 61 : *x=172;
                  if(turn==1)
                    *y=52;
                  else
                    *y=32;
                  *n=97;
                  break;

        /*Check for Snake*/
        case 41 : *x=132;
                  if(turn==1)
                    *y=412;
                  else
                    *y=392;
                  *n=3;
                  break;

        case 51 : *x=372;
                  if(turn==1)
                    *y=332;
                  else
                    *y=312;
                  *n=29;
                  break;

        case 86 : *x=132;
                  if(turn==1)
                    *y=292;
                  else
                    *y=272;
                  *n=38;
                  break;

        case 99 : *x=132;
                  if(turn==1)
                    *y=132;
                  else
                    *y=112;
                  *n=78;
                  break;

    }
}

void move(int dice,int *n,int *x,int *y)
{
    int i;
    delay(2000);
    if(*n+dice<=100)
    {
            for(i=1;i<=dice;i++)
            {
                if(*n<10)
                {
                    *n=*n+1;
                    *x=*x+40;
                    if(turn==1)
                        *y=412;
                    else
                        *y=392;
                }
                else if(*n==10)
                {
                    *n=*n+1;
                    *x=412;
                    if(turn==1)
                        *y=372;
                    else
                        *y=352;
                }
                else if(*n<20)
                {
                    *n=*n+1;
                    *x=*x-40;
                    if(turn==1)
                        *y=372;
                    else
                        *y=352;
                }
                else if(*n==20)
                {
                    *n=*n+1;
                    *x=52;
                    if(turn==1)
                        *y=332;
                    else
                        *y=312;
                }
                else if(*n<30)
                {
                     *n=*n+1;
                    *x=*x+40;
                    if(turn==1)
                        *y=332;
                    else
                        *y=312;
                }
                else if(*n==30)
                {
                    *n=*n+1;
                    *x=412;
                    if(turn==1)
                        *y=292;
                    else
                        *y=272;
                }
                else if(*n<40)
                {
                    *n=*n+1;
                    *x=*x-40;
                    if(turn==1)
                        *y=292;
                    else
                        *y=272;
                }
                else if(*n==40)
                {
                    *n=*n+1;
                    *x=52;
                    if(turn==1)
                        *y=252;
                    else
                        *y=232;
                }
                else if(*n<50)
                {
                    *n=*n+1;
                    *x=*x+40;
                    if(turn==1)
                        *y=252;
                    else
                        *y=232;
                }
                else if(*n==50)
                {
                    *n=*n+1;
                    *x=412;
                    if(turn==1)
                        *y=212;
                    else
                        *y=192;
                }
                else if(*n<60)
                {
                    *n=*n+1;
                    *x=*x-40;
                    if(turn==1)
                        *y=212;
                    else
                        *y=192;
                }
                else if(*n==60)
                {
                    *n=*n+1;
                    *x=52;
                    if(turn==1)
                        *y=172;
                    else
                        *y=152;
                }
                else if(*n<70)
                {
                    *n=*n+1;
                    *x=*x+40;
                    if(turn==1)
                        *y=172;
                    else
                        *y=152;
                }
                else if(*n==70)
                {
                    *n=*n+1;
                    *x=412;
                    if(turn==1)
                        *y=132;
                    else
                        *y=112;
                }
                else if(*n<80)
                {
                    *n=*n+1;
                    *x=*x-40;
                    if(turn==1)
                        *y=132;
                    else
                        *y=112;
                }
                else if(*n==80)
                {
                    *n=*n+1;
                    *x=52;
                    if(turn==1)
                        *y=92;
                    else
                        *y=72;
                }
                else if(*n<90)
                {
                    *n=*n+1;
                    *x=*x+40;
                    if(turn==1)
                        *y=92;
                    else
                        *y=72;
                }
                else if(*n==90)
                {
                    *n=*n+1;
                    *x=412;
                    if(turn==1)
                        *y=52;
                    else
                        *y=32;
                }
                else if (*n<100)
                {
                    *n=*n+1;
                    *x=*x-40;
                    if(turn==1)
                        *y=52;
                    else
                        *y=32;
                }
            }
            position();
            checkposition(n,x,y);
            position();
            delay(300);
    }
}

int dice()
{
	char x;
    int key;
    srand(time(NULL));
    setcolor(15);
    setfillstyle(1,15);
    setlinestyle(0,0,2);

    rectangle(490,60,570,140);
    floodfill(510,100,15);


    key=getch();
    if(key==13)
    {
            x=1+(rand()%6);

        setcolor(0);
        setfillstyle(1,0);
            switch(x)
            {
                case 1: fillellipse(530,100,5,5);
                        break;

                case 2: fillellipse(505,75,5,5);
                        fillellipse(555,125,5,5);
                        break;

                case 3: fillellipse(505,75,5,5);
                        fillellipse(555,125,5,5);
                        fillellipse(530,100,5,5);
                        break;

                case 4: fillellipse(505,75,5,5);
                        fillellipse(555,125,5,5);
                        fillellipse(555,75,5,5);
                        fillellipse(505,125,5,5);
                        break;

                case 5: fillellipse(505,75,5,5);
                        fillellipse(555,125,5,5);
                        fillellipse(555,75,5,5);
                        fillellipse(505,125,5,5);
                        fillellipse(530,100,5,5);
                        break;

                case 6: fillellipse(505,75,5,5);
                        fillellipse(555,125,5,5);
                        fillellipse(555,75,5,5);
                        fillellipse(505,125,5,5);
                        fillellipse(555,100,5,5);
                        fillellipse(505,100,5,5);
                        break;
            }
    }
    if(x==6)
    {
        setcolor(15);
        outtextxy(450,180,"Press Enter to play Again");
        delay(800);
        cleardevice();
        if(turn==1)
        {
          setcolor(5);
          outtextxy(500,20,"Player 1");
        }

        else
        {
            setcolor(3);
            outtextxy(500,20,"Player 2");
        }

        board();
        position();
        x=6+dice();
    }
    return x;
}

void winner()
{
    setcolor(9);
    settextstyle(8,HORIZ_DIR,4);
    outtextxy(120,190,"CONGRATULATIONS!!!");

    if(num1==100)
        outtextxy(150,235,"Player 1 wins!");


    else
        outtextxy(150,235,"Player 2 wins!");


}

void play()
{
    int diceValue;

    while(1)
    {
        board();
        position();
        if(turn==1)
        {
            setcolor(5);
            outtextxy(500,20,"Player 1");
            diceValue=dice();
            move(diceValue,&num1,&X1,&Y1);
        }
        else
        {
            setcolor(3);
            outtextxy(500,20,"Player 2");
            diceValue=dice();
            move(diceValue,&num2,&X2,&Y2);
        }

        cleardevice();
        turn=1+turn%2;
        if(num1==100||num2==100)
        {
           winner();
           break;

        }

    }
}

void load()
{
    int i,j,c;
    setcolor(15);
    outtextxy(250,200,"Loading...");
    for(i=225,j=220,c=0;c<10;i=i+10,c++)
    {
        delay(800);
        rectangle(i,j,i+10,j+20);
        floodfill(i+1,j+1,15);
    }
    cleardevice();
}

void cover()
{
    setcolor(9);
    outtextxy(250,170,"Snakes ");
    line(250,185,297,185);
    outtextxy(270,190,"&");
    line(270,205,280,205);
    outtextxy(250,210,"Ladders");
    line(250,225,303,225);
    delay(3000);
   cleardevice();
}

void instructions()
{
    outtextxy(0,5,"1.The Snakes & Ladders game is a 2 player game.");
    outtextxy(0,20,"2. To roll the dice player has to press the Enter key on the keyboard.");
    outtextxy(0,35,"3. Player can start the game on the first roll of dice itself.");
    outtextxy(0,50,"4. When player rolls a 6 on the dice then player will get one more chance to play.");
    outtextxy(0,65,"5. When player lands on starting position of ladder, he will directly climb till the end of ladder.");
    outtextxy(0,80,"6. When player lands on mouth of snake then he will fall down to the position of tail.");
    outtextxy(0,95,"7. The player who reaches 1st on 100th position will be the winner of the game.");
    outtextxy(0,120,"Press any key to go back to menu.");
    if(getch())
        return;
}

void menu()
{
    int key;
    outtextxy(250,170,"1. New Game");
    outtextxy(250,190,"2. Instructions");
    outtextxy(250,210,"3. Quit");
    outtextxy(250,240,"Enter option Number");
    srand(time(NULL));
    key=getch();
    if(key==49)
    {
        cleardevice();
        load();
        board();
        position();
        play();
    }
    if(key==50)
    {
        cleardevice();
        instructions();
        cleardevice();
        menu();
    }
    if(key==51)
    {
        cleardevice();
        outtextxy(250,170,"Are you sure you want to quit?");
        outtextxy(250,190,"1. Yes");
        outtextxy(250,210,"2. No");
        outtextxy(250,240,"Enter option Number");
        srand(time(NULL));
        key=getch();
        if(key==49)
        {
                cleardevice();
                outtextxy(250,170,"Thank You For Playing");
                return;
        }
        if (key==50)
            cleardevice();
            menu();
    }
}

int main()
{
    initgraph(&gd,&gm,"C:\\turboc3\\bgi");


    cover();
    load();
    menu();

    getch();
    closegraph();
    return 0;
}

 

