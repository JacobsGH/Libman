#Решение задачи Дирихле методом Либмана для области произвольного очертания
#Функция "f" задает функцию внутри фигуры, функция "Gran_Func" задает функцию на границе фигуры
#Функция "Search_Granica" задает форму граничных условий(проходит по всей сетке и определяет какие узлы сетки лежат на границе)
#Функция "Libman" задает метод Либмана который будет использоваться для полученной сетки
#-------------------------Библиотеки--------------------------
import math
#import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d;
import numpy as np
import time
from numba import njit
#-------------------------------------------------------------
start_time = time.time()
#------------Функции заданные внутри границы------------------

@njit(fastmath=True)
def f1 (x, y): 
    return 10

@njit(fastmath=True)
def f2(x, y): 
    return x*y

@njit(fastmath=True)#Точное
def f (x, y):
    return np.power((x-1),2)*np.power((y-1),2)*np.power((y),2)*np.power((x),2)

@njit(fastmath=True)#Неточное
def fN4 (x, y):
    return 24*np.power((x),2)*(np.power((x-1),2))+24*np.power((y),2)*(np.power((y-1),2))+2*(12*np.power((x),2)-12*x+2)*(12*np.power((y),2)-12*y+2)

@njit(fastmath=True)#Точное
def fT6 (x, y):
    #if 0<x<0.5 and 0<y<0.5:
    if 0.5<x<1 and 0.5<y<1:
        return np.power((x-0.5),2)*np.power((x-1),2)*np.power((y-0.5),2)*np.power((y-1),2)
    else:
        return 0

@njit(fastmath=True)#Неточное
def fN6 (x, y):
    #if 0<x<0.5 and 0<y<0.5:
    if 0.5<x<1 and 0.5<y<1:
        return 6*np.power((2*y-1),2)*np.power((y-1),2)+(24*np.power((x),2)-18*x+13)*(24*np.power((y),2)-18*y+13)/4+6*(np.power((2*x-1),2))*(np.power((x-1),2))
    else:
        return 0

@njit(fastmath=True)
def fPR1 (x, y):
    return 12*(np.power((y-1),2)*np.power((y),2)*(2*x-1)+np.power((x),2)*np.power((x-1),2)*(2*y-1))

@njit(fastmath=True)
def fPR2 (x, y):
    return 18*(np.power((y),4)+np.power((x),4)-(2*np.power((y),3))-(2*np.power((x),3))+x+y)

@njit(fastmath=True)
def fPR3 (x, y):
    return 288*(np.power((x),2)*np.power((y),2)-(y*np.power((x),2))-(x*np.power((y),2))+x*y)+24*(np.power((x),2)+np.power((y),2)-2*x-2*y)+8

@njit(fastmath=True)
def f1 (x, y): 
    return 10

@njit(fastmath=True)
def ft1 (x, y): 
    return (y**2)*np.sin(x)+(x**2)*np.sin(y)

@njit(fastmath=True)
def fn1 (x, y): 
    return (2-y**2)*np.sin(x)+(2-x**2)*np.sin(y)

@njit(fastmath=True)
def ft2 (x, y): 
    return (x**2-1)*(y**2+2)

@njit(fastmath=True)
def fn2 (x, y): 
    return 2*(x**2+y**2+1)

@njit(fastmath=True)
def ft3 (x, y): 
    return np.exp(x)*np.sin(y)

@njit(fastmath=True)
def fn4 (x, y): 
    return 8*np.cos(x+y)**2-4

@njit(fastmath=True)
def ft4 (x, y):
    return np.sin(x+y)**2

@njit(fastmath=True)
def fnn (x, y):
    return -(4*x**2+9)*np.sin(x**2+3)+2*np.cos(x**2+3)

@njit(fastmath=True)
def ftt (x, y):
    return np.sin(x**2+3*y)

@njit(fastmath=True)
def ft5 (x, y):
    return np.sin(x**2+3*y)

@njit(fastmath=True)
def fn6 (x, y):
    return 5/((y+2*x+1)**2*np.log(10))

@njit(fastmath=True)
def ft6 (x, y):
    return np.log(y+2*x+1)

@njit(fastmath=True)
def ft7 (x, y):
    return y**2*np.cos(x)

@njit(fastmath=True)
def fn7 (x, y):
    return -np.cos(x)*(y**2-2)

@njit(fastmath=True)
def fT9 (x, y):
    return np.exp(2*y)*(2*y**2+1+x)

@njit(fastmath=True)
def fN9 (x, y):
    return 4*np.exp(2*y)*(2*y**2+4*y+x+2)


@njit(fastmath=True)
def fT10 (x, y):
    return np.exp(3*x)*np.cos(2*y)+np.exp(2*x)*np.cos(3*y)

@njit(fastmath=True)
def fN10 (x, y):
    return 5*np.exp(3*x)*np.cos(2*y)-5*np.exp(2*x)*np.cos(3*y)


@njit(fastmath=True)
def fn11 (x, y):
    return (9*x-2*x**3+1)*np.cos(y)

@njit(fastmath=True)
def ft11 (x, y):
    return (2*x**3+3*x-1)*np.cos(y)
#-------------------------------------------------------------
#------------Функции заданные на границе----------------------
@njit(fastmath=True)
def Gran_Func(x, y): 
    return 0

@njit(fastmath=True)
def Gran_Func2(x, y): 
    return np.power((x-1),2)*np.power((y-1),2)*np.power((y),2)*np.power((x),2)

@njit(fastmath=True)
def Gran_Func3(x, y): 
    return (x-0.5)**4+(y-0.5)**4

@njit(fastmath=True)
def Gran_Func4(x, y):
    return 24*(x*(x-1))**2+24*(y*(y-1))**2+2*(12*x**2-12*x+2)*(12*y**2-12*y+2)

#-------------------------------------------------------------
#------------Функции задания границы--------------------------

@njit(fastmath=True)
def Search_Granica1(m,h, n,l): #Окружность
    #m = 100
    #n = 100
    #h = 0.01
    #l = 0.01
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if ((i*h)-0.5)**2 + ((j*l)-0.5)**2< 0.19:
                Granica.append((i*h, j*l))
    return Granica

@njit(fastmath=True)
def Search_Granica2(m,h, n,l): #Две Окружности
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if ((j*h)-25)**2 + ((i*l)-15)**2< 150:
                Granica.append((i*h, j*l))
            if ((j*h)-25)**2 + ((i*l)-35)**2< 150:
                Granica.append((i*h, j*l))
    return Granica 

@njit(fastmath=True)
def Search_Granica3(m,h, n,l): #Две Окружности разного размера
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for j in range(n+1):
        for i in range(m+1):
            if ((i*h)-15)**2 + ((j*l)-16)**2< 150:
                Granica.append((i*h, j*l))
            if ((i*h)-30)**2 + ((j*l)-35)**2< 190:
                Granica.append((i*h, j*l))
    return Granica

@njit(fastmath=True)
def Search_Granica4(m,h, n,l): #Четыре Окружности
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if 150  > ((i*h)-15)**2 + ((j*l)-15)**2:
                Granica.append((i*h, j*l))
            elif 150  > ((i*h)-15)**2 + ((j*l)-35)**2:
                Granica.append((i*h, j*l))
            elif 150  > ((i*h)-35)**2 + ((j*l)-15)**2:
                Granica.append((i*h, j*l))
            elif 150  > ((i*h)-35)**2 + ((j*l)-35)**2:
                Granica.append((i*h, j*l))
    return Granica

@njit(fastmath=True)
def Search_Granica6(m, h, n, l): #Парабола Ограниченная Прямой
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Parabola=[]
    a = 9
    for i in range(m+1):
        for j in range(n+1):
            if (i==80 and 21<j<79):
                Parabola.append((i*h, j*l))
                continue
            if (i<80) and (((j*l)-25)**2 - a*(i*h-20)) < 30:
                Parabola.append((i*h, j*l))
    return Parabola

@njit(fastmath=True)
def Search_Granica(m,h, n,l): #Прямоугольник по краям графика
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if  (j>=0 and j<=n):
                Granica.append((i*h, j*l))
            if  (i>=0 and i<=m):
                Granica.append((i*h, j*l))    
    return Granica

@njit(fastmath=True)
def Search_Granica8(m,h, n,l): #Прямоугольник внутри графика с учетом lh
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i*h>=10 and i*h<=30) and (j*l>=10 and j*l<=40):
                Granica.append((i*h, j*l))
    return Granica

@njit(fastmath=True)
def Search_Granica9(m,h, n,l): #Прямоугольник внутри графика без учета lh
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i>=20 and i<=80) and (j>=20 and j<=80):
                Granica.append((i*h, j*l))
    return Granica

#------------------------------------------------------------------
#-------------------------Методы Либмана---------------------------

@njit(fastmath=True)
def Libman1(h, l, coordinata, u0, E): # Пяти точечный метод Либмана(Классический) 
    dobavka = np.zeros((m+1, n+1))
    u1 = u0.copy()
    for i in range(m+1):
        for j in range(n+1):
            dobavka[i][j] = (((h**2)*(l**2))/(2*(h**2+l**2)))*f(i*h, j*l)
    iteration = 0
    max_val = E+1
    while max_val >= E:     
        max_val = 0
        for i in range(1, m):
            for j in range(1, n):
                if ((i* h, j* l) in coordinata):
                    u1[i][j] = (l**2*(u0[i+1][j]+u0[i-1][j]) + h**2*(u0[i][j+1]+u0[i][j-1])) / (2*(h**2+l**2)) - dobavka[i][j]
                    diff = abs(u1[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
        u0 = u1.copy()
        iteration += 1
        print('Iteration:', iteration, '\tError:', max_val)
    return u0, iteration

@njit(fastmath=True)
def Libman2(h, l, coordinata, u0, E): # Пяти точечное Усреднение (В основном для Лапласа и подобных)(исключена f)
    iteration = 0
    u1 = u0.copy()
    max_val = E+1
    while max_val >= E:     
        max_val = 0
        for i in range(1, m):
            for j in range(1, n):
                if ((i* h, j* l) in coordinata):
                    u1[i][j] = (l**2*(u0[i+1][j]+u0[i-1][j]) + h**2*(u0[i][j+1]+u0[i][j-1])) / (2*(h**2+l**2))
                    diff = abs(u1[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
        u0 = u1.copy()
        iteration += 1
        print('Iteration:', iteration, '\tError:', max_val)
    return u0, iteration

@njit(fastmath=True, parallel=True)
def Libman(h, l, coordinata, u0, E): #13ти точечный
    dobavka = np.zeros((m+1, n+1))
    u = u0.copy()
    for i in range(2, m-1):
        for j in range(2, n-1):
            dobavka[i][j] = ((h**4)*f(i*h, j*l))/20
    iteration = 0
    max_val = E+1
    while max_val >= E:     
        max_val = 0
        for i in range (2, m-1):
            for j in range (2, n-1):
                if ((i* h, j* l) in coordinata):
                    u[i][j] = dobavka[i][j]-((u[i+2][j]+u[i-2][j]+u[i][j+2]+u[i][j-2]-8*u[i+1][j]-8*u[i-1][j]-8*u[i][j+1]-8*u[i][j-1]+2*u[i+1][j+1]+2*u[i+1][j-1]+2*u[i-1][j-1]+2*u[i-1][j+1])/20)
                    diff = abs(u[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
        u0 = u.copy()
        iteration += 1
        print('Iteration:', iteration, '\tError:', max_val)
    return u0, iteration

def Libman4(h, l, coordinata, u0, E): #Для отключения метода Либмана
    return u0, 0

#------------------------------------------------------------------------------------------------------
#-------------------------------------------ОСНОВНАЯ ЧАСТЬ---------------------------------------------
#------------------------------------------------------------------------------------------------------
#-------------------------Задание размера сетки и точности--------------------------
#m = 200 #количество узлов сетки по оси х (максимальное значение по оси х / h)
#n = 200 #количество узлов сетки по оси у (максимальное значение по оси у / l)
#h = 0.1 #длина шага по оси х 
#l = 0.1 #длина шага по оси у 
#E = 0.1 #необходимая точность (епсилон)

#a = 50
#b = 50
#m = 100 #количество узлов сетки по оси х (максимальное значение по оси х / h)
#n = 100 #количество узлов сетки по оси у (максимальное значение по оси у / l)
#h = a / m #длина шага по оси х 
#l = b / n #длина шага по оси у 
#E = 0.01 #необходимая точность (епсилон)

a = 1 #длина отрезка
b = 1 #длина отрезка
m = 100 #количество узлов сетки по оси х (максимальное значение по оси х / h)
n = 100 #количество узлов сетки по оси у (максимальное значение по оси у / l)
h = a / m #длина шага по оси х 
l = b / n #длина шага по оси у 
E = 0.0001 #необходимая точность (епсилон)
#--------------------------------------------------------------------
#-------------------Создание массивов--------------------------------
u0 = np.zeros((m+1, n+1))
coordinata=Search_Granica(m,h, n,l)
coordinata_f=[]

##Для 5ти точечного шаблона
#for i in range(m + 1):
#    for j in range(n + 1):
#        if ((i * h,j*l)in coordinata):
#            if ((((i-1) * h,(j)*l)in coordinata) and
#                (((i+1) * h,(j)*l)in coordinata) and
#                (((i) * h,(j-1)*l)in coordinata) and
#                (((i) * h,(j+1)*l)in coordinata)):
#                    u0[i][j] = f(i*h,j*l)
#                    coordinata_f.append((i*h, j*l))
#            else:
#                u0[i][j] = Gran_Func(i*h,j*l)
#        else:
#            u0[i][j] = np.nan

#Для 13ти точечного шаблона
for i in range(m + 1):
    for j in range(n + 1):
        if ((i * h,j*l)in coordinata):
            if ((((i-2) * h,(j)*l)in coordinata) and
                (((i+2) * h,(j)*l)in coordinata) and
                (((i) * h,(j-2)*l)in coordinata) and
                (((i) * h,(j+2)*l)in coordinata)):
                    u0[i][j] = f(i*h,j*l)
                    coordinata_f.append((i*h, j*l))
            else:
                u0[i][j] = Gran_Func(i*h,j*l)
        else:
            u0[i][j] = np.nan


#----------------------------------------------------------------
#--------------------Метод Либмана-------------------------------
u0,iteration=Libman(h, l, coordinata_f, u0, E)
#----------------------------------------------------------------
#--------------Вывод полученных значений узлов сетки-------------
#print("Время выполнения - %s секунд." % (time.time() - start_time), f"Число итераций:{iteration}")
#print('*\t', end='') #Вывод сетки
#for i in range(m+1):
#    print(f'{i:.0f}', end='\t')
#print()
#for j in range(n+1):
#    print(f'{j:.0f}', end='\t')
#    for i in range(m+1):
#        print(f'{u0[i][j]:.3f}', end='\t')
#    print()
#----------------------------------------------------------------
#----------------Построение графика------------------------------
#-----------------------------2D График--------------------------
#print("Время выполнения - %s секунд." % (time.time() - start_time), f"Число итераций:{iteration}")
#plt.imshow(u0, cmap='plasma')
#plt.colorbar()
#plt.gca().invert_yaxis()
#plt.suptitle('Решение задачи Дирихле методом Либмана')
#plt.text(0.5, 1.01, f'Число итераций: {iteration}', transform=plt.gca().transAxes, ha='center')
#plt.show()
#-----------------------------3D График--------------------------
print("Время выполнения - %s секунд." % (time.time() - start_time), f"Число итераций:{iteration}")
u = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
a, b = zip(*coordinata)
for i in range(m + 1):
    for j in range(n + 1):
        u[i][j] = u0[i, j]

z = np.array(u)
x = [h * i for i in range(m + 1)]
y = [l * j for j in range(n + 1)]
x, y = np.meshgrid(x, y)

# Настройка визуализации
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
fig.suptitle(f'Решение задачи Дирихле методом Либмана\nЧисло итераций: {iteration}')
#ax.plot_surface(x, y, z, cmap='plasma') #для красивого градиента
ax.plot_wireframe(x, y, z, color="green") #для зеленой сетки
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
plt.ioff()  

#u0 = u0.tolist()
#with open(r"C:\Users\...", "w") as file:
#    json.dump(u0, file)
#----------------------------------------------------------------
