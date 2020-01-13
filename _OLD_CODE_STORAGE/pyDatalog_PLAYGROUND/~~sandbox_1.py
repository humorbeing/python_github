from pyDatalog_playground import pyDatalog


pyDatalog.create_terms('factorial, N')

factorial[N] = N * factorial[N - 1]
factorial[1] = 1
print(factorial[3] == N)
print(' -'*20)

pyDatalog.create_terms('X,Y')
print(X==1)
print((X==True) & (Y==False))
# print((X==input('Please enter your name : ')) & (Y=='Hello ' + X[0]))
print((Y==1) & (Y==X+1))
print((X==(1,2)+(3,)) & (Y==X[2]))
print('*'*20)


def twice(a):
    return a+a


pyDatalog.create_terms('twice')
print((X==1) & (Y==twice(X)))


import math
pyDatalog.create_terms('math')
print((X==2) & (Y==math.sqrt(X)))

pyDatalog.create_terms('X,Y,Z')

print(X.in_((0,1,2,3,4)))

print(X.in_(range(5)).data)
print(X.in_(range(5)) == set([(0,), (1,), (2,), (3,), (4,)]))

print("Data : ",X.data)
print("First value : ",  X.v())
# below, '>=' is a variable extraction operator
print("Extraction of first value of X: ", X.in_(range(5)) >= X)

print(X.in_(range(5)) & (X<2))

print(X.in_(range(5)) &
          Y.in_(range(5)) &
              (Z==X+Y) &
              (Z<3))

pyDatalog.create_terms('X,Y,Z, salary, tax_rate, tax_rate_for_salary_above, net_salary')


salary['foo'] = 60
salary['bar'] = 110

print(salary[X]==Y)

salary['foo'] = 70
print(salary['foo']==Y)

print(salary[X]==110)

print((salary[X]==Y) & ~(Y==110))

+(tax_rate[None]==0.33)

print((Z==salary[X]*(1-tax_rate[None])))
print((Z==salary[X]*(1-tax_rate[None])) & (Y==salary[X]))


net_salary[X] = salary[X]*(1-tax_rate[None])
print(net_salary[X]==Y)

print(net_salary['foo']==Y)


(tax_rate_for_salary_above[X] == 0.33) <= (0 <= X)
(tax_rate_for_salary_above[X] == 0.50) <= (100 <= X)
print(tax_rate_for_salary_above[70]==Y)
# print
print(tax_rate_for_salary_above[150]==Y)

(tax_rate_for_salary_above[X] == 0.33) <= (X >= 0)
(tax_rate_for_salary_above[X] == 0.50) <= (X >= 100)
print(tax_rate_for_salary_above[70]==Y)
# print
print(tax_rate_for_salary_above[150]==Y)

del net_salary[X]

net_salary[X] = salary[X]*(1-tax_rate_for_salary_above[salary[X]])
# give me all X and Y so that Y is the net salary of X
print(net_salary[X]==Y)
print(Y==net_salary[X])


pyDatalog.create_terms('X,Y,manager, count_of_direct_reports')
+(manager['Mary'] == 'John')
+(manager['Sam']  == 'Mary')
+(manager['Tom']  == 'Mary')

(count_of_direct_reports[X]==len_(Y)) <= (manager[Y]==X)
print(count_of_direct_reports['Mary']==Y)


pyDatalog.create_terms('X,Y,Z, works_in, department_size, manager, indirect_manager, count_of_indirect_reports')

+ works_in('Mary', 'Production')
+ works_in('Sam',  'Marketing')

+ works_in('John', 'Production')
+ works_in('John', 'Marketing')

print(works_in(X,  'Marketing'))

indirect_manager(X,Y) <= (manager[X] == Y)

indirect_manager(X,Y) <= (manager[X] == Z) & indirect_manager(Z,Y)
print(indirect_manager('Sam',X))


manager['John'] = 'Mary'

print(indirect_manager('John',X))

- works_in('John', 'Production')

(count_of_indirect_reports[X]==len_(Y)) <= indirect_manager(Y,X)
print(count_of_indirect_reports['John']==Y)



pyDatalog.create_terms('link, can_reach')

# there is a link between node 1 and node 2
+link(1,2)
+link(2,3)
+link(2,4)
+link(2,5)
+link(5,6)
+link(6,7)
+link(7,2)

link(X,Y) <= link(Y,X)


can_reach(X,Y) <= link(X,Y) # direct link
# via Z
can_reach(X,Y) <= link(X,Z) & can_reach(Z,Y) & (X!=Y)

print (can_reach(1,Y))

print(' -'*20)
pyDatalog.create_terms('ok,print')
ok(X) <= (3 < X) & (Y==print(X))

print(X.in_(range(5)) & ok(1))