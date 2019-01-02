from student import Student

students = []
for i in range(3):
    name = input("name: ")
    dorm = input("dorm: ")
    students.append(Student(name, dorm))

for student in students:
    print("{} is in {}.".format(student.name, student.dorm))
