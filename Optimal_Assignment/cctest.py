import numpy as np#just to formated print
class hungarian:
    def __init__(self):
        with open('assign.in','r') as f:
            a = f.readlines()
        b = a[0].split()
        self.N = int(b[0])
        del a[0]
        self.matrix = []
        self.maxinput = 0
        def findmax(s):
            n = int(s)
            if n>self.maxinput:
                self.maxinput = n
            return n
        for i in a:
            self.matrix.append([findmax(j) for j in i.split()])
        print('INIT--'*5)
        print('/'*79)
        print('\\'*79)
        print('self.N: {}'.format(self.N))
        print('self.matrix:\n {}'.format(np.array(self.matrix)))
        print('*'*79)
        print('self.maxinput: {}'.format(self.maxinput))
        self.minimum_value_matching_matrix = [[(self.maxinput - element) for element in row] for row in self.matrix]
        print('self.mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        print('*'*79)
        print('did self.matrix change:\n {}'.format(np.array(self.matrix)))
        print(' --INIT END-- '*5)
        print('/'*79)
        print('\\'*79)
    def step_one(self):
        #min_value_in_row = 0
        print('-in step 11111111111111111111111111111111')
        print('-old mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        for i in range(self.N):
            min_value_in_row = min(self.minimum_value_matching_matrix[i])
            print('-min value in this row: {} is {}'.format(self.minimum_value_matching_matrix[i],min_value_in_row))
            #print('-subtracting mini value on each row.')
            self.minimum_value_matching_matrix[i] = [(element - min_value_in_row) for element in self.minimum_value_matching_matrix[i]]

        print('-new mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        print('-did step 1 change self.matrix?:\n {}'.format(np.array(self.matrix)))
        print('-END step 1111111111111111111111111111111111111')
        self.step_two()

    def step_two(self):
        print('--in step 2222222222222222222222222222222')
        print('--old mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))


        for i in range(self.N):

            minimum_value_in_column = min([j[i] for j in self.minimum_value_matching_matrix])
            print('--the minimum value in {} column {} is {}'.format(i,[j[i] for j in self.minimum_value_matching_matrix],minimum_value_in_column))
            for j in range(self.N):
                self.minimum_value_matching_matrix[j][i] = self.minimum_value_matching_matrix[j][i] - minimum_value_in_column



        print('--new mini matrix:\n {}'.format(np.array(self.minimum_value_matching_matrix)))
        print('--did step 2 change self.matrix?:\n {}'.format(np.array(self.matrix)))
        print('--END step 2222222222222222222222222222222222')

    def test(self):
        #print(self.minimum_value_matching_matrix[::][1])
        pass
optimal_assingment = hungarian()
optimal_assingment.step_one()
#optimal_assingment.test()
