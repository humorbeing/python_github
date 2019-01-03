from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import sqrt



TopLeftX = 0
TopLeftY = 0
BottomRightX = 0
BottomRightY = 0

def MyDisplay():
    glViewport(0, 0, 300, 300)
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(0.5, 0.5, 0.5)
    r = sqrt((TopLeftX-BottomRightX)**2 + (TopLeftY-BottomRightY)**2)
    r = 0.002*r
    print(r)
    glutWireSphere(r, 15, 15)
    glFlush()

def MyMouseClick(Button, State, X, Y):
    global TopLeftX, TopLeftY
    if Button==GLUT_LEFT_BUTTON and State == GLUT_DOWN:
        TopLeftX = X
        TopLeftY = Y   

def MyMouseMove(X, Y):
    global BottomRightX, BottomRightY
    BottomRightX = X
    BottomRightY = Y
    glutPostRedisplay()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Drawing Example")  # not only string, put 'b' in front of string.
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)  #square
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # sphere
    glutDisplayFunc(MyDisplay)
    glutMouseFunc(MyMouseClick)
    glutMotionFunc(MyMouseMove)
    glutMainLoop()

if __name__ == '__main__':
    main()
