from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def MyDisplay():
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_POLYGON)
    glVertex3f(-0.5, -0.5, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glVertex3f(0.5, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, 0.0)
    glEnd()
    glFlush()


def MyKeyboard(*key):
    if key[0] == b'q':
        print('you pressed "q".')
        exit()



def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Example Drawing")  # not only string, put 'b' in front of string.
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(MyDisplay)
    glutKeyboardFunc(MyKeyboard)
    glutMainLoop()

if __name__ == '__main__':
    main()
