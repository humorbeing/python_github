from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

MyListID = 1


def MyCreateList():
    global MyListID
    MyListID = glGenLists(1)
    glNewList(MyListID, GL_COMPILE)
    glBegin(GL_POLYGON)
    glColor3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glVertex3f(0.5, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, 0.0)
    glEnd()
    glEndList()


def MyDisplay():
    global MyListID
    glClear(GL_COLOR_BUFFER_BIT)
    glViewport(0, 0, 300, 300)
    glCallList(MyListID)
    glFlush()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Example Drawing")  # not only string, put 'b' in front of string.
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0)
    glutDisplayFunc(MyDisplay)
    MyCreateList()
    glutMainLoop()

if __name__ == '__main__':
    main()
