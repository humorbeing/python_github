from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def MyDisplay():
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_POLYGON)
    glVertex3f(-0.5, -0.5, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glVertex3f(0.5, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, 0.0)
    glEnd()
    glFlush()


def main():
    glutInit()
    glutCreateWindow(b"OpenGL Drawing Example")  # not only string, put 'b' in front of string.
    glutDisplayFunc(MyDisplay)
    glutMainLoop()

if __name__ == '__main__':
    main()
