from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

Delta = 0.0


def MyDisplay():
    global Delta
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_POLYGON)
    glColor3f(0.0, 0.5, 0.8)

    glVertex3f(-1.0 + Delta, -0.5, 0.0)
    glVertex3f(0.0 + Delta, -0.5, 0.0)
    glVertex3f(0.0 + Delta, 0.5, 0.0)
    glVertex3f(-1.0 + Delta, 0.5, 0.0)
    glEnd()
    glutSwapBuffers()


def MyIdle():
    global Delta
    Delta += 0.001
    glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Drawing Example")  # not only string, put 'b' in front of string.
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0)
    glutDisplayFunc(MyDisplay)
    glutIdleFunc(MyIdle)
    glutMainLoop()

if __name__ == '__main__':
    main()
