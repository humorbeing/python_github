from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def MyDisplay():
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_POLYGON)
    glVertex3f(-0.5, -0.5, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glVertex3f(0.5, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, 0.0)
    glEnd()
    glFlush()


def MyReshape(NewWidth, NewHeight):
    glViewport(0, 0, NewWidth, NewHeight)
    WidthFactor = float(NewWidth/300)  # GLfloat
    HeightFactor = float(NewHeight/300)  # GLfloat
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0*WidthFactor, 1.0*WidthFactor,
            -1.0*HeightFactor, 1.0*HeightFactor,
            -1.0, 1.0)


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Example Drawing")  # not only string, put 'b' in front of string.
    glClearColor(0.0, 0.0, 0.0, 1.0)

    glutDisplayFunc(MyDisplay)
    glutReshapeFunc(MyReshape)
    glutMainLoop()

if __name__ == '__main__':
    main()
