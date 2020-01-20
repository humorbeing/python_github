from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

IsSphere = True


def MyDisplay():
    global IsSphere
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(0.5, 0.5, 0.5)
    if IsSphere:
        glutWireSphere(0.2, 15, 15)
    else:
        glutWireTorus(0.1, 0.3, 40, 20)
    glFlush()


def MyMainMenu(entryID):
    global IsSphere
    if entryID == 1:
        IsSphere = True
    elif entryID == 2:
        IsSphere = False
    elif entryID ==3:
        exit(0)
    glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Example Drawing")  # not only string, put 'b' in front of string.
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    MyMainMenuID = glutCreateMenu(MyMainMenu)
    glutAddMenuEntry('Draw Sphere', 1)
    glutAddMenuEntry('Draw Torus', 2)
    glutAddMenuEntry('Exit', 3)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    glutDisplayFunc(MyDisplay)
    glutMainLoop()

if __name__ == '__main__':
    main()
