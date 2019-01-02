from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

IsSphere = True
IsSmall = True


def MyDisplay():
    global IsSphere, IsSmall
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(0.5, 0.0, 0.5)
    if IsSphere and IsSmall:
        glutWireSphere(0.2, 15, 15)
    elif IsSphere and not IsSmall:
        glutWireSphere(0.4, 15, 15)
    elif not IsSphere and IsSmall:
        glutWireTorus(0.1, 0.3, 40, 20)
    else:
        glutWireTorus(0.1, 0.5, 40, 20)
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
    return 0


def MySubMenu(entryID):
    global IsSmall
    if entryID == 1:
        IsSmall = True
    elif entryID == 2:
        IsSmall = False
    glutPostRedisplay()
    return 0


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
    MySubMenuID = glutCreateMenu(MySubMenu)
    glutAddMenuEntry('Small One', 1)
    glutAddMenuEntry('Big One', 2)
    MyMainMenuID = glutCreateMenu(MyMainMenu)
    glutAddMenuEntry('Draw Sphere', 1)
    glutAddMenuEntry('Draw Torus', 2)
    glutAddSubMenu('Change Size', MySubMenuID)
    glutAddMenuEntry('Exit', 3)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    glutDisplayFunc(MyDisplay)
    glutMainLoop()

if __name__ == '__main__':
    main()
