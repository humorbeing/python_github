from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import sqrt

Draw_Option = 1
IsSmall = True

TopLeftX = 0
TopLeftY = 0
BottomRightX = 0
BottomRightY = 0

def MyDisplay():
    global Draw_Option, IsSmall
    glClear(GL_COLOR_BUFFER_BIT)
    # glViewport(0, 0, 300, 300)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if Draw_Option == 3:
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    else:
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    # glMatrixMode(GL_MODELVIEW)
    glColor3f(0.5, 0.0, 0.5)
    if Draw_Option==1 and IsSmall:
        glutWireSphere(0.2, 15, 15)
    elif Draw_Option==1 and not IsSmall:
        glutWireSphere(0.4, 15, 15)
    elif Draw_Option==2 and IsSmall:
        glutWireTorus(0.1, 0.3, 40, 20)
    elif Draw_Option==2 and not IsSmall:
        glutWireTorus(0.1, 0.5, 40, 20)
    elif Draw_Option==3:
        DrawRec()
    elif Draw_Option==4:
        r = sqrt((TopLeftX - BottomRightX) ** 2 + (TopLeftY - BottomRightY) ** 2)
        r = 0.002 * r
        glutSolidSphere(r, 15, 15)
    elif Draw_Option==5:
        r = sqrt((TopLeftX - BottomRightX) ** 2 + (TopLeftY - BottomRightY) ** 2)
        r = 0.002 * r
        glutSolidTorus(0.1, r, 40, 20)
    glFlush()


def DrawRec():
    glViewport(0, 0, 300, 300)
    glBegin(GL_POLYGON)
    glVertex3f(TopLeftX / 300.0, (300 - TopLeftY) / 300.0, 0.0)
    glVertex3f(TopLeftX / 300.0, (300 - BottomRightY) / 300.0, 0.0)
    glVertex3f(BottomRightX / 300.0, (300 - BottomRightY) / 300.0, 0.0)
    glVertex3f(BottomRightX / 300.0, (300 - TopLeftY) / 300.0, 0.0)
    glEnd()


def MyMainMenu(entryID):
    global Draw_Option
    if entryID == 0:
        exit(0)
    else:
        Draw_Option = entryID
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


def MyMouseClick(Button, State, X, Y):
    global TopLeftX, TopLeftY
    if Draw_Option!=1 and Draw_Option!=2:
        if Button==GLUT_LEFT_BUTTON and State == GLUT_DOWN:
            TopLeftX = X
            TopLeftY = Y


def MyMouseMove(X, Y):
    global BottomRightX, BottomRightY
    if Draw_Option != 1 and Draw_Option != 2:
        BottomRightX = X
        BottomRightY = Y
        glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(500, 300)
    glutCreateWindow(b"OpenGL Example Drawing")  # not only string, put 'b' in front of string.
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    # glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    MySubMenuID = glutCreateMenu(MySubMenu)
    glutAddMenuEntry('Small One', 1)
    glutAddMenuEntry('Big One', 2)
    MyMainMenuID = glutCreateMenu(MyMainMenu)
    glutAddMenuEntry('Draw Sphere', 1)
    glutAddMenuEntry('Draw Torus', 2)
    glutAddMenuEntry('Draw Rectangle with Mouse', 3)
    glutAddMenuEntry('Draw Sphere with Mouse', 4)
    glutAddMenuEntry('Draw Torus with Mouse', 5)
    glutAddSubMenu('Change Size', MySubMenuID)
    glutAddMenuEntry('Exit', 0)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    glutDisplayFunc(MyDisplay)
    glutMouseFunc(MyMouseClick)
    glutMotionFunc(MyMouseMove)
    glutMainLoop()

if __name__ == '__main__':
    main()
